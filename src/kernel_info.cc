
#include "kernel_info.h"
#include <algorithm>
#include <cstdio>
#include "../libcuda/gpgpu_context.h"
#include "cuda-sim/memory.h"
#include "cuda-sim/ptx_ir.h"
#include "gpgpu-sim/gpu-sim.h"
#include "stream_manager.h"

void increment_x_then_y_then_z(dim3 &i, const dim3 &bound) {
  i.x++;
  if (i.x >= bound.x) {
    i.x = 0;
    i.y++;
    if (i.y >= bound.y) {
      i.y = 0;
      if (i.z < bound.z) i.z++;
    }
  }
}

kernel_info_t::kernel_info_t(dim3 gridDim, dim3 blockDim,
                             class function_info *entry,
                             unsigned long long streamID) {
  m_kernel_entry = entry;
  m_grid_dim = gridDim;
  m_block_dim = blockDim;
  m_next_cta.x = 0;
  m_next_cta.y = 0;
  m_next_cta.z = 0;
  m_next_tid = m_next_cta;
  m_num_cores_running = 0;
  m_uid = (entry->gpgpu_ctx->kernel_info_m_next_uid)++;
  m_streamID = streamID;
  m_param_mem = new memory_space_impl<8192>("param", 64 * 1024);

  // Jin: parent and child kernel management for CDP
  m_parent_kernel = NULL;

  // Jin: launch latency management
  m_launch_latency = entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency;

  m_kernel_TB_latency =
      entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency +
      num_blocks() * entry->gpgpu_ctx->device_runtime->g_TB_launch_latency;

  cache_config_set = false;
  m_dynamic_smem = 0;
}

/*A snapshot of the texture mappings needs to be stored in the kernel's info as
kernels should use the texture bindings seen at the time of launch and textures
 can be bound/unbound asynchronously with respect to streams. */
kernel_info_t::kernel_info_t(
    dim3 gridDim, dim3 blockDim, class function_info *entry,
    std::map<std::string, const struct cudaArray *> nameToCudaArray,
    std::map<std::string, const struct textureInfo *> nameToTextureInfo) {
  m_kernel_entry = entry;
  m_grid_dim = gridDim;
  m_block_dim = blockDim;
  m_next_cta.x = 0;
  m_next_cta.y = 0;
  m_next_cta.z = 0;
  m_next_tid = m_next_cta;
  m_num_cores_running = 0;
  m_uid = (entry->gpgpu_ctx->kernel_info_m_next_uid)++;
  m_param_mem = new memory_space_impl<8192>("param", 64 * 1024);

  // Jin: parent and child kernel management for CDP
  m_parent_kernel = NULL;

  // Jin: launch latency management
  m_launch_latency = entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency;

  m_kernel_TB_latency =
      entry->gpgpu_ctx->device_runtime->g_kernel_launch_latency +
      num_blocks() * entry->gpgpu_ctx->device_runtime->g_TB_launch_latency;

  cache_config_set = false;
  m_NameToCudaArray = nameToCudaArray;
  m_NameToTextureInfo = nameToTextureInfo;
  m_dynamic_smem = 0;
}

kernel_info_t::~kernel_info_t() {
  assert(m_active_threads.empty());
  destroy_cta_streams();
  delete m_param_mem;
}

std::string kernel_info_t::name() const { return m_kernel_entry->get_name(); }

// Jin: parent and child kernel management for CDP
void kernel_info_t::set_parent(kernel_info_t *parent, dim3 parent_ctaid,
                               dim3 parent_tid) {
  m_parent_kernel = parent;
  m_parent_ctaid = parent_ctaid;
  m_parent_tid = parent_tid;
  parent->set_child(this);
}

void kernel_info_t::set_child(kernel_info_t *child) {
  m_child_kernels.push_back(child);
}

void kernel_info_t::remove_child(kernel_info_t *child) {
  assert(std::find(m_child_kernels.begin(), m_child_kernels.end(), child) !=
         m_child_kernels.end());
  m_child_kernels.remove(child);
}

bool kernel_info_t::is_finished() {
  if (done() && children_all_finished())
    return true;
  else
    return false;
}

bool kernel_info_t::children_all_finished() {
  if (!m_child_kernels.empty()) return false;

  return true;
}

void kernel_info_t::notify_parent_finished() {
  if (m_parent_kernel) {
    m_kernel_entry->gpgpu_ctx->device_runtime->g_total_param_size -=
        ((m_kernel_entry->get_args_aligned_size() + 255) / 256 * 256);
    m_parent_kernel->remove_child(this);
    m_kernel_entry->gpgpu_ctx->the_gpgpusim->g_stream_manager
        ->register_finished_kernel(m_parent_kernel->get_uid());
  }
}

CUstream_st *kernel_info_t::create_stream_cta(dim3 ctaid) {
  assert(get_default_stream_cta(ctaid));
  CUstream_st *stream = new CUstream_st();
  m_kernel_entry->gpgpu_ctx->the_gpgpusim->g_stream_manager->add_stream(stream);
  assert(m_cta_streams.find(ctaid) != m_cta_streams.end());
  assert(m_cta_streams[ctaid].size() >= 1);  // must have default stream
  m_cta_streams[ctaid].push_back(stream);

  return stream;
}

CUstream_st *kernel_info_t::get_default_stream_cta(dim3 ctaid) {
  if (m_cta_streams.find(ctaid) != m_cta_streams.end()) {
    assert(m_cta_streams[ctaid].size() >=
           1);  // already created, must have default stream
    return *(m_cta_streams[ctaid].begin());
  } else {
    m_cta_streams[ctaid] = std::list<CUstream_st *>();
    CUstream_st *stream = new CUstream_st();
    m_kernel_entry->gpgpu_ctx->the_gpgpusim->g_stream_manager->add_stream(
        stream);
    m_cta_streams[ctaid].push_back(stream);
    return stream;
  }
}

bool kernel_info_t::cta_has_stream(dim3 ctaid, CUstream_st *stream) {
  if (m_cta_streams.find(ctaid) == m_cta_streams.end()) return false;

  std::list<CUstream_st *> &stream_list = m_cta_streams[ctaid];
  if (std::find(stream_list.begin(), stream_list.end(), stream) ==
      stream_list.end())
    return false;
  else
    return true;
}

void kernel_info_t::print_parent_info() {
  if (m_parent_kernel) {
    printf("Parent %d: \'%s\', Block (%d, %d, %d), Thread (%d, %d, %d)\n",
           m_parent_kernel->get_uid(), m_parent_kernel->name().c_str(),
           m_parent_ctaid.x, m_parent_ctaid.y, m_parent_ctaid.z, m_parent_tid.x,
           m_parent_tid.y, m_parent_tid.z);
  }
}

void kernel_info_t::destroy_cta_streams() {
  printf("Destroy streams for kernel %d: ", get_uid());
  size_t stream_size = 0;
  for (auto s = m_cta_streams.begin(); s != m_cta_streams.end(); s++) {
    stream_size += s->second.size();
    for (auto ss = s->second.begin(); ss != s->second.end(); ss++)
      m_kernel_entry->gpgpu_ctx->the_gpgpusim->g_stream_manager->destroy_stream(
          *ss);
    s->second.clear();
  }
  printf("size %lu\n", stream_size);
  m_cta_streams.clear();
}