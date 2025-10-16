#ifndef KERNEL_INFO_INCLUDED
#define KERNEL_INFO_INCLUDED

#include <assert.h>
#include <list>
#include <map>
#include <string>

#ifdef FLASH_GPGPU_SIM_OMP
#include <mutex>
#include <shared_mutex>
#endif

// Forward declarations
class function_info;
class ptx_thread_info;
class memory_space;
struct CUstream_st;
struct cudaArray;
struct textureInfo;

#if !defined(__VECTOR_TYPES_H__)
#include "vector_types.h"
#endif
struct dim3comp {
  bool operator()(const dim3 &a, const dim3 &b) const {
    if (a.z < b.z)
      return true;
    else if (a.y < b.y)
      return true;
    else if (a.x < b.x)
      return true;
    else
      return false;
  }
};

// Function declaration
void increment_x_then_y_then_z(dim3 &i, const dim3 &bound);

/**
 * ! This is shared among all simulator threads and causes race conditions.
 * ! So far, the issue_block2core() is not in parallel region, so CTA allocation
 * ! is fine. However, thread exiting has race conditions as the call chain:
 * !
 * ! gpgpu_sim::cycle() ->
 * ! // Parallel region starts
 * ! simt_core_cluster::core_cycle() ->
 * ! shader_core_ctx::cycle() ->
 * ! shader_core_ctx::fetch() ->
 * ! shader_core_ctx::register_cta_thread_exit() ->
 * ! kernel->dec_running()
 *
 * ! Our solution so far is to use a mutex in dec_running()...
 */
class kernel_info_t {
public:
  kernel_info_t(dim3 gridDim, dim3 blockDim, class function_info *entry,
                unsigned long long streamID);
  kernel_info_t(
      dim3 gridDim, dim3 blockDim, class function_info *entry,
      std::map<std::string, const struct cudaArray *> nameToCudaArray,
      std::map<std::string, const struct textureInfo *> nameToTextureInfo);
  ~kernel_info_t();

  void inc_running() { m_num_cores_running++; }
  void dec_running() {
#ifdef FLASH_GPGPU_SIM_OMP
    std::unique_lock<std::shared_mutex> lock(m_mutex);
#endif
    assert(m_num_cores_running > 0);
    m_num_cores_running--;
  }
  bool running() const { return m_num_cores_running > 0; }
  bool done() const { return no_more_ctas_to_run() && !running(); }
  class function_info *entry() { return m_kernel_entry; }
  const class function_info *entry() const { return m_kernel_entry; }

  size_t num_blocks() const {
    return m_grid_dim.x * m_grid_dim.y * m_grid_dim.z;
  }

  size_t threads_per_cta() const {
    return m_block_dim.x * m_block_dim.y * m_block_dim.z;
  }

  dim3 get_grid_dim() const { return m_grid_dim; }
  dim3 get_cta_dim() const { return m_block_dim; }

  void increment_cta_id() {
    increment_x_then_y_then_z(m_next_cta, m_grid_dim);
    m_next_tid.x = 0;
    m_next_tid.y = 0;
    m_next_tid.z = 0;
  }
  dim3 get_next_cta_id() const { return m_next_cta; }
  unsigned get_next_cta_id_single() const {
    return m_next_cta.x + m_grid_dim.x * m_next_cta.y +
           m_grid_dim.x * m_grid_dim.y * m_next_cta.z;
  }
  bool no_more_ctas_to_run() const {
    return (m_next_cta.x >= m_grid_dim.x || m_next_cta.y >= m_grid_dim.y ||
            m_next_cta.z >= m_grid_dim.z);
  }

  void increment_thread_id() {
    increment_x_then_y_then_z(m_next_tid, m_block_dim);
  }
  dim3 get_next_thread_id_3d() const { return m_next_tid; }
  unsigned get_next_thread_id() const {
    return m_next_tid.x + m_block_dim.x * m_next_tid.y +
           m_block_dim.x * m_block_dim.y * m_next_tid.z;
  }
  bool more_threads_in_cta() const {
    return m_next_tid.z < m_block_dim.z && m_next_tid.y < m_block_dim.y &&
           m_next_tid.x < m_block_dim.x;
  }
  unsigned get_uid() const { return m_uid; }
  unsigned long long get_streamID() const { return m_streamID; }
  std::string get_name() const { return name(); }
  std::string name() const;

  std::list<class ptx_thread_info *> &active_threads() {
    return m_active_threads;
  }
  class memory_space *get_param_memory() { return m_param_mem; }

  // The following functions access texture bindings present at the kernel's
  // launch

  const struct cudaArray *get_texarray(const std::string &texname) const {
    std::map<std::string, const struct cudaArray *>::const_iterator t =
        m_NameToCudaArray.find(texname);
    assert(t != m_NameToCudaArray.end());
    return t->second;
  }

  const struct textureInfo *get_texinfo(const std::string &texname) const {
    std::map<std::string, const struct textureInfo *>::const_iterator t =
        m_NameToTextureInfo.find(texname);
    assert(t != m_NameToTextureInfo.end());
    return t->second;
  }

private:
  kernel_info_t(const kernel_info_t &);  // disable copy constructor
  void operator=(const kernel_info_t &); // disable copy operator

  class function_info *m_kernel_entry;

  unsigned m_uid; // Kernel ID
  unsigned long long m_streamID;

  // These maps contain the snapshot of the texture mappings at kernel launch
  std::map<std::string, const struct cudaArray *> m_NameToCudaArray;
  std::map<std::string, const struct textureInfo *> m_NameToTextureInfo;

  dim3 m_grid_dim;
  dim3 m_block_dim;
  dim3 m_next_cta;
  dim3 m_next_tid;

  unsigned m_num_cores_running;

  std::list<class ptx_thread_info *> m_active_threads;
  class memory_space *m_param_mem;

#ifdef FLASH_GPGPU_SIM_OMP
  mutable std::shared_mutex m_mutex;
#endif

public:
  // Jin: parent and child kernel management for CDP
  void set_parent(kernel_info_t *parent, dim3 parent_ctaid, dim3 parent_tid);
  void set_child(kernel_info_t *child);
  void remove_child(kernel_info_t *child);
  bool is_finished();
  bool children_all_finished();
  void notify_parent_finished();
  CUstream_st *create_stream_cta(dim3 ctaid);
  CUstream_st *get_default_stream_cta(dim3 ctaid);
  bool cta_has_stream(dim3 ctaid, CUstream_st *stream);
  void destroy_cta_streams();
  void print_parent_info();
  kernel_info_t *get_parent() { return m_parent_kernel; }

private:
  kernel_info_t *m_parent_kernel;
  dim3 m_parent_ctaid;
  dim3 m_parent_tid;
  std::list<kernel_info_t *> m_child_kernels; // child kernel launched
  std::map<dim3, std::list<CUstream_st *>, dim3comp>
      m_cta_streams; // streams created in each CTA

  // Jin: kernel timing
public:
  unsigned long long launch_cycle;
  unsigned long long start_cycle;
  unsigned long long end_cycle;
  unsigned m_launch_latency;

  mutable bool cache_config_set;

  unsigned m_kernel_TB_latency; // this used for any CPU-GPU kernel latency and
                                // counted in the gpu_cycle
};

#endif // #ifndef KERNEL_INFO_INCLUDED