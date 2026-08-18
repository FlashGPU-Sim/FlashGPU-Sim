#include "tma.h"
#include "tensormap.h"
#include "tma_opaque_tensormap.h"
#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <utility>

#include "../cuda-sim/ptx_ir.h"
#include "../gpu-sim.h"
#include "../shader.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../trace.h"
#include "ptx.tab.h"

#include <vector>

// Atomic because this UID counter is shared across all SM instances (which may
// run on different OpenMP threads in Flash mode).  The per-SM tma_unit_impl_t
// members are not shared and therefore do not require synchronization.
std::atomic<unsigned int> tma_next_tx_uid = 0;

namespace flash_gpgpu_sim {

static std::atomic<unsigned long long> g_tma_tx_started{0};
static std::atomic<unsigned long long> g_tma_read_tx_started{0};
static std::atomic<unsigned long long> g_tma_write_tx_started{0};
static std::atomic<unsigned long long> g_tma_tx_completed{0};
static std::atomic<unsigned long long> g_tma_read_tx_completed{0};
static std::atomic<unsigned long long> g_tma_write_tx_completed{0};
static std::atomic<unsigned long long> g_tma_mf_issued{0};
static std::atomic<unsigned long long> g_tma_read_mf_issued{0};
static std::atomic<unsigned long long> g_tma_write_mf_issued{0};
static std::atomic<unsigned long long> g_tma_mf_responses{0};
static std::atomic<unsigned long long> g_tma_read_mf_responses{0};
static std::atomic<unsigned long long> g_tma_write_mf_responses{0};
static std::atomic<unsigned long long> g_tma_bytes_issued{0};
static std::atomic<unsigned long long> g_tma_bytes_completed{0};

static std::atomic<unsigned long long> g_cp_async_tx_started{0};
static std::atomic<unsigned long long> g_cp_async_tx_completed{0};
static std::atomic<unsigned long long> g_cp_async_mf_issued{0};
static std::atomic<unsigned long long> g_cp_async_mf_responses{0};
static std::atomic<unsigned long long> g_cp_async_bytes_issued{0};
static std::atomic<unsigned long long> g_cp_async_bytes_completed{0};
static std::atomic<unsigned long long> g_cp_async_issue_queue_cycles{0};
static std::atomic<unsigned long long> g_cp_async_issue_active_cycles{0};
static std::atomic<unsigned long long> g_cp_async_issue_width_limited_cycles{0};
static std::atomic<unsigned long long> g_cp_async_issue_blocked_inflight_cycles{
    0};
static std::atomic<unsigned long long> g_cp_async_issue_blocked_icnt_cycles{0};
static std::atomic<unsigned long long> g_cp_async_max_issue_queue{0};
static std::atomic<unsigned long long> g_cp_async_max_inflight{0};
static std::atomic<unsigned long long> g_cp_async_wait_calls{0};
static std::atomic<unsigned long long> g_cp_async_wait_immediate{0};
static std::atomic<unsigned long long> g_cp_async_wait_blocked{0};
static std::atomic<unsigned long long> g_cp_async_wait_releases{0};
static std::atomic<unsigned long long> g_cp_async_waiting_warp_cycles{0};
static std::atomic<unsigned long long> g_cp_async_response_fifo_nonempty_cycles{
    0};
static std::atomic<unsigned long long> g_cp_async_response_width_limited_cycles{
    0};
static std::atomic<unsigned long long> g_cp_async_max_response_fifo{0};

static void atomic_update_max(std::atomic<unsigned long long> &target,
                              unsigned long long value) {
  unsigned long long current = target.load(std::memory_order_relaxed);
  while (current < value && !target.compare_exchange_weak(
                                current, value, std::memory_order_relaxed,
                                std::memory_order_relaxed)) {
  }
}

tma_progress_counters_t get_global_tma_progress_counters() {
  tma_progress_counters_t counters;
  counters.tx_started = g_tma_tx_started.load(std::memory_order_relaxed);
  counters.read_tx_started =
      g_tma_read_tx_started.load(std::memory_order_relaxed);
  counters.write_tx_started =
      g_tma_write_tx_started.load(std::memory_order_relaxed);
  counters.tx_completed = g_tma_tx_completed.load(std::memory_order_relaxed);
  counters.read_tx_completed =
      g_tma_read_tx_completed.load(std::memory_order_relaxed);
  counters.write_tx_completed =
      g_tma_write_tx_completed.load(std::memory_order_relaxed);
  counters.mf_issued = g_tma_mf_issued.load(std::memory_order_relaxed);
  counters.read_mf_issued =
      g_tma_read_mf_issued.load(std::memory_order_relaxed);
  counters.write_mf_issued =
      g_tma_write_mf_issued.load(std::memory_order_relaxed);
  counters.mf_responses = g_tma_mf_responses.load(std::memory_order_relaxed);
  counters.read_mf_responses =
      g_tma_read_mf_responses.load(std::memory_order_relaxed);
  counters.write_mf_responses =
      g_tma_write_mf_responses.load(std::memory_order_relaxed);
  counters.bytes_issued = g_tma_bytes_issued.load(std::memory_order_relaxed);
  counters.bytes_completed =
      g_tma_bytes_completed.load(std::memory_order_relaxed);
  return counters;
}

cp_async_debug_counters_t get_global_cp_async_debug_counters() {
  cp_async_debug_counters_t counters;
  counters.tx_started = g_cp_async_tx_started.load(std::memory_order_relaxed);
  counters.tx_completed =
      g_cp_async_tx_completed.load(std::memory_order_relaxed);
  counters.mf_issued = g_cp_async_mf_issued.load(std::memory_order_relaxed);
  counters.mf_responses =
      g_cp_async_mf_responses.load(std::memory_order_relaxed);
  counters.bytes_issued =
      g_cp_async_bytes_issued.load(std::memory_order_relaxed);
  counters.bytes_completed =
      g_cp_async_bytes_completed.load(std::memory_order_relaxed);
  counters.issue_queue_cycles =
      g_cp_async_issue_queue_cycles.load(std::memory_order_relaxed);
  counters.issue_active_cycles =
      g_cp_async_issue_active_cycles.load(std::memory_order_relaxed);
  counters.issue_width_limited_cycles =
      g_cp_async_issue_width_limited_cycles.load(std::memory_order_relaxed);
  counters.issue_blocked_inflight_cycles =
      g_cp_async_issue_blocked_inflight_cycles.load(std::memory_order_relaxed);
  counters.issue_blocked_icnt_cycles =
      g_cp_async_issue_blocked_icnt_cycles.load(std::memory_order_relaxed);
  counters.max_issue_queue =
      g_cp_async_max_issue_queue.load(std::memory_order_relaxed);
  counters.max_inflight =
      g_cp_async_max_inflight.load(std::memory_order_relaxed);
  counters.wait_calls = g_cp_async_wait_calls.load(std::memory_order_relaxed);
  counters.wait_immediate =
      g_cp_async_wait_immediate.load(std::memory_order_relaxed);
  counters.wait_blocked =
      g_cp_async_wait_blocked.load(std::memory_order_relaxed);
  counters.wait_releases =
      g_cp_async_wait_releases.load(std::memory_order_relaxed);
  counters.waiting_warp_cycles =
      g_cp_async_waiting_warp_cycles.load(std::memory_order_relaxed);
  counters.response_fifo_nonempty_cycles =
      g_cp_async_response_fifo_nonempty_cycles.load(std::memory_order_relaxed);
  counters.response_width_limited_cycles =
      g_cp_async_response_width_limited_cycles.load(std::memory_order_relaxed);
  counters.max_response_fifo =
      g_cp_async_max_response_fifo.load(std::memory_order_relaxed);
  return counters;
}

static void record_tma_tx_started(bool is_write) {
  g_tma_tx_started.fetch_add(1, std::memory_order_relaxed);
  if (is_write) {
    g_tma_write_tx_started.fetch_add(1, std::memory_order_relaxed);
  } else {
    g_tma_read_tx_started.fetch_add(1, std::memory_order_relaxed);
  }
}

static void record_tma_tx_completed(bool is_write) {
  g_tma_tx_completed.fetch_add(1, std::memory_order_relaxed);
  if (is_write) {
    g_tma_write_tx_completed.fetch_add(1, std::memory_order_relaxed);
  } else {
    g_tma_read_tx_completed.fetch_add(1, std::memory_order_relaxed);
  }
}

static void record_tma_mf_issued(bool is_write, unsigned bytes) {
  g_tma_mf_issued.fetch_add(1, std::memory_order_relaxed);
  g_tma_bytes_issued.fetch_add(bytes, std::memory_order_relaxed);
  if (is_write) {
    g_tma_write_mf_issued.fetch_add(1, std::memory_order_relaxed);
  } else {
    g_tma_read_mf_issued.fetch_add(1, std::memory_order_relaxed);
  }
}

static void record_tma_mf_response(bool is_write, unsigned bytes) {
  g_tma_mf_responses.fetch_add(1, std::memory_order_relaxed);
  g_tma_bytes_completed.fetch_add(bytes, std::memory_order_relaxed);
  if (is_write) {
    g_tma_write_mf_responses.fetch_add(1, std::memory_order_relaxed);
  } else {
    g_tma_read_mf_responses.fetch_add(1, std::memory_order_relaxed);
  }
}

static void record_tma_bytes_completed(unsigned bytes) {
  g_tma_bytes_completed.fetch_add(bytes, std::memory_order_relaxed);
}

static void tma_trace_emit(unsigned long long cycle, const char *event,
                           unsigned tx_uid, const char *kind, unsigned tma_type,
                           unsigned long long pc, unsigned cta_id,
                           unsigned warp_id, unsigned lane_id, unsigned tid,
                           unsigned long long src, unsigned long long dst,
                           unsigned size, unsigned mbar, unsigned mf_uid,
                           unsigned long long mf_addr, unsigned mf_size,
                           unsigned issued_mf, unsigned received_mf,
                           unsigned bytes_completed, unsigned tx_inflight,
                           unsigned global_inflight, unsigned response_fifo) {
  const char *path = std::getenv("FLASHGPU_TMA_TRACE_CSV");
  if (path == nullptr || path[0] == '\0')
    return;

  const char *limit = std::getenv("FLASHGPU_TMA_TRACE_CYCLE_LIMIT");
  if (limit != nullptr && limit[0] != '\0') {
    unsigned long long limit_cycle = std::strtoull(limit, nullptr, 0);
    if (limit_cycle != 0 && cycle > limit_cycle)
      return;
  }

  static std::mutex trace_mutex;
  static FILE *trace_file = nullptr;
  static bool header_written = false;

  std::lock_guard<std::mutex> lock(trace_mutex);
  if (trace_file == nullptr) {
    trace_file = std::strcmp(path, "-") == 0 ? stdout : std::fopen(path, "a");
    if (trace_file == nullptr)
      return;
  }
  if (!header_written) {
    std::fprintf(trace_file,
                 "cycle,event,tx_uid,kind,tma_type,pc,cta,warp,lane,tid,"
                 "src,dst,size,mbar,mf_uid,mf_addr,mf_size,issued_mf,"
                 "received_mf,bytes_completed,tx_inflight,global_inflight,"
                 "response_fifo\n");
    header_written = true;
  }

  std::fprintf(trace_file,
               "%llu,%s,%u,%s,%u,0x%llx,%u,%u,%u,%u,0x%llx,0x%llx,%u,"
               "0x%x,%u,0x%llx,%u,%u,%u,%u,%u,%u,%u\n",
               cycle, event, tx_uid, kind, tma_type, pc, cta_id, warp_id,
               lane_id, tid, src, dst, size, mbar, mf_uid, mf_addr, mf_size,
               issued_mf, received_mf, bytes_completed, tx_inflight,
               global_inflight, response_fifo);
  std::fflush(trace_file);
}

static bool tma_trace_mf_enabled(unsigned long long cycle) {
  const char *enabled = std::getenv("FLASHGPU_TMA_TRACE_MF");
  if (enabled == nullptr || enabled[0] == '\0' ||
      std::strcmp(enabled, "0") == 0)
    return false;

  const char *limit = std::getenv("FLASHGPU_TMA_TRACE_MF_CYCLE_LIMIT");
  if (limit != nullptr && limit[0] != '\0') {
    unsigned long long limit_cycle = std::strtoull(limit, nullptr, 0);
    if (limit_cycle != 0 && cycle > limit_cycle)
      return false;
  }
  return true;
}

//=============================================================================
// Shared Helper Functions
//=============================================================================

static uint32_t
effective_tma_request_granularity(const shader_core_config *config) {
  unsigned granularity = config->gpgpu_tma_request_granularity;
  if (granularity >= MAX_MEMORY_ACCESS_SIZE)
    return MAX_MEMORY_ACCESS_SIZE;
  if (granularity >= 2 * SECTOR_SIZE)
    return 2 * SECTOR_SIZE;
  return SECTOR_SIZE;
}

static unsigned response_payload_bytes(const mem_fetch *mf,
                                       const mem_fetch *parent_mf) {
  unsigned bytes = std::min(mf->get_data_size(), parent_mf->get_data_size());
  unsigned valid_bytes = mf->get_access_byte_mask().count();
  assert(valid_bytes > 0 && "TMA/cp.async response has an empty byte mask");
  return std::min(bytes, valid_bytes);
}

static uint32_t
effective_cp_async_request_granularity(const shader_core_config *config) {
  unsigned granularity = config->gpgpu_cp_async_request_granularity;
  if (granularity >= MAX_MEMORY_ACCESS_SIZE)
    return MAX_MEMORY_ACCESS_SIZE;
  if (granularity >= 2 * SECTOR_SIZE)
    return 2 * SECTOR_SIZE;
  return SECTOR_SIZE;
}

static new_addr_type align_down(new_addr_type addr, unsigned alignment) {
  return addr - (addr % alignment);
}

static bool cp_async_access_merge_candidate(const mem_access_t &access,
                                            unsigned granularity,
                                            new_addr_type base) {
  if (access.get_type() != CP_ASYNC_ACC_R || access.is_write())
    return false;
  if (access.get_size() == 0 || access.get_size() > granularity)
    return false;
  if ((access.get_addr() % SECTOR_SIZE) != 0 ||
      (access.get_size() % SECTOR_SIZE) != 0)
    return false;
  return access.get_addr() >= base &&
         access.get_addr() + access.get_size() <= base + granularity;
}

static void
append_cp_async_access_group(const std::vector<mem_access_t> &group,
                             unsigned granularity, gpgpu_context *ctx,
                             std::vector<mem_access_t> &merged_accesses) {
  if (group.empty())
    return;
  if (group.size() == 1) {
    merged_accesses.push_back(group.front());
    return;
  }

  const new_addr_type base = align_down(group.front().get_addr(), granularity);
  active_mask_t active_mask;
  mem_access_byte_mask_t byte_mask;
  mem_access_sector_mask_t sector_mask;

  for (const auto &access : group) {
    if (!cp_async_access_merge_candidate(access, granularity, base)) {
      merged_accesses.insert(merged_accesses.end(), group.begin(), group.end());
      return;
    }
    active_mask |= access.get_warp_mask();
    byte_mask |= access.get_byte_mask();

    mem_access_sector_mask_t access_sectors = access.get_sector_mask();
    if (access_sectors.none()) {
      const unsigned start_sector =
          (access.get_addr() % MAX_MEMORY_ACCESS_SIZE) / SECTOR_SIZE;
      const unsigned num_sectors = access.get_size() / SECTOR_SIZE;
      for (unsigned i = 0; i < num_sectors; ++i)
        access_sectors.set(start_sector + i);
    }
    sector_mask |= access_sectors;
  }

  mem_access_sector_mask_t required_sectors;
  const unsigned start_sector = (base % MAX_MEMORY_ACCESS_SIZE) / SECTOR_SIZE;
  const unsigned num_sectors = granularity / SECTOR_SIZE;
  for (unsigned i = 0; i < num_sectors; ++i)
    required_sectors.set(start_sector + i);

  if ((sector_mask & required_sectors) != required_sectors) {
    merged_accesses.insert(merged_accesses.end(), group.begin(), group.end());
    return;
  }

  merged_accesses.emplace_back(CP_ASYNC_ACC_R, base, granularity, false,
                               active_mask, byte_mask, required_sectors, ctx);
}

static std::vector<mem_access_t>
coalesce_cp_async_accesses(std::vector<mem_access_t> accesses,
                           unsigned granularity, gpgpu_context *ctx) {
  if (granularity <= SECTOR_SIZE || accesses.size() <= 1)
    return accesses;

  std::sort(accesses.begin(), accesses.end(),
            [](const mem_access_t &a, const mem_access_t &b) {
              return a.get_addr() < b.get_addr();
            });

  std::vector<mem_access_t> merged_accesses;
  for (auto it = accesses.begin(); it != accesses.end();) {
    const new_addr_type base = align_down(it->get_addr(), granularity);
    std::vector<mem_access_t> group;
    do {
      group.push_back(*it);
      ++it;
    } while (it != accesses.end() &&
             align_down(it->get_addr(), granularity) == base);
    append_cp_async_access_group(group, granularity, ctx, merged_accesses);
  }
  return merged_accesses;
}

// Get a 32-bit unsigned value from an operand
static uint32_t get_operand_u32(ptx_thread_info *thread,
                                const operand_info &op) {
  ptx_reg_t reg = thread->get_operand_value(op, op, U32_TYPE, thread, 0);
  return reg.u32;
}

// Get a 64-bit unsigned value from an operand
static uint64_t get_operand_u64(ptx_thread_info *thread,
                                const operand_info &op) {
  ptx_reg_t reg = thread->get_operand_value(op, op, U64_TYPE, thread, 0);
  return reg.u64;
}

[[noreturn]] static void reject_unsupported_tma(const char *reason,
                                                const ptx_instruction *pI) {
  fprintf(stderr, "TMA ERROR: %s\n", reason);
  pI->print_insn(stderr);
  fflush(stderr);
  abort();
}

// Compute instruction dimension from option
static unsigned compute_inst_dim(unsigned dim_option) {
  switch (dim_option) {
  case DIM_1D_OPTION:
    return 1;
  case DIM_2D_OPTION:
    return 2;
  case DIM_3D_OPTION:
    return 3;
  case DIM_4D_OPTION:
    return 4;
  case DIM_5D_OPTION:
    return 5;
  default:
    printf("TMA ERROR: Unknown dimension option %d\n", dim_option);
    abort();
  }
}

// Parse coordinates from an operand into a coords array
static void parse_tensor_coords(ptx_thread_info *thread,
                                const operand_info &coord_operand,
                                int32_t coords[5]) {
  // Initialize all coords to 0
  for (int i = 0; i < 5; i++)
    coords[i] = 0;

  if (coord_operand.is_vector()) {
    unsigned n_coords = coord_operand.get_vect_nelem();
    ptx_reg_t coord_regs[8];
    thread->get_vector_operand_values(coord_operand, coord_regs, n_coords);

    for (unsigned i = 0; i < n_coords && i < 5; i++) {
      coords[i] = coord_regs[i].s32;
    }
  } else {
    // Single coordinate (1D case with non-vector operand)
    coords[0] = thread
                    ->get_operand_value(coord_operand, coord_operand, S32_TYPE,
                                        thread, 0)
                    .s32;
  }
}

// Validate tensormap and dimension match
// Returns true if valid, false otherwise
static bool validate_tensormap(uint64_t tensormap_addr,
                               const tensormap_descriptor_t &tensormap,
                               unsigned inst_dim) {
  if (!tensormap.is_valid()) {
    printf("TMA ERROR: Invalid tensormap at 0x%llx\n",
           (unsigned long long)tensormap_addr);
    tensormap.print();
    fflush(stdout);
    exit(1);
  }

  unsigned tensormap_dim = tensormap.num_dims();
  if (tensormap_dim != inst_dim) {
    printf("TMA ERROR: Dimension mismatch - instruction is %uD but tensormap "
           "num_dims is %u (dim=%uD)\n",
           inst_dim, tensormap_dim, tensormap_dim);
    fflush(stdout);
    exit(1);
  }
  return true;
}

static bool is_valid_tensormap_for_dim(const tensormap_descriptor_t &tensormap,
                                       unsigned inst_dim) {
  return tensormap.is_valid() && tensormap.num_dims() == inst_dim;
}

static bool
is_valid_tensormap_for_optional_dim(const tensormap_descriptor_t &tensormap,
                                    unsigned inst_dim) {
  return inst_dim == 0 ? tensormap.is_valid()
                       : is_valid_tensormap_for_dim(tensormap, inst_dim);
}

static bool try_decode_tensormap_descriptor(tensormap_descriptor_t &tensormap,
                                            unsigned inst_dim) {
  if (is_valid_tensormap_for_optional_dim(tensormap, inst_dim))
    return true;

  tensormap_descriptor_t decoded;
  if (decode_sm100_opaque_tensormap(tensormap, decoded) &&
      is_valid_tensormap_for_optional_dim(decoded, inst_dim)) {
    tensormap = decoded;
    return true;
  }

  return false;
}

static void read_tensormap_descriptor(memory_space *global_mem,
                                      memory_space *param_mem,
                                      uint64_t tensormap_addr,
                                      unsigned inst_dim,
                                      tensormap_descriptor_t &out_tensormap) {
  if (tensormap_addr < STATIC_ALLOC_LIMIT && param_mem != nullptr) {
    param_mem->read(tensormap_addr, TENSORMAP_DESCRIPTOR_SIZE, &out_tensormap);
    if (try_decode_tensormap_descriptor(out_tensormap, inst_dim)) {
      return;
    }
    if (out_tensormap.raw_u64[0] != 0) {
      return;
    }
  }

  global_mem->read(tensormap_addr, TENSORMAP_DESCRIPTOR_SIZE, &out_tensormap);
  try_decode_tensormap_descriptor(out_tensormap, inst_dim);
}

static void
cache_tensormap_descriptor(inst_t::tma_dyn_info_t &tma_dyn_info,
                           const tensormap_descriptor_t &tensormap) {
  static_assert(inst_t::tma_dyn_info_t::TMA_DESCRIPTOR_BYTES ==
                    TENSORMAP_DESCRIPTOR_SIZE,
                "TMA descriptor cache size must match tensor map size");
  memcpy(tma_dyn_info.tensormap_descriptor, tensormap.raw_bytes,
         TENSORMAP_DESCRIPTOR_SIZE);
  tma_dyn_info.has_tensormap_descriptor = true;
}

//=============================================================================
// Performance Simulation: TMA AGU (Address Generation Unit)
//=============================================================================

struct tma_agu_state_t {
  // Mode and completion state
  bool is_tensor = false;       // true=tensor mode, false=linear mode
  bool done = true;             // true=iteration complete
  bool is_fill_request = false; // true=current request is OOB fill

  // Tensor mode state (multi-dimensional tile traversal)
  uint32_t num_dims = 0;            // Number of dimensions (1-5)
  uint32_t elem_size = 0;           // Element size in bytes
  uint32_t box_dim[5] = {0};        // Tile dimensions [d0, d1, d2, d3, d4]
  uint32_t global_dim[5] = {0};     // Global tensor dimensions
  int32_t start_coords[5] = {0};    // Starting coordinates of this tile
  uint64_t global_strides[5] = {0}; // Strides for each dimension
  uint32_t tile_coords[5] = {0};    // Current position within tile (odometer)
  uint64_t curr_row_addr = 0;       // Base physical address of current row
  uint32_t offset_in_row = 0;       // Byte offset within current row
  uint32_t row_bytes = 0;           // Total bytes in a row (dim0 extent)

  // Cached OOB state (avoids per-sector dimension traversal)
  bool row_is_oob = false;            // Higher-dim OOB: entire row is fill
  uint32_t valid_row_start_bytes = 0; // First valid dim0 byte in the tile row
  uint32_t valid_row_end_bytes = 0;   // One-past-last valid dim0 byte

  // Linear mode state (simple 1D address range)
  uint64_t linear_addr = 0;      // Current address
  uint32_t linear_remaining = 0; // Remaining bytes
};

class tma_agu_unit_t {
public:
  void init_tensor(tma_agu_state_t &state, const tensormap_descriptor_t &tm,
                   const int32_t start_coords[5]) {

    if (!tm.is_valid()) {
      assert(false && "TMA AGU ERROR: Invalid tensormap in init_tensor");
    }

    state = tma_agu_state_t(); // Reset to defaults
    state.is_tensor = true;
    state.num_dims = tm.num_dims();
    state.elem_size = tm.get_element_size();

    // Cache box dimensions and global dimensions
    for (uint32_t d = 0; d < state.num_dims; d++) {
      state.box_dim[d] = tm.fields.boxDim[d];
      state.global_dim[d] = tm.fields.globalDim[d];
      state.start_coords[d] = start_coords[d];
    }

    // Dim0 may be a packed U4/U6 format, so its byte offset is computed via
    // the tensor map helper below.  Higher dimensions always use byte strides.
    state.global_strides[0] = state.elem_size;
    for (uint32_t d = 1; d < state.num_dims; d++) {
      state.global_strides[d] = tm.fields.globalStrides[d - 1];
    }

    // Initialize tile coordinates to (0,0,...,0)
    for (int i = 0; i < 5; i++) {
      state.tile_coords[i] = 0;
    }

    // Calculate initial row address using signed coordinates.  OOB requests
    // may precede the tensor base when OOB L2 traffic is enabled.
    const int64_t dim0_coord = start_coords[0];
    const uint64_t dim0_magnitude =
        static_cast<uint64_t>(dim0_coord < 0 ? -dim0_coord : dim0_coord);
    const int64_t dim0_byte_offset =
        static_cast<int64_t>(tm.get_dim0_gmem_byte_offset(dim0_magnitude));
    int64_t byte_offset = dim0_coord < 0 ? -dim0_byte_offset : dim0_byte_offset;
    for (uint32_t d = 1; d < state.num_dims; d++) {
      byte_offset += static_cast<int64_t>(start_coords[d]) *
                     static_cast<int64_t>(state.global_strides[d]);
    }
    if (byte_offset < 0) {
      state.curr_row_addr =
          tm.fields.globalAddress - static_cast<uint64_t>(-byte_offset);
    } else {
      state.curr_row_addr =
          tm.fields.globalAddress + static_cast<uint64_t>(byte_offset);
    }

    state.row_bytes =
        static_cast<uint32_t>(tm.get_dim0_gmem_span_bytes(state.box_dim[0]));
    state.offset_in_row = 0;
    state.done = false;

    // Precompute the dim0 intersection with the tensor.  Both boundaries are
    // tile-relative so a negative start coordinate produces leading fill
    // bytes followed by valid data.
    const int64_t row_start = start_coords[0];
    const int64_t row_end = row_start + state.box_dim[0];
    const int64_t valid_start = std::max<int64_t>(row_start, 0);
    const int64_t valid_end = std::min<int64_t>(row_end, state.global_dim[0]);
    if (valid_start < valid_end) {
      state.valid_row_start_bytes = static_cast<uint32_t>(
          tm.get_dim0_gmem_byte_offset(valid_start - row_start));
      state.valid_row_end_bytes = static_cast<uint32_t>(
          tm.get_dim0_gmem_span_bytes(valid_end - row_start));
    } else {
      state.valid_row_start_bytes = state.row_bytes;
      state.valid_row_end_bytes = state.row_bytes;
    }

    // Precompute higher-dim OOB for first row (tile_coords all zero)
    state.row_is_oob = false;
    for (uint32_t d = 1; d < state.num_dims; d++) {
      if (start_coords[d] < 0 ||
          static_cast<uint32_t>(start_coords[d]) >= state.global_dim[d]) {
        state.row_is_oob = true;
        break;
      }
    }
  }

  void init_linear(tma_agu_state_t &state, uint64_t start_addr,
                   uint32_t total_bytes) {
    state = tma_agu_state_t(); // Reset to defaults
    state.is_tensor = false;
    state.linear_addr = start_addr;
    state.linear_remaining = total_bytes;
    state.done = (total_bytes == 0);
  }

  /**
   * Generate next memory request using AGU
   *
   * For tensor mode:
   *   - Traverses tile in row-major order
   *   - Skips OOB coordinates (zero-padding)
   *   - Breaks rows into configured request-granularity chunks
   *
   * For linear mode:
   *   - Simple sequential address generation
   *   - Aligns to configured request-granularity boundaries
   */
  bool gen_next_req(tma_agu_state_t &state, uint64_t &out_addr,
                    uint32_t &out_size, uint32_t request_granularity) {
    if (state.done)
      return false;

    if (!state.is_tensor) {
      // Linear mode: simple sequential address generation
      if (state.linear_remaining == 0) {
        state.done = true;
        return false;
      }

      state.is_fill_request = false; // Linear mode never fills
      out_addr = state.linear_addr;
      uint32_t to_boundary =
          request_granularity - (state.linear_addr % request_granularity);
      out_size =
          std::min({request_granularity, state.linear_remaining, to_boundary});

      state.linear_addr += out_size;
      state.linear_remaining -= out_size;

      if (state.linear_remaining == 0) {
        state.done = true;
      }

    } else {
      // Tensor mode: odometer-style multi-dimensional traversal
      // OOB check using cached state (no per-sector dimension traversal)
      if (state.row_is_oob) {
        state.is_fill_request = true; // Higher-dim OOB: entire row is fill
      } else {
        state.is_fill_request =
            state.offset_in_row < state.valid_row_start_bytes ||
            state.offset_in_row >= state.valid_row_end_bytes;
      }

      // Calculate address: current_row_base + offset_within_row
      out_addr = state.curr_row_addr + state.offset_in_row;

      // Never mix fill and valid bytes in one request.
      uint32_t row_remaining = state.row_bytes - state.offset_in_row;
      if (!state.row_is_oob) {
        if (state.offset_in_row < state.valid_row_start_bytes) {
          row_remaining = std::min(row_remaining, state.valid_row_start_bytes -
                                                      state.offset_in_row);
        } else if (state.offset_in_row < state.valid_row_end_bytes) {
          row_remaining = std::min(row_remaining, state.valid_row_end_bytes -
                                                      state.offset_in_row);
        }
      }
      uint32_t to_boundary =
          request_granularity - (out_addr % request_granularity);
      out_size = std::min({request_granularity, row_remaining, to_boundary});

      // Advance position within row
      state.offset_in_row += out_size;

      // If row complete, advance to next row (odometer increment)
      if (state.offset_in_row >= state.row_bytes) {
        advance_to_next_row(state);
      }
    }

    return true;
  }

private:
  // Advance to next row using odometer-style increment of higher dimensions.
  // Updates row_is_oob cache after advancing.
  void advance_to_next_row(tma_agu_state_t &state) {
    state.offset_in_row = 0;

    // Increment coordinates starting from dimension 1 (dimension 0 is row)
    bool carried_all = true;
    for (uint32_t d = 1; d < state.num_dims; d++) {
      state.tile_coords[d]++;
      state.curr_row_addr += state.global_strides[d];

      if (state.tile_coords[d] < state.box_dim[d]) {
        carried_all = false;
        break; // No carry - done advancing
      }

      // Carry to next dimension: reset this dimension
      state.curr_row_addr -= state.box_dim[d] * state.global_strides[d];
      state.tile_coords[d] = 0;
    }

    if (carried_all) {
      state.done = true;
      return;
    }

    // Recompute higher-dim OOB for new row (once per row, not per sector)
    state.row_is_oob = false;
    for (uint32_t d = 1; d < state.num_dims; d++) {
      const int64_t coord =
          static_cast<int64_t>(state.start_coords[d]) + state.tile_coords[d];
      if (coord < 0 || coord >= state.global_dim[d]) {
        state.row_is_oob = true;
        break;
      }
    }
  }
};

//=============================================================================
// Performance Simulation: TMA Unit
//=============================================================================

class tma_unit_impl_t {
public:
  tma_unit_impl_t(shader_core_ctx *shader_ctx, barrier_set_t *barriers,
                  mem_fetch_interface *icnt,
                  shader_core_mem_fetch_allocator *mf_allocator)
      : m_shader_ctx(shader_ctx), m_barriers(barriers), m_icnt(icnt),
        m_mf_allocator(mf_allocator) {}

private:
  shader_core_ctx *m_shader_ctx;
  barrier_set_t *m_barriers;
  mem_fetch_interface *m_icnt;
  shader_core_mem_fetch_allocator *m_mf_allocator;

  tma_agu_unit_t m_agu;
  unsigned m_mf_inflight = 0; // Actual mem_fetch in L2 pipeline (not skips)
  unsigned m_request_byte_credit = 0;

  unsigned long long current_cycle() const {
    return m_shader_ctx->get_gpu()->gpu_sim_cycle +
           m_shader_ctx->get_gpu()->gpu_tot_sim_cycle;
  }

  struct tma_transaction_t {
    ptx_thread_info *m_thread = nullptr;
    ptx_instruction *m_inst = nullptr;
    inst_t::tma_static_info_t m_static_info;
    inst_t::tma_dyn_info_t m_dyn_info;
    unsigned m_cta_id = 0;
    unsigned m_warp_id = 0;
    unsigned m_lane_id = 0;
    unsigned m_tid = 0;
    unsigned long long m_pc = 0;
    unsigned long long m_create_cycle = 0;
    unsigned long long m_first_issue_cycle = 0;
    unsigned long long m_last_issue_cycle = 0;
    unsigned long long m_first_response_cycle = 0;
    unsigned long long m_complete_cycle = 0;
    uint32_t m_bytes_completed = 0;
    tma_agu_state_t agu_state; // AGU state for this transaction

    // Debug counters
    uint32_t m_mf_issued_count = 0;   // Number of mem_fetch requests issued
    uint32_t m_mf_received_count = 0; // Number of mem_fetch responses received
    uint32_t m_mf_tx_inflight = 0; // Currently in-flight for this transaction

    void reset() {
      m_thread = nullptr;
      m_inst = nullptr;
      m_cta_id = 0;
      m_warp_id = 0;
      m_lane_id = 0;
      m_tid = 0;
      m_pc = 0;
      m_create_cycle = 0;
      m_first_issue_cycle = 0;
      m_last_issue_cycle = 0;
      m_first_response_cycle = 0;
      m_complete_cycle = 0;
      m_bytes_completed = 0;
      agu_state = tma_agu_state_t(); // Reset to default state
      m_mf_issued_count = 0;
      m_mf_received_count = 0;
      m_mf_tx_inflight = 0;
    }

    bool is_valid() const { return m_thread != nullptr; }
  };
  std::unordered_map<unsigned, tma_transaction_t> m_transactions;

  std::list<unsigned> issue_queue;
  std::unordered_map<unsigned, unsigned> m_mf_to_tx;
  std::unordered_map<unsigned, uint32_t> m_mf_pending_bytes;

  struct cp_async_transaction_t {
    mem_access_t access;
    warp_inst_t inst;
    unsigned cta_id = 0;
    unsigned warp_id = 0;
    unsigned long long pc = 0;
    unsigned long long stream_id = 0;
    unsigned long long create_cycle = 0;
    unsigned long long first_issue_cycle = 0;
    unsigned long long first_response_cycle = 0;
    uint32_t bytes_completed = 0;
    uint32_t mf_issued_count = 0;
    uint32_t mf_received_count = 0;
    std::set<unsigned> dependent_arrival_uids;

    cp_async_transaction_t(gpgpu_context *ctx) : access(ctx) {}
  };

  struct cp_async_mbarrier_arrival_t {
    unsigned cta_id;
    unsigned warp_id;
    uint32_t mbarrier_addr;
    std::set<unsigned> pending_tx_uids;
  };

  struct cp_async_warp_group_info_t {
    unsigned next_group_id = 1;
    unsigned wait_allowance = 0;
    bool is_waiting = false;
    std::set<unsigned> pending_txs;
    std::map<unsigned, std::set<unsigned>> pending_groups;
    std::map<unsigned, unsigned> tx_to_group;

    void add_tx(unsigned tx_uid) { pending_txs.insert(tx_uid); }

    void commit_group() {
      if (!pending_txs.empty()) {
        for (unsigned tx_uid : pending_txs) {
          tx_to_group[tx_uid] = next_group_id;
        }
        pending_groups.emplace(next_group_id, pending_txs);
        pending_txs.clear();
      }
      next_group_id++;
    }

    bool wait_group(unsigned max_pending_groups) {
      if (pending_groups.size() <= max_pending_groups)
        return true;
      assert(!is_waiting && "warp is already waiting on cp.async group");
      wait_allowance = max_pending_groups;
      is_waiting = true;
      return false;
    }

    bool complete_tx(unsigned tx_uid) {
      auto tx_it = tx_to_group.find(tx_uid);
      if (tx_it == tx_to_group.end()) {
        pending_txs.erase(tx_uid);
      } else {
        unsigned group_id = tx_it->second;
        tx_to_group.erase(tx_it);
        auto group_it = pending_groups.find(group_id);
        assert(group_it != pending_groups.end());
        group_it->second.erase(tx_uid);
        if (group_it->second.empty())
          pending_groups.erase(group_it);
      }

      if (is_waiting && pending_groups.size() <= wait_allowance) {
        is_waiting = false;
        return true;
      }
      return false;
    }

    bool has_pending() const {
      return !pending_txs.empty() || !pending_groups.empty() || is_waiting;
    }
  };

  std::unordered_map<unsigned, cp_async_transaction_t> m_cp_transactions;
  std::list<unsigned> m_cp_issue_queue;
  std::unordered_map<unsigned, unsigned> m_cp_mf_to_tx;
  std::unordered_map<unsigned, uint32_t> m_cp_mf_pending_bytes;
  std::map<std::pair<unsigned, unsigned>, cp_async_warp_group_info_t>
      m_cp_group_info;
  std::unordered_map<unsigned, cp_async_mbarrier_arrival_t>
      m_cp_mbarrier_arrivals;
  unsigned m_next_cp_mbarrier_arrival_uid = 0;
  unsigned m_cp_mf_inflight = 0;

  std::list<mem_fetch *> m_response_fifo;

  // Delayed arrive_tx queue: complete_tx is deferred by arrive_latency cycles
  struct pending_arrive_t {
    unsigned remaining;
    unsigned cta_id;
    unsigned warp_id;
    uint32_t mbar_addr;
    uint32_t size_in_bytes;
    bool is_write;
    unsigned tx_uid; // for bulk_group (write path)
  };
  std::vector<pending_arrive_t> m_pending_arrives;

  static bool is_write_transaction(const tma_transaction_t &tx) {
    return tx.m_static_info.dst_space == inst_t::tma_static_info_t::TMA_GLOBAL;
  }

  static bool same_tma_dyn_info(const inst_t::tma_dyn_info_t &lhs,
                                const inst_t::tma_dyn_info_t &rhs) {
    if (lhs.dst_addr != rhs.dst_addr || lhs.src_addr != rhs.src_addr ||
        lhs.size_in_bytes != rhs.size_in_bytes ||
        lhs.mbar_addr != rhs.mbar_addr ||
        lhs.has_tensormap_descriptor != rhs.has_tensormap_descriptor)
      return false;

    for (unsigned i = 0; i < 5; ++i) {
      if (lhs.coords[i] != rhs.coords[i])
        return false;
    }

    if (lhs.has_tensormap_descriptor &&
        std::memcmp(lhs.tensormap_descriptor, rhs.tensormap_descriptor,
                    inst_t::tma_dyn_info_t::TMA_DESCRIPTOR_BYTES) != 0)
      return false;

    return true;
  }

  // Helper function to finalize a completed transaction
  void finalize_transaction(unsigned tx_uid) {
    auto it = m_transactions.find(tx_uid);
    if (it == m_transactions.end())
      return;

    auto &tx = it->second;
    unsigned cta_id = tx.m_cta_id;
    unsigned warp_id = tx.m_warp_id;

    bool is_write = is_write_transaction(tx);
    tx.m_complete_cycle = current_cycle();
    tma_trace_emit(
        tx.m_complete_cycle, "COMPLETE", tx_uid, is_write ? "WRITE" : "READ",
        tx.m_static_info.tma_type, tx.m_pc, tx.m_cta_id, tx.m_warp_id,
        tx.m_lane_id, tx.m_tid, tx.m_dyn_info.src_addr, tx.m_dyn_info.dst_addr,
        tx.m_dyn_info.size_in_bytes, tx.m_dyn_info.mbar_addr, 0, 0, 0,
        tx.m_mf_issued_count, tx.m_mf_received_count, tx.m_bytes_completed,
        tx.m_mf_tx_inflight, m_mf_inflight, m_response_fifo.size());
    m_shader_ctx->inc_tma_tx_completed(is_write);
    record_tma_tx_completed(is_write);
    GPPRINTF_TMA(
        TMA,
        "[TMA %s COMPLETE] tx_uid=%u, cta_id=%u, warp_id=%u, mbar=0x%x, "
        "issued_mf=%u, received_mf=%u, bytes_completed=%u/%u\n",
        is_write ? "WRITE" : "READ", tx_uid, cta_id, warp_id,
        tx.m_dyn_info.mbar_addr, tx.m_mf_issued_count, tx.m_mf_received_count,
        tx.m_bytes_completed, tx.m_dyn_info.size_in_bytes);
    fflush(stdout);

    GPPRINTF_TMA(TMA,
                 "Complete transaction dst=0x%llx, src=0x%llx, "
                 "size_in_bytes=%u, mbar=0x%x, tx_uid=%u\n",
                 (unsigned long long)tx.m_dyn_info.dst_addr,
                 (unsigned long long)tx.m_dyn_info.src_addr,
                 tx.m_dyn_info.size_in_bytes, tx.m_dyn_info.mbar_addr, tx_uid);

    // For TMA read, notify mbarrier of completion
    // For TMA write, use bulk_group completion mechanism
    unsigned arrive_latency =
        m_shader_ctx->get_config()->gpgpu_mbarrier_arrive_latency;
    if (arrive_latency > 0) {
      m_pending_arrives.push_back(
          {arrive_latency, cta_id, warp_id, tx.m_dyn_info.mbar_addr,
           tx.m_dyn_info.size_in_bytes, is_write, tx_uid});
    } else {
      if (!is_write) {
        m_barriers->complete_tx(cta_id, warp_id, tx.m_dyn_info.mbar_addr,
                                tx.m_dyn_info.size_in_bytes);
      } else {
        m_barriers->complete_bulk_tx(cta_id, warp_id, tx_uid);
      }
      tma_trace_emit(current_cycle(), "ARRIVE", tx_uid,
                     is_write ? "WRITE" : "READ", tx.m_static_info.tma_type,
                     tx.m_pc, tx.m_cta_id, tx.m_warp_id, tx.m_lane_id, tx.m_tid,
                     tx.m_dyn_info.src_addr, tx.m_dyn_info.dst_addr,
                     tx.m_dyn_info.size_in_bytes, tx.m_dyn_info.mbar_addr, 0, 0,
                     0, tx.m_mf_issued_count, tx.m_mf_received_count,
                     tx.m_bytes_completed, tx.m_mf_tx_inflight, m_mf_inflight,
                     m_response_fifo.size());
    }

    m_transactions.erase(it);

    // Remove all mappings for this transaction
    for (auto map_it = m_mf_to_tx.begin(); map_it != m_mf_to_tx.end();) {
      if (map_it->second == tx_uid) {
        map_it = m_mf_to_tx.erase(map_it);
      } else {
        ++map_it;
      }
    }
    for (auto pending_it = m_mf_pending_bytes.begin();
         pending_it != m_mf_pending_bytes.end();) {
      auto map_it = m_mf_to_tx.find(pending_it->first);
      if (map_it == m_mf_to_tx.end()) {
        pending_it = m_mf_pending_bytes.erase(pending_it);
      } else {
        ++pending_it;
      }
    }
  }

  void finalize_cp_async_transaction(unsigned tx_uid) {
    auto it = m_cp_transactions.find(tx_uid);
    if (it == m_cp_transactions.end())
      return;

    unsigned cta_id = it->second.cta_id;
    unsigned warp_id = it->second.warp_id;
    auto group_it = m_cp_group_info.find(std::make_pair(cta_id, warp_id));
    assert(group_it != m_cp_group_info.end());
    bool should_release = group_it->second.complete_tx(tx_uid);
    if (should_release) {
      g_cp_async_wait_releases.fetch_add(1, std::memory_order_relaxed);
      m_barriers->release_cp_async_warp(warp_id);
    }

    for (unsigned arrival_uid : it->second.dependent_arrival_uids) {
      auto arrival_it = m_cp_mbarrier_arrivals.find(arrival_uid);
      assert(arrival_it != m_cp_mbarrier_arrivals.end());
      auto &arrival = arrival_it->second;
      const size_t erased = arrival.pending_tx_uids.erase(tx_uid);
      assert(erased == 1);
      if (arrival.pending_tx_uids.empty()) {
        m_barriers->arrive_mbarrier_async(arrival.cta_id, arrival.warp_id,
                                          arrival.mbarrier_addr);
        m_cp_mbarrier_arrivals.erase(arrival_it);
      }
    }

    g_cp_async_tx_completed.fetch_add(1, std::memory_order_relaxed);
    m_cp_transactions.erase(it);
  }

  void process_cp_async_response(mem_fetch *mf) {
    mem_fetch *parent_mf = mf->get_original_mf() ? mf->get_original_mf() : mf;
    unsigned parent_uid = parent_mf->get_request_uid();
    auto mf_it = m_cp_mf_to_tx.find(parent_uid);
    assert(mf_it != m_cp_mf_to_tx.end() &&
           "cp.async response has no live transaction");
    unsigned tx_uid = mf_it->second;
    auto tx_it = m_cp_transactions.find(tx_uid);
    assert(tx_it != m_cp_transactions.end());
    auto &tx = tx_it->second;

    assert(mf->get_access_type() == CP_ASYNC_ACC_R);
    tx.mf_received_count++;
    g_cp_async_mf_responses.fetch_add(1, std::memory_order_relaxed);
    if (tx.first_response_cycle == 0)
      tx.first_response_cycle = current_cycle();

    // cp.async transactions track physical request bytes. A sparse per-lane
    // byte mask still returns every requested cache sector.
    unsigned mf_size = mf->get_data_size();
    unsigned parent_size = parent_mf->get_data_size();
    unsigned bytes_to_add = std::min(mf_size, parent_size);
    tx.bytes_completed += bytes_to_add;
    g_cp_async_bytes_completed.fetch_add(bytes_to_add,
                                         std::memory_order_relaxed);

    bool parent_complete = false;
    auto pending_it = m_cp_mf_pending_bytes.find(parent_uid);
    assert(pending_it != m_cp_mf_pending_bytes.end());
    if (bytes_to_add >= pending_it->second) {
      assert(m_cp_mf_inflight > 0);
      m_cp_mf_inflight--;
      m_cp_mf_pending_bytes.erase(pending_it);
      m_cp_mf_to_tx.erase(parent_uid);
      parent_complete = true;
    } else {
      pending_it->second -= bytes_to_add;
    }

    if (tx.bytes_completed >= tx.access.get_size()) {
      finalize_cp_async_transaction(tx_uid);
    }

    mem_fetch *parent_to_delete =
        (parent_complete && parent_mf != mf) ? parent_mf : NULL;
    delete mf;
    delete parent_to_delete;
  }

  void issue_cp_async_requests() {
    if (m_cp_issue_queue.empty())
      return;

    g_cp_async_issue_queue_cycles.fetch_add(1, std::memory_order_relaxed);
    atomic_update_max(g_cp_async_max_issue_queue, m_cp_issue_queue.size());
    atomic_update_max(g_cp_async_max_inflight, m_cp_mf_inflight);

    unsigned max_inflight =
        m_shader_ctx->get_config()->gpgpu_cp_async_max_inflight;
    unsigned request_width =
        m_shader_ctx->get_config()->gpgpu_cp_async_request_width
            ? m_shader_ctx->get_config()->gpgpu_cp_async_request_width
            : 1;
    unsigned issued_requests = 0;

    while (issued_requests < request_width && !m_cp_issue_queue.empty()) {
      if (max_inflight > 0 && m_cp_mf_inflight >= max_inflight) {
        g_cp_async_issue_blocked_inflight_cycles.fetch_add(
            1, std::memory_order_relaxed);
        break;
      }
      if (m_icnt->full(READ_PACKET_SIZE, false)) {
        g_cp_async_issue_blocked_icnt_cycles.fetch_add(
            1, std::memory_order_relaxed);
        break;
      }

      unsigned tx_uid = m_cp_issue_queue.front();
      auto tx_it = m_cp_transactions.find(tx_uid);
      assert(tx_it != m_cp_transactions.end());
      auto &tx = tx_it->second;

      mem_fetch *mf =
          m_mf_allocator->alloc(tx.inst, tx.access, current_cycle());
      m_cp_mf_to_tx.emplace(mf->get_request_uid(), tx_uid);
      m_cp_mf_pending_bytes.emplace(mf->get_request_uid(),
                                    tx.access.get_size());

      if (tx.first_issue_cycle == 0)
        tx.first_issue_cycle = current_cycle();
      tx.mf_issued_count++;
      g_cp_async_mf_issued.fetch_add(1, std::memory_order_relaxed);
      g_cp_async_bytes_issued.fetch_add(tx.access.get_size(),
                                        std::memory_order_relaxed);
      m_icnt->push(mf);
      m_cp_mf_inflight++;
      atomic_update_max(g_cp_async_max_inflight, m_cp_mf_inflight);
      issued_requests++;
      m_cp_issue_queue.pop_front();
    }

    if (issued_requests > 0)
      g_cp_async_issue_active_cycles.fetch_add(1, std::memory_order_relaxed);
    if (issued_requests >= request_width && !m_cp_issue_queue.empty())
      g_cp_async_issue_width_limited_cycles.fetch_add(
          1, std::memory_order_relaxed);
  }

public:
  void warp_reaches_tma(unsigned cta_id, unsigned warp_id, warp_inst_t *inst) {
    ptx_instruction *pI = dynamic_cast<ptx_instruction *>(inst);
    if (pI == nullptr) {
      printf("Error: TMA inst is not a ptx_inst\n");
      abort();
    }

    const auto &tma_static_info = pI->get_tma_static_info();

    // Regular TMA copy instruction - queue for async processing
    const auto warp_size = m_shader_ctx->get_warp_size();
    // SM100/FA4 execution can record equivalent dynamic TMA state on multiple
    // lanes. Coalesce only equivalent records; divergent lane state cannot be
    // represented by one TMA transaction and remains unsupported.
    int canonical_lane = -1;
    const inst_t::tma_dyn_info_t *canonical_tma_dyn_info = nullptr;
    for (unsigned laneid = 0; laneid < warp_size; laneid++) {
      const auto &tma_dyn_info = pI->get_tma_dyn_info(laneid);
      if (!tma_dyn_info.is_valid())
        continue;

      if (canonical_tma_dyn_info == nullptr) {
        canonical_lane = laneid;
        canonical_tma_dyn_info = &tma_dyn_info;
        continue;
      }

      if (!same_tma_dyn_info(*canonical_tma_dyn_info, tma_dyn_info)) {
        printf("Error: non-uniform active lanes for one TMA instruction are "
               "not supported: canonical_lane=%d lane=%u pc=0x%llx inst=%s\n",
               canonical_lane, laneid, (unsigned long long)pI->get_PC(),
               pI->to_string().c_str());
        printf("  canonical dst=0x%llx src=0x%llx size=%u mbar=0x%x "
               "coords=[%d,%d,%d,%d,%d]\n",
               (unsigned long long)canonical_tma_dyn_info->dst_addr,
               (unsigned long long)canonical_tma_dyn_info->src_addr,
               canonical_tma_dyn_info->size_in_bytes,
               canonical_tma_dyn_info->mbar_addr,
               canonical_tma_dyn_info->coords[0],
               canonical_tma_dyn_info->coords[1],
               canonical_tma_dyn_info->coords[2],
               canonical_tma_dyn_info->coords[3],
               canonical_tma_dyn_info->coords[4]);
        printf("  lane       dst=0x%llx src=0x%llx size=%u mbar=0x%x "
               "coords=[%d,%d,%d,%d,%d]\n",
               (unsigned long long)tma_dyn_info.dst_addr,
               (unsigned long long)tma_dyn_info.src_addr,
               tma_dyn_info.size_in_bytes, tma_dyn_info.mbar_addr,
               tma_dyn_info.coords[0], tma_dyn_info.coords[1],
               tma_dyn_info.coords[2], tma_dyn_info.coords[3],
               tma_dyn_info.coords[4]);
        abort();
      }
    }

    for (unsigned laneid = 0; laneid < warp_size; laneid++) {
      if (canonical_lane >= 0 && (int)laneid != canonical_lane)
        continue;

      const auto &tma_dyn_info = pI->get_tma_dyn_info(laneid);
      if (tma_dyn_info.is_valid()) {
        unsigned tid = warp_size * warp_id + laneid;
        auto thread = m_shader_ctx->get_thread_info()[tid];

        // Create a TMA transaction for this thread.
        tma_transaction_t tx{
            .m_thread = thread,
            .m_inst = pI,
            .m_static_info = tma_static_info,
            .m_dyn_info = tma_dyn_info,
            .m_cta_id = cta_id,
            .m_warp_id = warp_id,
            .m_lane_id = laneid,
            .m_tid = tid,
            .m_pc = pI->get_PC(),
            .m_create_cycle = current_cycle(),
            .m_bytes_completed = 0,
        };

        // Initialize address generator
        if (tma_static_info.tma_type == inst_t::tma_static_info_t::TMA_TENSOR) {

          memory_space *global_mem = thread->get_global_memory();
          tensormap_descriptor_t tensormap;
          // For reads (shared <- global): tensormap is at src_addr
          // For writes (global <- shared): tensormap is at dst_addr
          bool is_write_op = (tma_static_info.dst_space ==
                              inst_t::tma_static_info_t::TMA_GLOBAL);
          uint64_t tensormap_addr =
              is_write_op ? tma_dyn_info.dst_addr : tma_dyn_info.src_addr;
          if (tma_dyn_info.has_tensormap_descriptor) {
            memcpy(tensormap.raw_bytes, tma_dyn_info.tensormap_descriptor,
                   TENSORMAP_DESCRIPTOR_SIZE);
          } else {
            read_tensormap_descriptor(global_mem, thread->get_param_memory(),
                                      tensormap_addr,
                                      tma_static_info.tensor_dim, tensormap);
          }
          if (tma_static_info.tensor_dim != 0) {
            validate_tensormap(tensormap_addr, tensormap,
                               tma_static_info.tensor_dim);
          } else if (!tensormap.is_valid()) {
            printf("TMA ERROR: Invalid tensormap at 0x%llx\n",
                   (unsigned long long)tensormap_addr);
            tensormap.print();
            fflush(stdout);
            exit(1);
          }
          m_agu.init_tensor(tx.agu_state, tensormap, tma_dyn_info.coords);

        } else {
          // For linear mode, AGU generates global memory addresses
          // Load (shared <- global): use src_addr (global)
          // Store (global <- shared): use dst_addr (global)
          bool is_write_op = (tma_static_info.dst_space ==
                              inst_t::tma_static_info_t::TMA_GLOBAL);
          uint64_t global_addr =
              is_write_op ? tma_dyn_info.dst_addr : tma_dyn_info.src_addr;
          m_agu.init_linear(tx.agu_state, global_addr,
                            tma_dyn_info.size_in_bytes);
        }

        unsigned tx_uid =
            tma_next_tx_uid.fetch_add(1, std::memory_order_relaxed);
        m_transactions.emplace(tx_uid, tx);

        bool is_write_op = (tma_static_info.dst_space ==
                            inst_t::tma_static_info_t::TMA_GLOBAL);
        record_tma_tx_started(is_write_op);
        tma_trace_emit(tx.m_create_cycle, "NEW", tx_uid,
                       is_write_op ? "WRITE" : "READ", tma_static_info.tma_type,
                       tx.m_pc, tx.m_cta_id, tx.m_warp_id, tx.m_lane_id,
                       tx.m_tid, tma_dyn_info.src_addr, tma_dyn_info.dst_addr,
                       tma_dyn_info.size_in_bytes, tma_dyn_info.mbar_addr, 0, 0,
                       0, 0, 0, 0, 0, m_mf_inflight, m_response_fifo.size());

        bool idealized =
            m_shader_ctx->get_config()->gpgpu_tma_idealized_memory != 0;

        // For TMA write operations, add transaction to bulk group
        if (is_write_op) {
          m_barriers->add_bulk_tx(cta_id, warp_id, tx_uid);
        }

        if (idealized) {
          finalize_transaction(tx_uid);
        } else {
          issue_queue.push_back(tx_uid);
        }

        GPPRINTF_INST_EXEC(
            TMA,
            "[TMA START] cta_id=%u, warp_id=%u, lane=%d, tid=%u, tx_uid=%u, "
            "dst=0x%llx, src=0x%llx, size=%u, mbar=0x%x\n",
            thread->get_hw_ctaid(), warp_id, laneid, tid, tx_uid,
            (unsigned long long)tma_dyn_info.dst_addr,
            (unsigned long long)tma_dyn_info.src_addr,
            tma_dyn_info.size_in_bytes, tma_dyn_info.mbar_addr);
        fflush(stdout);

        GPPRINTF_INST_EXEC(
            TMA,
            "Start transaction dst=0x%llx, tensormap at 0x%llx, "
            "size_in_bytes=%u, mbar=0x%x, tx_uid=%u, tma_type=%d\n",
            (unsigned long long)tma_dyn_info.dst_addr,
            (unsigned long long)tma_dyn_info.src_addr,
            tma_dyn_info.size_in_bytes, tma_dyn_info.mbar_addr, tx_uid,
            tma_static_info.tma_type);
      }
    }
  }

  void warp_reaches_cp_async(unsigned cta_id, unsigned warp_id,
                             const warp_inst_t &inst,
                             const ptx_instruction *static_inst) {
    unsigned cp_size = inst.data_size ? inst.data_size : 16;
    unsigned src_size = cp_size;
    if (static_inst != nullptr) {
      const operand_info *src_size_op =
          static_inst->cp_async_source_control_operand();
      if (src_size_op != nullptr && src_size_op->is_literal()) {
        src_size = (unsigned)src_size_op->get_literal_value().u64;
      }
    }
    if (src_size > cp_size)
      src_size = cp_size;
    if (src_size == 0)
      return;

    warp_inst_t access_inst = inst;
    access_inst.space.set_type(global_space);
    access_inst.memory_op = memory_load;
    access_inst.data_size = src_size;
    access_inst.memory_coalescing_arch(false, CP_ASYNC_ACC_R);

    std::vector<mem_access_t> accesses;
    while (!access_inst.accessq_empty()) {
      accesses.push_back(access_inst.accessq_back());
      access_inst.accessq_pop_back();
    }
    accesses = coalesce_cp_async_accesses(
        accesses,
        effective_cp_async_request_granularity(m_shader_ctx->get_config()),
        m_shader_ctx->get_gpu()->gpgpu_ctx);

    auto &group = m_cp_group_info[std::make_pair(cta_id, warp_id)];
    for (const auto &access : accesses) {
      unsigned tx_uid = tma_next_tx_uid.fetch_add(1, std::memory_order_relaxed);

      cp_async_transaction_t tx(m_shader_ctx->get_gpu()->gpgpu_ctx);
      tx.access = access;
      tx.inst = inst;
      tx.cta_id = cta_id;
      tx.warp_id = warp_id;
      tx.pc = inst.pc;
      tx.stream_id = inst.get_streamID();
      tx.create_cycle = current_cycle();

      m_cp_transactions.emplace(tx_uid, tx);
      g_cp_async_tx_started.fetch_add(1, std::memory_order_relaxed);
      group.add_tx(tx_uid);

      if (m_shader_ctx->get_config()->gpgpu_cp_async_idealized_memory != 0) {
        finalize_cp_async_transaction(tx_uid);
      } else {
        m_cp_issue_queue.push_back(tx_uid);
      }
    }
  }

  void
  warp_reaches_cp_async_mbarrier_arrive(unsigned cta_id, unsigned warp_id,
                                        const warp_inst_t &inst,
                                        const ptx_instruction &dynamic_inst) {
    bool increment_pending = true;
    for (int option : dynamic_inst.get_options()) {
      if (option == NOINC_OPTION)
        increment_pending = false;
    }

    const active_mask_t &active_mask = inst.get_active_mask();
    const unsigned warp_size = m_shader_ctx->get_warp_size();
    for (unsigned lane = 0; lane < warp_size; ++lane) {
      if (!active_mask.test(lane))
        continue;

      const auto &mbarrier_info = dynamic_inst.get_mbarrier_info(lane);
      assert(mbarrier_info.bar_id != (unsigned)-1);
      const uint32_t mbarrier_addr = mbarrier_info.bar_id;
      m_barriers->prepare_mbarrier_async_arrival(cta_id, warp_id, mbarrier_addr,
                                                 increment_pending);

      std::set<unsigned> pending_tx_uids;
      // ponytail: Scan live transactions here; add a per-lane index only if
      // profiling shows this marker becoming a bottleneck.
      for (const auto &entry : m_cp_transactions) {
        const auto &tx = entry.second;
        if (tx.cta_id == cta_id && tx.warp_id == warp_id &&
            tx.access.get_warp_mask().test(lane)) {
          pending_tx_uids.insert(entry.first);
        }
      }

      if (pending_tx_uids.empty()) {
        m_barriers->arrive_mbarrier_async(cta_id, warp_id, mbarrier_addr);
        continue;
      }

      const unsigned arrival_uid = m_next_cp_mbarrier_arrival_uid++;
      auto inserted = m_cp_mbarrier_arrivals.emplace(
          arrival_uid,
          cp_async_mbarrier_arrival_t{cta_id, warp_id, mbarrier_addr,
                                      std::move(pending_tx_uids)});
      assert(inserted.second);
      for (unsigned tx_uid : inserted.first->second.pending_tx_uids) {
        m_cp_transactions.at(tx_uid).dependent_arrival_uids.insert(arrival_uid);
      }
    }
  }

  void commit_cp_async_group(unsigned cta_id, unsigned warp_id) {
    m_cp_group_info[std::make_pair(cta_id, warp_id)].commit_group();
  }

  void wait_cp_async_group(unsigned cta_id, unsigned warp_id,
                           unsigned max_pending_groups) {
    auto key = std::make_pair(cta_id, warp_id);
    g_cp_async_wait_calls.fetch_add(1, std::memory_order_relaxed);
    m_barriers->wait_cp_async_group(warp_id);
    auto it = m_cp_group_info.find(key);
    if (it == m_cp_group_info.end()) {
      g_cp_async_wait_immediate.fetch_add(1, std::memory_order_relaxed);
      m_barriers->release_cp_async_warp(warp_id);
      return;
    }
    bool satisfied = it->second.wait_group(max_pending_groups);
    if (satisfied) {
      g_cp_async_wait_immediate.fetch_add(1, std::memory_order_relaxed);
      m_barriers->release_cp_async_warp(warp_id);
    } else {
      g_cp_async_wait_blocked.fetch_add(1, std::memory_order_relaxed);
    }
  }

  void cycle() {
    const auto *config = m_shader_ctx->get_config();
    const unsigned request_bytes_per_cycle =
        config->gpgpu_tma_request_bytes_per_cycle;
    const uint32_t request_granularity =
        effective_tma_request_granularity(config);
    if (request_bytes_per_cycle > 0) {
      const unsigned request_width =
          config->gpgpu_tma_request_width ? config->gpgpu_tma_request_width : 1;
      const unsigned credit_cap =
          std::max(request_bytes_per_cycle,
                   request_width * static_cast<unsigned>(request_granularity));
      m_request_byte_credit =
          std::min(m_request_byte_credit + request_bytes_per_cycle, credit_cap);
    }

    // Process delayed arrive_tx notifications
    if (!m_pending_arrives.empty()) {
      for (auto &entry : m_pending_arrives) {
        entry.remaining--;
      }
      for (int i = m_pending_arrives.size() - 1; i >= 0; i--) {
        if (m_pending_arrives[i].remaining == 0) {
          auto &entry = m_pending_arrives[i];
          tma_trace_emit(current_cycle(), "ARRIVE", entry.tx_uid,
                         entry.is_write ? "WRITE" : "READ", 0, 0, entry.cta_id,
                         entry.warp_id, 0, 0, 0, 0, entry.size_in_bytes,
                         entry.mbar_addr, 0, 0, 0, 0, 0, entry.size_in_bytes, 0,
                         m_mf_inflight, m_response_fifo.size());
          if (!entry.is_write) {
            m_barriers->complete_tx(entry.cta_id, entry.warp_id,
                                    entry.mbar_addr, entry.size_in_bytes);
          } else {
            m_barriers->complete_bulk_tx(entry.cta_id, entry.warp_id,
                                         entry.tx_uid);
          }
          m_pending_arrives.erase(m_pending_arrives.begin() + i);
        }
      }
    }

    unsigned cp_waiting_warps = 0;
    for (const auto &entry : m_cp_group_info) {
      if (entry.second.is_waiting)
        cp_waiting_warps++;
    }
    if (cp_waiting_warps > 0) {
      g_cp_async_waiting_warp_cycles.fetch_add(cp_waiting_warps,
                                               std::memory_order_relaxed);
    }

    // Process pending async-copy/TMA responses.
    unsigned response_width =
        m_shader_ctx->get_config()->gpgpu_tma_response_width
            ? m_shader_ctx->get_config()->gpgpu_tma_response_width
            : 1;
    unsigned cp_response_width =
        m_shader_ctx->get_config()->gpgpu_cp_async_response_width
            ? m_shader_ctx->get_config()->gpgpu_cp_async_response_width
            : 1;
    unsigned tma_responses = 0;
    unsigned cp_responses = 0;
    if (!m_response_fifo.empty()) {
      g_cp_async_response_fifo_nonempty_cycles.fetch_add(
          1, std::memory_order_relaxed);
      atomic_update_max(g_cp_async_max_response_fifo, m_response_fifo.size());
    }
    while (!m_response_fifo.empty()) {
      mem_fetch *mf = m_response_fifo.front();
      if (mf->get_access_type() == CP_ASYNC_ACC_R) {
        if (cp_responses >= cp_response_width) {
          g_cp_async_response_width_limited_cycles.fetch_add(
              1, std::memory_order_relaxed);
          break;
        }
        m_response_fifo.pop_front();
        process_cp_async_response(mf);
        cp_responses++;
        continue;
      }
      if (tma_responses >= response_width)
        break;
      m_response_fifo.pop_front();
      tma_responses++;
      assert((mf->get_access_type() == TMA_ACC_R ||
              mf->get_access_type() == TMA_ACC_W) &&
             "unexpected response type in TMA/cp.async unit");

      mem_fetch *parent_mf = mf->get_original_mf() ? mf->get_original_mf() : mf;
      unsigned parent_uid = parent_mf->get_request_uid();
      auto mf_it = m_mf_to_tx.find(parent_uid);
      assert(mf_it != m_mf_to_tx.end() &&
             "TMA response has no live transaction");
      unsigned tx_uid = mf_it->second;
      auto tx_it = m_transactions.find(tx_uid);

      assert(tx_it != m_transactions.end());
      auto &tx = tx_it->second;
      tx.m_mf_received_count++;

      bool is_write = is_write_transaction(tx);
      unsigned long long response_cycle = current_cycle();
      if (tma_trace_mf_enabled(response_cycle)) {
        tma_trace_emit(response_cycle, "MF_RESPONSE", tx_uid,
                       is_write ? "WRITE" : "READ", tx.m_static_info.tma_type,
                       tx.m_pc, tx.m_cta_id, tx.m_warp_id, tx.m_lane_id,
                       tx.m_tid, tx.m_dyn_info.src_addr, tx.m_dyn_info.dst_addr,
                       tx.m_dyn_info.size_in_bytes, tx.m_dyn_info.mbar_addr,
                       parent_uid, parent_mf->get_addr(), mf->get_data_size(),
                       tx.m_mf_issued_count, tx.m_mf_received_count,
                       tx.m_bytes_completed, tx.m_mf_tx_inflight, m_mf_inflight,
                       m_response_fifo.size());
      }
      if (tx.m_first_response_cycle == 0) {
        tx.m_first_response_cycle = response_cycle;
        tma_trace_emit(tx.m_first_response_cycle, "FIRST_RESPONSE", tx_uid,
                       is_write ? "WRITE" : "READ", tx.m_static_info.tma_type,
                       tx.m_pc, tx.m_cta_id, tx.m_warp_id, tx.m_lane_id,
                       tx.m_tid, tx.m_dyn_info.src_addr, tx.m_dyn_info.dst_addr,
                       tx.m_dyn_info.size_in_bytes, tx.m_dyn_info.mbar_addr,
                       parent_uid, parent_mf->get_addr(), mf->get_data_size(),
                       tx.m_mf_issued_count, tx.m_mf_received_count,
                       tx.m_bytes_completed, tx.m_mf_tx_inflight, m_mf_inflight,
                       m_response_fifo.size());
      }

      GPPRINTF_TMA(TMA,
                   "TMA %s response received for mf uid=%u, tx_uid=%u, "
                   "data_size=%u, response fifo size=%lu\n",
                   is_write ? "WRITE" : "READ", mf->get_request_uid(), tx_uid,
                   mf->get_data_size(), m_response_fifo.size());

      if (is_write) {
        // TMA Write (shared → global): response is write acknowledgment
        assert(mf->get_access_type() == TMA_ACC_W);
      } else {
        // TMA Read (global → shared): response contains read data
        assert(mf->get_access_type() == TMA_ACC_R);

        // Validate destination space
        if (tx.m_static_info.dst_space !=
                inst_t::tma_static_info_t::TMA_SHARED_CTA &&
            tx.m_static_info.dst_space !=
                inst_t::tma_static_info_t::TMA_SHARED_CLUSTER) {
          assert(false && "Unrecognized TMA destination space");
        }
      }

      // L2 returns whole sectors, but edge sectors can contain fewer valid
      // bytes than their 32-byte packet size.
      unsigned bytes_to_add = response_payload_bytes(mf, parent_mf);
      tx.m_bytes_completed += bytes_to_add;
      record_tma_mf_response(is_write, bytes_to_add);

      bool parent_complete = false;
      auto pending_it = m_mf_pending_bytes.find(parent_uid);
      assert(pending_it != m_mf_pending_bytes.end());
      if (bytes_to_add >= pending_it->second) {
        assert(m_mf_inflight > 0);
        m_mf_inflight--;
        assert(tx.m_mf_tx_inflight > 0);
        tx.m_mf_tx_inflight--;
        m_mf_pending_bytes.erase(pending_it);
        m_mf_to_tx.erase(parent_uid);
        parent_complete = true;
      } else {
        pending_it->second -= bytes_to_add;
      }

      // Check if transaction is complete
      if (tx.m_bytes_completed >= tx.m_dyn_info.size_in_bytes)
        finalize_transaction(tx_uid);

      mem_fetch *parent_to_delete =
          (parent_complete && parent_mf != mf) ? parent_mf : NULL;
      delete mf;
      delete parent_to_delete;
    }

    issue_cp_async_requests();

    // Issue memory requests using shadow stride accumulation
    if (!issue_queue.empty()) {
      // Check in-flight mem_fetch limit (0 = unlimited)
      // Count truly in-flight requests: issued minus received across all
      // active transactions.  m_mf_to_tx.size() cannot be used because
      // entries are only bulk-erased on finalize, so it over-counts.
      unsigned max_inflight =
          m_shader_ctx->get_config()->gpgpu_tma_max_inflight;
      if (max_inflight > 0 && m_mf_inflight >= max_inflight)
        return;

      const unsigned tx_quota = m_shader_ctx->get_config()->gpgpu_tma_tx_quota;
      const unsigned quota_segment_bytes =
          m_shader_ctx->get_config()->gpgpu_tma_quota_segment_bytes;
      const auto effective_tx_quota = [tx_quota, quota_segment_bytes](
                                          const tma_transaction_t &candidate) {
        if (tx_quota == 0 || quota_segment_bytes == 0)
          return tx_quota;
        const unsigned segments = std::max(
            1u, (candidate.m_dyn_info.size_in_bytes + quota_segment_bytes - 1) /
                    quota_segment_bytes);
        return tx_quota * segments;
      };
      bool borrowing_quota = false;
      unsigned tx_uid = 0;
      tma_transaction_t *selected_tx = nullptr;
      const unsigned num_candidates = issue_queue.size();

      for (unsigned attempt = 0; attempt < num_candidates; ++attempt) {
        tx_uid = issue_queue.front();
        auto it = m_transactions.find(tx_uid);
        assert(it != m_transactions.end());
        auto &candidate = it->second;

        const bool over_quota =
            tx_quota > 0 &&
            candidate.m_mf_tx_inflight >= effective_tx_quota(candidate);
        if (over_quota) {
          issue_queue.pop_front();
          issue_queue.push_back(tx_uid);
          continue;
        }
        selected_tx = &candidate;
        break;
      }

      if (selected_tx == nullptr) {
        // Legacy unsegmented quotas are soft fairness targets: once every
        // transaction reaches its quota, keep the TMA unit busy under the
        // SM-wide limit. Segmented quotas model hard transaction credits.
        if (quota_segment_bytes > 0)
          return;
        borrowing_quota = true;
        tx_uid = issue_queue.front();
        auto it = m_transactions.find(tx_uid);
        assert(it != m_transactions.end());
        selected_tx = &it->second;
      }
      auto &tx = *selected_tx;

      bool is_write =
          (tx.m_static_info.dst_space == inst_t::tma_static_info_t::TMA_GLOBAL);
      unsigned packet_size = is_write ? WRITE_PACKET_SIZE : READ_PACKET_SIZE;
      bool made_progress = false;
      bool transaction_finalized = false;

      if (!m_icnt->full(packet_size, is_write)) {
        uint64_t addr;
        uint32_t size;
        bool oob_l2 = m_shader_ctx->get_config()->gpgpu_tma_oob_l2_traffic;
        const unsigned request_width =
            m_shader_ctx->get_config()->gpgpu_tma_request_width
                ? m_shader_ctx->get_config()->gpgpu_tma_request_width
                : 1;
        unsigned issued_requests = 0;

        // Loop: batch-consume consecutive skip-OOB requests in one cycle,
        // then issue up to request_width mem_fetches to L2 before breaking.
        while (issued_requests < request_width && !transaction_finalized) {
          if (max_inflight > 0 && m_mf_inflight >= max_inflight)
            break;
          if (!borrowing_quota && tx_quota > 0 &&
              tx.m_mf_tx_inflight >= effective_tx_quota(tx))
            break;
          if (m_icnt->full(packet_size, is_write))
            break;
          if (request_bytes_per_cycle > 0 &&
              m_request_byte_credit < request_granularity)
            break;

          bool issued_this_iteration = false;
          while (m_agu.gen_next_req(tx.agu_state, addr, size,
                                    request_granularity)) {
            made_progress = true;

            bool first_request = tx.m_mf_issued_count == 0;
            if (first_request) {
              GPPRINTF_TMA(TMA,
                           "[TMA AGU] tx_uid=%u starting to issue %s "
                           "mem_fetch requests\n",
                           tx_uid, is_write ? "WRITE" : "READ");
              fflush(stdout);
            }
            tx.m_mf_issued_count++;

            // Determine if this OOB request should skip L2:
            //   - Writes: always skip OOB (must not write beyond tensor bounds)
            //   - Reads:  skip OOB only when gpgpu_tma_oob_l2_traffic is off
            bool skip_l2 =
                tx.agu_state.is_fill_request && (is_write || !oob_l2);

            if (skip_l2) {
              // No interconnect needed; just count bytes.
              if (request_bytes_per_cycle > 0)
                m_request_byte_credit -= std::min<unsigned>(
                    m_request_byte_credit, std::max(size, request_granularity));
              tx.m_bytes_completed += size;
              record_tma_bytes_completed(size);
              if (tx.m_bytes_completed >= tx.m_dyn_info.size_in_bytes) {
                finalize_transaction(tx_uid);
                transaction_finalized = true;
                break;
              }
              continue;
            }

            // Issue mem_fetch to interconnect -> L2.
            // For OOB fill requests with gpgpu_tma_oob_l2_traffic=1:
            // Real HW TMA sends full-tile requests to L2 unconditionally,
            // then zero-fills OOB data after response.
            mem_access_byte_mask_t byte_mask;
            mem_access_sector_mask_t sector_mask;

            unsigned start_byte = addr % MAX_MEMORY_ACCESS_SIZE;
            for (unsigned i = 0; i < size; i++) {
              byte_mask.set((start_byte + i) % MAX_MEMORY_ACCESS_SIZE);
            }

            unsigned start_sector = start_byte / SECTOR_SIZE;
            unsigned end_sector = (start_byte + size - 1) / SECTOR_SIZE;
            for (unsigned i = start_sector;
                 i <= end_sector && i < SECTOR_CHUNCK_SIZE; i++) {
              sector_mask.set(i);
            }

            active_mask_t active_mask;

            mem_access_type access_type = is_write ? TMA_ACC_W : TMA_ACC_R;

            mem_access_t access(access_type, addr, size, is_write, active_mask,
                                byte_mask, sector_mask,
                                m_shader_ctx->get_gpu()->gpgpu_ctx);

            mem_fetch *mf = m_mf_allocator->alloc(
                access,
                m_shader_ctx->get_gpu()->gpu_sim_cycle +
                    m_shader_ctx->get_gpu()->gpu_tot_sim_cycle,
                (unsigned long long)-1);

            m_mf_to_tx.emplace(mf->get_request_uid(), tx_uid);
            m_mf_pending_bytes.emplace(mf->get_request_uid(), size);

            unsigned long long issue_cycle = current_cycle();
            if (tma_trace_mf_enabled(issue_cycle)) {
              tma_trace_emit(
                  issue_cycle, "MF_ISSUE", tx_uid, is_write ? "WRITE" : "READ",
                  tx.m_static_info.tma_type, tx.m_pc, tx.m_cta_id, tx.m_warp_id,
                  tx.m_lane_id, tx.m_tid, tx.m_dyn_info.src_addr,
                  tx.m_dyn_info.dst_addr, tx.m_dyn_info.size_in_bytes,
                  tx.m_dyn_info.mbar_addr, mf->get_request_uid(), addr, size,
                  tx.m_mf_issued_count, tx.m_mf_received_count,
                  tx.m_bytes_completed, tx.m_mf_tx_inflight, m_mf_inflight,
                  m_response_fifo.size());
            }
            if (first_request) {
              tx.m_first_issue_cycle = issue_cycle;
              tma_trace_emit(
                  issue_cycle, "FIRST_MF_ISSUE", tx_uid,
                  is_write ? "WRITE" : "READ", tx.m_static_info.tma_type,
                  tx.m_pc, tx.m_cta_id, tx.m_warp_id, tx.m_lane_id, tx.m_tid,
                  tx.m_dyn_info.src_addr, tx.m_dyn_info.dst_addr,
                  tx.m_dyn_info.size_in_bytes, tx.m_dyn_info.mbar_addr,
                  mf->get_request_uid(), addr, size, tx.m_mf_issued_count,
                  tx.m_mf_received_count, tx.m_bytes_completed,
                  tx.m_mf_tx_inflight, m_mf_inflight, m_response_fifo.size());
            }
            tx.m_last_issue_cycle = issue_cycle;

            m_icnt->push(mf);
            m_mf_inflight++;
            tx.m_mf_tx_inflight++;
            record_tma_mf_issued(is_write, size);
            if (request_bytes_per_cycle > 0)
              m_request_byte_credit -=
                  std::min<unsigned>(m_request_byte_credit, size);
            issued_requests++;
            issued_this_iteration = true;
            break;
          }

          if (!issued_this_iteration)
            break;
        }
      }

      // Remove from issue queue when all requests have been issued
      if (transaction_finalized) {
        issue_queue.pop_front();
      } else if (tx.agu_state.done) {
        tma_trace_emit(current_cycle(), "ISSUE_DONE", tx_uid,
                       is_write ? "WRITE" : "READ", tx.m_static_info.tma_type,
                       tx.m_pc, tx.m_cta_id, tx.m_warp_id, tx.m_lane_id,
                       tx.m_tid, tx.m_dyn_info.src_addr, tx.m_dyn_info.dst_addr,
                       tx.m_dyn_info.size_in_bytes, tx.m_dyn_info.mbar_addr, 0,
                       0, 0, tx.m_mf_issued_count, tx.m_mf_received_count,
                       tx.m_bytes_completed, tx.m_mf_tx_inflight, m_mf_inflight,
                       m_response_fifo.size());
        GPPRINTF_TMA(TMA,
                     "[TMA AGU DONE] tx_uid=%u issued %u %s mem_fetch requests "
                     "(total bytes: %u)\n",
                     tx_uid, tx.m_mf_issued_count, is_write ? "WRITE" : "READ",
                     tx.m_dyn_info.size_in_bytes);
        fflush(stdout);
        issue_queue.pop_front();
      } else if (borrowing_quota && made_progress) {
        issue_queue.pop_front();
        issue_queue.push_back(tx_uid);
      }
    }
  }

  void fill(mem_fetch *mf) {
    mf->set_status(IN_TMA_RESPONSE_FIFO,
                   m_shader_ctx->get_gpu()->gpu_sim_cycle +
                       m_shader_ctx->get_gpu()->gpu_tot_sim_cycle);
    m_response_fifo.push_back(mf);
    atomic_update_max(g_cp_async_max_response_fifo, m_response_fifo.size());
  }

  bool response_buffer_full() const {
    // ! assume infinite buffer for simplicity
    return false;
  }

  bool has_pending_for_cta(unsigned cta_id) const {
    for (const auto &entry : m_transactions) {
      if (entry.second.m_cta_id == cta_id)
        return true;
    }
    for (const auto &entry : m_cp_transactions) {
      if (entry.second.cta_id == cta_id)
        return true;
    }
    for (const auto &entry : m_pending_arrives) {
      if (entry.cta_id == cta_id)
        return true;
    }
    for (const auto &entry : m_cp_group_info) {
      if (entry.first.first == cta_id && entry.second.has_pending())
        return true;
    }
    return false;
  }

  void cleanup_cta(unsigned cta_id) {
    for (auto it = m_cp_group_info.begin(); it != m_cp_group_info.end();) {
      if (it->first.first == cta_id) {
        it = m_cp_group_info.erase(it);
      } else {
        ++it;
      }
    }
  }
};

tma_unit_t::tma_unit_t(shader_core_ctx *shader_ctx, barrier_set_t *barriers,
                       mem_fetch_interface *icnt,
                       shader_core_mem_fetch_allocator *mf_allocator)
    : m_impl(std::make_unique<tma_unit_impl_t>(shader_ctx, barriers, icnt,
                                               mf_allocator)) {}

tma_unit_t::~tma_unit_t() = default;

void tma_unit_t::warp_reaches_tma(unsigned cta_id, unsigned warp_id,
                                  warp_inst_t *inst) {
  m_impl->warp_reaches_tma(cta_id, warp_id, inst);
}

void tma_unit_t::warp_reaches_cp_async(unsigned cta_id, unsigned warp_id,
                                       const warp_inst_t &inst,
                                       const ptx_instruction *static_inst) {
  m_impl->warp_reaches_cp_async(cta_id, warp_id, inst, static_inst);
}

void tma_unit_t::warp_reaches_cp_async_mbarrier_arrive(
    unsigned cta_id, unsigned warp_id, const warp_inst_t &inst,
    const ptx_instruction &dynamic_inst) {
  m_impl->warp_reaches_cp_async_mbarrier_arrive(cta_id, warp_id, inst,
                                                dynamic_inst);
}

void tma_unit_t::commit_cp_async_group(unsigned cta_id, unsigned warp_id) {
  m_impl->commit_cp_async_group(cta_id, warp_id);
}

void tma_unit_t::wait_cp_async_group(unsigned cta_id, unsigned warp_id,
                                     unsigned max_pending_groups) {
  m_impl->wait_cp_async_group(cta_id, warp_id, max_pending_groups);
}

void tma_unit_t::cleanup_cta(unsigned cta_id) { m_impl->cleanup_cta(cta_id); }

void tma_unit_t::cycle() { m_impl->cycle(); }

void tma_unit_t::fill(mem_fetch *mf) { m_impl->fill(mf); }

bool tma_unit_t::response_buffer_full() const {
  return m_impl->response_buffer_full();
}

bool tma_unit_t::has_pending_for_cta(unsigned cta_id) const {
  return m_impl->has_pending_for_cta(cta_id);
}

} // namespace flash_gpgpu_sim

//=============================================================================
// Functional Simulation: Helper Functions
//=============================================================================

struct tma_request_t {
  uint64_t global_addr;
  uint64_t tile_offset;
  uint32_t global_size;
  uint32_t smem_size;
};

static void compute_tile_strides(const tensormap_descriptor_t &tensormap,
                                 uint64_t tile_strides[5]) {
  uint32_t num_dims = tensormap.num_dims();

  tile_strides[0] = 0;
  if (num_dims > 1) {
    tile_strides[1] =
        tensormap.get_dim0_smem_span_bytes(tensormap.fields.boxDim[0]);
  }
  for (uint32_t d = 2; d < num_dims; d++) {
    tile_strides[d] = tile_strides[d - 1] * tensormap.fields.boxDim[d - 1];
  }
}

// Apply TMA swizzle transformation to shared memory address
//
// Mask-based implementation of the PTX tensor-map swizzle tables. Shared
// memory is divided into 128B lines; each mode XORs a line-dependent value
// into the chunk index. The base row accounts for a destination address that
// is not aligned to the full repeating pattern.
static uint64_t apply_tma_swizzle(uint64_t linear_offset,
                                  uint32_t smem_base_addr,
                                  uint32_t swizzle_mode, uint32_t row_bytes) {
  if (swizzle_mode == TMA_SWIZZLE_NONE)
    return linear_offset;

  uint32_t mask = 0;
  uint32_t shift = 4;
  bool flip_8b = false;

  switch (swizzle_mode) {
  case TMA_SWIZZLE_128B:
    mask = 0x7;
    break; // 3 bits, cycle of 8
  case TMA_SWIZZLE_64B:
    mask = 0x3;
    break; // 2 bits, cycle of 4
  case TMA_SWIZZLE_32B:
    mask = 0x1;
    break; // 1 bit, cycle of 2
  case TMA_SWIZZLE_96B:
    assert(row_bytes <= 96);
    mask = 0x1;
    break;
  case TMA_SWIZZLE_128B_ATOM_32B:
    mask = 0x3;
    shift = 5;
    break;
  case TMA_SWIZZLE_128B_ATOM_32B_FLIP_8B:
    mask = 0x3;
    shift = 5;
    flip_8b = true;
    break;
  case TMA_SWIZZLE_128B_ATOM_64B:
    mask = 0x1;
    shift = 6;
    break;
  default:
    printf("ERROR: Unknown TMA swizzle mode %u\n", swizzle_mode);
    abort();
  }

  const uint32_t row = (uint32_t)(linear_offset >> 7);
  const uint32_t base_row = (smem_base_addr >> 7) & mask;
  const uint32_t swizzle_row = (row + base_row) & mask;

  uint64_t swizzled = linear_offset ^ ((uint64_t)swizzle_row << shift);
  if (flip_8b && (swizzle_row & 1))
    swizzled ^= 8;
  return swizzled;
}

// Generate 128B-aligned memory fetch requests
// Splits a contiguous memory range into cache-line-aligned requests
static void gen_aligned_req(uint64_t start_addr, uint64_t start_tile_offset,
                            uint32_t total_bytes,
                            std::vector<tma_request_t> &requests) {
  if (total_bytes == 0)
    return;

  constexpr uint32_t CACHE_LINE_SIZE = 128;
  uint64_t current_addr = start_addr;
  uint64_t current_tile_offset = start_tile_offset;
  uint32_t remaining_bytes = total_bytes;

  while (remaining_bytes > 0) {
    // Calculate bytes to next cache line boundary
    uint64_t next_boundary =
        (current_addr + CACHE_LINE_SIZE) & ~(CACHE_LINE_SIZE - 1);
    uint32_t bytes_to_boundary =
        static_cast<uint32_t>(next_boundary - current_addr);

    // Determine request size: min(remaining, bytes_to_boundary,
    // CACHE_LINE_SIZE)
    uint32_t request_size =
        std::min({remaining_bytes, bytes_to_boundary, CACHE_LINE_SIZE});

    requests.push_back(
        {current_addr, current_tile_offset, request_size, request_size});

    current_addr += request_size;
    current_tile_offset += request_size;
    remaining_bytes -= request_size;
  }
}

// ALIGN16B sub-byte formats are dense in global memory. Each group is copied
// into a 16-byte shared-memory container with trailing padding. Tensor-map
// dimensions and coordinates are constrained to groups of 16 elements.
static void gen_align16_padded_req(uint64_t start_addr,
                                   uint64_t start_tile_offset,
                                   uint32_t logical_elements,
                                   uint32_t tensor_data_type,
                                   std::vector<tma_request_t> &requests) {
  assert(logical_elements % 16 == 0 &&
         "ALIGN16B TMA spans must contain complete 16-element groups");
  const uint32_t packed_group_bytes =
      tensor_data_type == TMA_DTYPE_16U4_ALIGN16B ? 8 : 12;
  const uint32_t num_groups = logical_elements / 16;
  for (uint32_t group = 0; group < num_groups; ++group) {
    requests.push_back({start_addr + group * packed_group_bytes,
                        start_tile_offset + group * 16, packed_group_bytes,
                        16});
  }
}

// Recursive helper to traverse tensor dimensions and generate memory requests
// dim: current dimension being processed (Rank-1 down to 0)
// current_coords: accumulated coordinates from higher dimensions
// base_addr: current physical address
// tensormap: tensor descriptor
// requests: output vector of (address, size) pairs
static void traverse_tensor_dim(int dim, const int32_t current_coords[5],
                                uint64_t base_addr,
                                const tensormap_descriptor_t &tensormap,
                                const uint64_t tile_strides[5],
                                uint64_t tile_offset,
                                std::vector<tma_request_t> &requests) {
  if (dim == 0) {
    // Base case: innermost dimension (contiguous)
    int64_t start_x = current_coords[0];
    uint32_t box_width = tensormap.fields.boxDim[0];
    uint32_t global_width = tensormap.fields.globalDim[0];

    // Calculate valid range intersection: [start_x, start_x + box_width) ∩ [0,
    // global_width)
    int64_t valid_start = start_x;
    int64_t valid_end = start_x + box_width;

    // Check if completely out of bounds
    if (valid_start >= global_width || valid_end <= 0) {
      return; // No valid data, skip
    }

    // Clamp to valid tensor boundaries
    valid_start = std::max<int64_t>(valid_start, 0);
    valid_end = std::min<int64_t>(valid_end, global_width);

    if (valid_start >= valid_end)
      return;

    // Calculate physical start address and total bytes
    uint64_t phys_start_addr =
        base_addr +
        tensormap.get_dim0_gmem_byte_offset(static_cast<uint64_t>(valid_start));
    uint64_t valid_tile_offset =
        tile_offset + tensormap.get_dim0_smem_byte_offset(
                          static_cast<uint64_t>(valid_start - start_x));
    const uint32_t valid_elements =
        static_cast<uint32_t>(valid_end - valid_start);
    uint32_t valid_bytes = static_cast<uint32_t>(
        tensormap.get_dim0_gmem_span_bytes(valid_elements));

    // Generate aligned memory fetch requests
    if (tensormap.fields.tensorDataType == TMA_DTYPE_16U4_ALIGN16B ||
        tensormap.fields.tensorDataType == TMA_DTYPE_16U6_ALIGN16B) {
      assert(valid_start % 16 == 0 &&
             "ALIGN16B TMA coordinates must be 16-element aligned");
      gen_align16_padded_req(phys_start_addr, valid_tile_offset, valid_elements,
                             tensormap.fields.tensorDataType, requests);
    } else {
      gen_aligned_req(phys_start_addr, valid_tile_offset, valid_bytes,
                      requests);
    }

  } else {
    // Recursive case: traverse higher dimensions
    uint32_t box_extent = tensormap.fields.boxDim[dim];
    uint32_t global_extent = tensormap.fields.globalDim[dim];
    uint64_t stride = tensormap.fields.globalStrides[dim - 1];

    for (uint32_t i = 0; i < box_extent; i++) {
      int64_t current_coord = static_cast<int64_t>(current_coords[dim]) + i;

      // OOB check: skip if outside valid tensor range
      if (current_coord < 0 || current_coord >= global_extent) {
        continue; // Skip this branch (zero-padding)
      }

      // Calculate address offset for this coordinate
      uint64_t next_addr =
          base_addr + static_cast<uint64_t>(current_coord) * stride;
      uint64_t next_tile_offset = tile_offset + i * tile_strides[dim];

      // Recurse to next lower dimension
      traverse_tensor_dim(dim - 1, current_coords, next_addr, tensormap,
                          tile_strides, next_tile_offset, requests);
    }
  }
}

// Generate TMA memory fetch requests for a tensor tile
// start_coords: starting coordinate for each dimension (e.g., {x, y, z, w, v})
// Returns: vector of (physical_address, size_in_bytes) pairs
static std::vector<tma_request_t>
generate_tma_requests(const tensormap_descriptor_t &tensormap,
                      const int32_t start_coords[5]) {
  std::vector<tma_request_t> requests;

  if (!tensormap.is_valid() || tensormap.fields.tensorRank > 4) {
    return requests; // Empty result for invalid tensormap
  }

  // Start recursive traversal from highest dimension (0-based index)
  int highest_dim = static_cast<int>(tensormap.num_dims()) -
                    1; // highest dimension index (0-based)
  uint64_t base_addr = tensormap.fields.globalAddress;
  uint64_t tile_strides[5] = {};
  compute_tile_strides(tensormap, tile_strides);

  traverse_tensor_dim(highest_dim, start_coords, base_addr, tensormap,
                      tile_strides, 0, requests);

  return requests;
}

//=============================================================================
// Functional Simulation: OOB Fill Pattern Table
//=============================================================================

class tma_oob_fill_table_t {
public:
  static constexpr uint32_t CHUNK_SIZE = 128; // Cache line size
  static constexpr uint32_t NUM_DTYPES = 16;  // Max data type index + 1

  // Fill patterns: [oob_mode][dtype] -> 128-byte pattern
  // oob_mode: 0 = ZERO, 1 = NAN
  alignas(16) unsigned char patterns[2][NUM_DTYPES][CHUNK_SIZE];
  bool nan_supported[NUM_DTYPES]; // Whether NaN is valid for this dtype

  tma_oob_fill_table_t() {
    // Initialize all patterns
    for (uint32_t dtype = 0; dtype < NUM_DTYPES; dtype++) {
      // Zero pattern (always valid)
      memset(patterns[TMA_OOB_ZERO][dtype], 0, CHUNK_SIZE);

      // NaN pattern (depends on dtype)
      nan_supported[dtype] = false;
      switch (dtype) {
      case TMA_DTYPE_F32: {
        nan_supported[dtype] = true;
        uint32_t nan_val = 0x7FFFFFFF;
        for (uint32_t i = 0; i < CHUNK_SIZE / sizeof(uint32_t); i++) {
          memcpy(patterns[TMA_OOB_NAN][dtype] + i * sizeof(uint32_t), &nan_val,
                 sizeof(uint32_t));
        }
        break;
      }
      case TMA_DTYPE_F64: {
        nan_supported[dtype] = true;
        uint64_t nan_val = 0x7FFFFFFFFFFFFFFFULL;
        for (uint32_t i = 0; i < CHUNK_SIZE / sizeof(uint64_t); i++) {
          memcpy(patterns[TMA_OOB_NAN][dtype] + i * sizeof(uint64_t), &nan_val,
                 sizeof(uint64_t));
        }
        break;
      }
      case TMA_DTYPE_F16:
      case TMA_DTYPE_BF16: {
        nan_supported[dtype] = true;
        uint16_t nan_val = 0x7FFF;
        for (uint32_t i = 0; i < CHUNK_SIZE / sizeof(uint16_t); i++) {
          memcpy(patterns[TMA_OOB_NAN][dtype] + i * sizeof(uint16_t), &nan_val,
                 sizeof(uint16_t));
        }
        break;
      }
      case TMA_DTYPE_U8: {
        // Treat as FP8 (E4M3/E5M2) NaN: 0x7F
        nan_supported[dtype] = true;
        memset(patterns[TMA_OOB_NAN][dtype], 0x7F, CHUNK_SIZE);
        break;
      }
      default:
        // Integer types: NaN not supported, fall back to zero
        nan_supported[dtype] = false;
        memset(patterns[TMA_OOB_NAN][dtype], 0, CHUNK_SIZE);
        break;
      }
    }
  }

  // Get fill pattern for given oob mode and data type
  const unsigned char *get_pattern(uint32_t oob_mode, uint32_t dtype) const {
    if (dtype >= NUM_DTYPES)
      dtype = 0; // Safety bound
    if (oob_mode == TMA_OOB_NAN && !nan_supported[dtype]) {
      static bool warned[NUM_DTYPES] = {false};
      if (!warned[dtype]) {
        printf("TMA ERROR: OOB_NAN not supported for data type %u, using zero "
               "fill\n",
               dtype);
        warned[dtype] = true;
      }
      return patterns[TMA_OOB_ZERO][dtype];
    }
    return patterns[oob_mode][dtype];
  }
};

// Global static instance (initialized at startup)
static tma_oob_fill_table_t g_oob_fill_table;

//=============================================================================
// Functional Simulation: TMA Data Transfer
//=============================================================================

static void expand_align16_subbytes(uint32_t tensor_data_type,
                                    const unsigned char *packed,
                                    uint32_t packed_size, unsigned char *padded,
                                    uint32_t padded_size) {
  assert(padded_size == 16);
  const uint32_t expected_packed_size =
      tensor_data_type == TMA_DTYPE_16U4_ALIGN16B ? 8 : 12;
  assert((tensor_data_type == TMA_DTYPE_16U4_ALIGN16B ||
          tensor_data_type == TMA_DTYPE_16U6_ALIGN16B) &&
         packed_size == expected_packed_size);
  memcpy(padded, packed, packed_size);
  // PTX leaves this padding uninitialized. Zeroing it gives functional
  // simulation deterministic bytes without changing any logical value.
  memset(padded + packed_size, 0, padded_size - packed_size);
}

static void pack_align16_subbytes(uint32_t tensor_data_type,
                                  const unsigned char *unpacked,
                                  uint32_t unpacked_size, unsigned char *packed,
                                  uint32_t packed_size) {
  assert(unpacked_size == 16);
  memset(packed, 0, packed_size);
  // Encoding 14 (.b4x16_p64) is load-only. Encoding 15 denotes .b6x16_p32
  // on load and .b6p2x16 on store; the latter discards each byte's top bits.
  assert(tensor_data_type == TMA_DTYPE_16U6_ALIGN16B && packed_size == 12);
  for (uint32_t i = 0; i < unpacked_size; ++i) {
    const uint32_t bit_offset = i * 6;
    const uint32_t byte_offset = bit_offset / 8;
    const uint32_t bit_in_byte = bit_offset % 8;
    const uint32_t value = unpacked[i] & 0x3f;
    packed[byte_offset] |= static_cast<unsigned char>(value << bit_in_byte);
    if (bit_in_byte > 2)
      packed[byte_offset + 1] |=
          static_cast<unsigned char>(value >> (8 - bit_in_byte));
  }
}

// Execute TMA data transfer (load or store)
// is_load=true: global -> shared, is_load=false: shared -> global
static void do_tma_transfer(const tensormap_descriptor_t &tensormap,
                            const int32_t coords[5], memory_space *shared_mem,
                            memory_space *global_mem, uint32_t smem_addr,
                            ptx_thread_info *thread, const ptx_instruction *pI,
                            bool is_load) {
  // For load operations, pre-fill the entire tile in shared memory with OOB
  // fill value
  if (is_load) {
    uint32_t tile_size_bytes = tensormap.get_tile_smem_size_bytes();
    const unsigned char *fill_pattern = g_oob_fill_table.get_pattern(
        tensormap.fields.oobFill, tensormap.fields.tensorDataType);

    constexpr uint32_t CHUNK_SIZE = tma_oob_fill_table_t::CHUNK_SIZE;
    uint32_t offset = 0;
    while (offset < tile_size_bytes) {
      uint32_t chunk_size = std::min(tile_size_bytes - offset, CHUNK_SIZE);
      shared_mem->write(smem_addr + offset, chunk_size, fill_pattern, thread,
                        pI);
      offset += chunk_size;
    }
  }

  auto memory_requests = generate_tma_requests(tensormap, coords);
  uint32_t swizzle_mode = tensormap.fields.swizzle;
  uint32_t row_bytes = static_cast<uint32_t>(
      tensormap.get_dim0_smem_span_bytes(tensormap.fields.boxDim[0]));

  for (const auto &req : memory_requests) {
    uint64_t global_req_addr = req.global_addr;
    uint32_t global_req_size = req.global_size;
    uint32_t smem_req_size = req.smem_size;
    uint64_t tile_offset = req.tile_offset;
    const uint32_t swizzle_granularity =
        swizzle_mode == TMA_SWIZZLE_128B_ATOM_32B_FLIP_8B ? 8 : 16;
    std::vector<unsigned char> global_data(global_req_size);
    std::vector<unsigned char> smem_data(smem_req_size);

    if (is_load) {
      // Load: global -> shared
      global_mem->read(global_req_addr, global_req_size, global_data.data());
      if (global_req_size == smem_req_size) {
        memcpy(smem_data.data(), global_data.data(), global_req_size);
      } else {
        expand_align16_subbytes(tensormap.fields.tensorDataType,
                                global_data.data(), global_req_size,
                                smem_data.data(), smem_req_size);
      }

      float debug_fp32 = 0.0f;
      if (global_req_size >= sizeof(debug_fp32))
        memcpy(&debug_fp32, global_data.data(), sizeof(debug_fp32));

      GPPRINTF_INST_EXEC(TMA,
                         "coord[%d,%d,%d,%d,%d] "
                         "swizzle_mode %u "
                         "gmem=0x%llx -> "
                         "smem=0x%x, space=%p, size=%u, tile_offset=0x%llx, "
                         "data[0..3]=0x%02x%02x%02x%02x fp32 %.3f\n",
                         coords[0], coords[1], coords[2], coords[3], coords[4],
                         swizzle_mode, (unsigned long long)global_req_addr,
                         (unsigned)smem_addr, shared_mem, global_req_size,
                         (unsigned long long)tile_offset,
                         global_req_size > 0 ? global_data[0] : 0,
                         global_req_size > 1 ? global_data[1] : 0,
                         global_req_size > 2 ? global_data[2] : 0,
                         global_req_size > 3 ? global_data[3] : 0, debug_fp32);

      if (swizzle_mode != TMA_SWIZZLE_NONE) {
        for (uint32_t sub_offset = 0; sub_offset < smem_req_size;
             sub_offset += swizzle_granularity) {
          uint32_t sub_size =
              std::min(swizzle_granularity, smem_req_size - sub_offset);
          uint64_t logical_offset = tile_offset + sub_offset;
          uint64_t swizzled_offset = apply_tma_swizzle(
              logical_offset, smem_addr, swizzle_mode, row_bytes);

          shared_mem->write(smem_addr + swizzled_offset, sub_size,
                            smem_data.data() + sub_offset, thread, pI);
        }
      } else {
        // No swizzle - write contiguously
        shared_mem->write(smem_addr + tile_offset, smem_req_size,
                          smem_data.data(), thread, pI);
      }

    } else {
      // Store: shared -> global (reverse swizzle)
      if (swizzle_mode != TMA_SWIZZLE_NONE) {
        for (uint32_t sub_offset = 0; sub_offset < smem_req_size;
             sub_offset += swizzle_granularity) {
          uint32_t sub_size =
              std::min(swizzle_granularity, smem_req_size - sub_offset);
          uint64_t logical_offset = tile_offset + sub_offset;
          uint64_t swizzled_offset = apply_tma_swizzle(
              logical_offset, smem_addr, swizzle_mode, row_bytes);

          shared_mem->read(smem_addr + swizzled_offset, sub_size,
                           smem_data.data() + sub_offset);
        }
      } else {
        // No swizzle - read contiguously
        shared_mem->read(smem_addr + tile_offset, smem_req_size,
                         smem_data.data());
      }

      if (global_req_size == smem_req_size) {
        memcpy(global_data.data(), smem_data.data(), global_req_size);
      } else {
        assert(tensormap.fields.tensorDataType == TMA_DTYPE_16U6_ALIGN16B &&
               "TMA .b4x16_p64 does not support shared-to-global copies");
        pack_align16_subbytes(tensormap.fields.tensorDataType, smem_data.data(),
                              smem_req_size, global_data.data(),
                              global_req_size);
      }

      global_mem->write(global_req_addr, global_req_size, global_data.data(),
                        thread, pI);
    }
  }
}

//=============================================================================
// Functional Simulation: TMA Instruction Handlers
//=============================================================================

// Copy data between two memory spaces
static void copy_mem(memory_space *src_mem, uint64_t src_addr,
                     memory_space *dst_mem, uint64_t dst_addr,
                     uint32_t size_in_bytes, ptx_thread_info *thread,
                     const ptx_instruction *pI) {
  unsigned char *buf = new unsigned char[size_in_bytes];
  src_mem->read(src_addr, size_in_bytes, buf);
  dst_mem->write(dst_addr, size_in_bytes, buf, thread, pI);
  delete[] buf;
}

static void reduce_add_f32_mem(memory_space *src_mem, uint64_t src_addr,
                               memory_space *dst_mem, uint64_t dst_addr,
                               uint32_t size_in_bytes, ptx_thread_info *thread,
                               const ptx_instruction *pI) {
  if (size_in_bytes % sizeof(float) != 0) {
    printf("TMA ERROR: f32 reduce size must be a multiple of 4, got %u\n",
           size_in_bytes);
    abort();
  }

  std::vector<float> src(size_in_bytes / sizeof(float));
  std::vector<float> dst(size_in_bytes / sizeof(float));
  src_mem->read(src_addr, size_in_bytes, src.data());
  dst_mem->read(dst_addr, size_in_bytes, dst.data());
  for (size_t i = 0; i < dst.size(); ++i) {
    dst[i] += src[i];
  }
  dst_mem->write(dst_addr, size_in_bytes, dst.data(), thread, pI);
}

// Check 16-byte alignment for TMA addresses and size
static void check_tma_alignment(uint64_t dst_addr, uint64_t src_addr,
                                uint32_t size_in_bytes) {
  if (dst_addr % 16 != 0 || src_addr % 16 != 0 || size_in_bytes % 16 != 0) {
    printf("TMA ERROR: unaligned TMA copy dst=0x%llx, src=0x%llx, "
           "size_in_bytes=%u\n",
           (unsigned long long)dst_addr, (unsigned long long)src_addr,
           size_in_bytes);
    abort();
  }
}

// Read tensormap from memory, validate, parse coordinates, compute size.
// FA3/CUTLASS may pass descriptors embedded in kernel .param storage via
// cvta.param; those addresses are simulator param-memory offsets.
static void
setup_tensor_tma(memory_space *global_mem, uint64_t tensormap_addr,
                 unsigned inst_dim, const operand_info &coord_operand,
                 ptx_thread_info *thread, tensormap_descriptor_t &out_tensormap,
                 int32_t out_coords[5], uint32_t &out_size_in_bytes) {
  using namespace flash_gpgpu_sim;
  read_tensormap_descriptor(global_mem, thread->get_param_memory(),
                            tensormap_addr, inst_dim, out_tensormap);
  validate_tensormap(tensormap_addr, out_tensormap, inst_dim);

  out_size_in_bytes = out_tensormap.get_tile_size_bytes();
  assert(out_size_in_bytes > 0 && "TMA tensor transfer size cannot be zero");

  parse_tensor_coords(thread, coord_operand, out_coords);
}

// Handle TMA linear copy instruction (non-tensor)
static void handle_tma_copy(ptx_instruction *pI, ptx_thread_info *thread) {
  using namespace flash_gpgpu_sim;

  const auto &options = pI->get_options();
  std::vector<int> space_options;
  int completion_option = 0;
  bool reduce_add = false;

  for (auto op : options) {
    switch (op) {
    case GLOBAL_OPTION:
    case CTA_OPTION:
    case CLUSTER_OPTION:
      space_options.push_back(op);
      break;
    case TMA_MBAR_COMPLETE_BYTES:
    case BULK_GROUP_OPTION:
      completion_option = op;
      break;
    case ATOMIC_ADD:
      reduce_add = true;
      break;
    case L2_CACHE_HINT_OPTION:
      break;
    default:
      break;
    }
  }

  if (space_options.size() < 2 || completion_option == 0) {
    printf("TMA ERROR: unsupported linear TMA option list: ");
    for (auto op : options) {
      printf("%d ", op);
    }
    printf("\n");
    pI->print_insn();
    abort();
  }

  auto dst_option = space_options[0];
  auto src_option = space_options[1];

  memory_space *global_mem = thread->get_global_memory();
  memory_space *shared_mem = thread->m_shared_mem;

  if ((dst_option == CTA_OPTION || dst_option == CLUSTER_OPTION) &&
      src_option == GLOBAL_OPTION &&
      completion_option == TMA_MBAR_COMPLETE_BYTES) {
    // shared::cta/shared::cluster <- global with MBAR completion
    auto dst_addr = get_operand_u32(thread, pI->dst());
    auto src_addr = get_operand_u64(thread, pI->src1());
    auto size_in_bytes = get_operand_u32(thread, pI->src2());
    auto mbar_addr = get_operand_u32(thread, pI->src3());

    check_tma_alignment(dst_addr, src_addr, size_in_bytes);

    inst_t::tma_static_info_t tma_static_info{
        .tma_type = inst_t::tma_static_info_t::TMA_NORMAL,
        .dst_space = dst_option == CLUSTER_OPTION
                         ? inst_t::tma_static_info_t::TMA_SHARED_CLUSTER
                         : inst_t::tma_static_info_t::TMA_SHARED_CTA,
        .src_space = inst_t::tma_static_info_t::TMA_GLOBAL,
    };
    pI->set_tma_static_info(tma_static_info);

    inst_t::tma_dyn_info_t tma_dyn_info{
        .dst_addr = dst_addr,
        .src_addr = src_addr,
        .size_in_bytes = size_in_bytes,
        .mbar_addr = mbar_addr,
    };
    pI->set_tma_dyn_info(thread->get_laneid(), tma_dyn_info);

    copy_mem(global_mem, src_addr, shared_mem, dst_addr, size_in_bytes, thread,
             pI);

    GPPRINTF_INST_EXEC(TMA,
                       "Functional Sim: "
                       "TMA shared::%s <- global dst=0x%x, src=0x%llx, "
                       "size_in_bytes=%u, mbar=0x%x\n",
                       dst_option == CLUSTER_OPTION ? "cluster" : "cta",
                       dst_addr, (unsigned long long)src_addr, size_in_bytes,
                       mbar_addr);

  } else if (dst_option == GLOBAL_OPTION && src_option == CTA_OPTION &&
             completion_option == BULK_GROUP_OPTION) {
    // global <- shared::cta with bulk group completion
    auto dst_addr = get_operand_u64(thread, pI->dst());
    auto src_addr = get_operand_u32(thread, pI->src1());
    auto size_in_bytes = get_operand_u32(thread, pI->src2());
    auto laneid = thread->get_laneid();

    auto ctaid = thread->get_ctaid();
    auto warp_id = thread->get_hw_wid();

    GPPRINTF_INST_EXEC(TMA,
                       "[TMA STORE] CTA(%u,%u,%u) warp=%u lane=%u: "
                       "dst=0x%llx, src=0x%x, size=%u\n",
                       ctaid.x, ctaid.y, ctaid.z, warp_id, laneid,
                       (unsigned long long)dst_addr, src_addr, size_in_bytes);

    check_tma_alignment(dst_addr, src_addr, size_in_bytes);

    inst_t::tma_static_info_t tma_static_info{
        .tma_type = inst_t::tma_static_info_t::TMA_NORMAL,
        .dst_space = inst_t::tma_static_info_t::TMA_GLOBAL,
        .src_space = inst_t::tma_static_info_t::TMA_SHARED_CTA,
    };
    pI->set_tma_static_info(tma_static_info);

    inst_t::tma_dyn_info_t tma_dyn_info{
        .dst_addr = dst_addr,
        .src_addr = src_addr,
        .size_in_bytes = size_in_bytes,
    };
    pI->set_tma_dyn_info(laneid, tma_dyn_info);

    if (reduce_add) {
      const auto scalar_types = pI->get_scalar_type();
      if (scalar_types.empty() || scalar_types.front() != F32_TYPE) {
        printf("TMA ERROR: only cp.reduce.async.bulk.add.f32 is supported\n");
        pI->print_insn();
        abort();
      }
      reduce_add_f32_mem(shared_mem, src_addr, global_mem, dst_addr,
                         size_in_bytes, thread, pI);
    } else {
      copy_mem(shared_mem, src_addr, global_mem, dst_addr, size_in_bytes,
               thread, pI);
    }

    GPPRINTF_INST_EXEC(TMA,
                       "Functional Sim: "
                       "TMA global <- shared::cta%s dst=0x%llx, src=0x%x, "
                       "size_in_bytes=%u\n",
                       reduce_add ? " reduce.add.f32" : "",
                       (unsigned long long)dst_addr, src_addr, size_in_bytes);
  } else {
    printf("TMA ERROR: unsupported linear TMA instruction options: dst=%d "
           "src=%d completion=%d reduce_add=%d\n",
           dst_option, src_option, completion_option, reduce_add ? 1 : 0);
    printf("  raw options: ");
    for (auto op : options) {
      printf("%d ", op);
    }
    printf("\n");
    pI->print_insn();
    abort();
  }
}

// Handle cp.async.bulk.commit_group
static void handle_tma_commit_group(ptx_instruction *pI,
                                    ptx_thread_info *thread) {
  inst_t::tma_static_info_t tma_static_info{
      .tma_type = inst_t::tma_static_info_t::TMA_BULK_COMMIT,
      .dst_space = inst_t::tma_static_info_t::TMA_SPACE_INVALID,
      .src_space = inst_t::tma_static_info_t::TMA_SPACE_INVALID,
  };
  pI->set_tma_static_info(tma_static_info);
  GPPRINTF_INST_EXEC(TMA, "Functional Sim: cp.async.bulk.commit_group%s\n", "");
}

// Handle cp.async.bulk.wait_group N
static void handle_tma_wait_group(ptx_instruction *pI,
                                  ptx_thread_info *thread) {
  using namespace flash_gpgpu_sim;
  unsigned group_num = get_operand_u32(thread, pI->dst());
  bool read_only = false;
  for (auto op : pI->get_options()) {
    if (op == READ_OPTION) {
      read_only = true;
      break;
    }
  }

  inst_t::tma_static_info_t tma_static_info{
      .tma_type = inst_t::tma_static_info_t::TMA_BULK_WAIT,
      .dst_space = inst_t::tma_static_info_t::TMA_SPACE_INVALID,
      .src_space = inst_t::tma_static_info_t::TMA_SPACE_INVALID,
      .bulk_wait_num = group_num,
      .bulk_wait_read_only = read_only,
  };
  pI->set_tma_static_info(tma_static_info);
  GPPRINTF_INST_EXEC(TMA, "Functional Sim: cp.async.bulk.wait_group%s %u\n",
                     read_only ? ".read" : "", group_num);
}

// Handle cp.async.bulk.tensor.Nd (tensor load/store)
static void handle_tma_tensor(ptx_instruction *pI, ptx_thread_info *thread) {
  using namespace flash_gpgpu_sim;

  const auto &options = pI->get_options();
  int dim_option = 0;
  int completion_option = 0;
  std::vector<int> space_options;
  for (auto op : options) {
    switch (op) {
    case DIM_1D_OPTION:
    case DIM_2D_OPTION:
    case DIM_3D_OPTION:
    case DIM_4D_OPTION:
    case DIM_5D_OPTION:
      dim_option = op;
      break;
    case GLOBAL_OPTION:
    case CTA_OPTION:
    case CLUSTER_OPTION:
      space_options.push_back(op);
      break;
    case TMA_MBAR_COMPLETE_BYTES:
    case BULK_GROUP_OPTION:
      completion_option = op;
      break;
    case TENSOR_OPTION:
    case TILE_OPTION:
    case L2_CACHE_HINT_OPTION:
      break;
    default:
      break;
    }
  }

  if (dim_option == 0 || space_options.size() < 2 || completion_option == 0) {
    printf("TMA ERROR: unsupported tensor TMA option list: ");
    for (auto op : options) {
      printf("%d ", op);
    }
    printf("\n");
    pI->print_insn();
    abort();
  }

  auto dst_option = space_options[0];
  auto src_option = space_options[1];

  memory_space *shared_mem = thread->m_shared_mem;
  memory_space *global_mem = thread->get_global_memory();
  unsigned inst_dim = compute_inst_dim(dim_option);

  if ((dst_option == CTA_OPTION || dst_option == CLUSTER_OPTION) &&
      src_option == GLOBAL_OPTION &&
      completion_option == TMA_MBAR_COMPLETE_BYTES) {
    // Tensor load: shared <- global
    auto dst_addr = get_operand_u32(thread, pI->dst());
    uint64_t tensormap_addr = get_operand_u64(thread, pI->src1());
    auto mbar_addr = get_operand_u32(thread, pI->src3());

    tensormap_descriptor_t tensormap;
    int32_t coords[5];
    uint32_t size_in_bytes;
    setup_tensor_tma(global_mem, tensormap_addr, inst_dim,
                     pI->get_operands()[2], thread, tensormap, coords,
                     size_in_bytes);

    inst_t::tma_static_info_t tma_static_info{
        .tma_type = inst_t::tma_static_info_t::TMA_TENSOR,
        .dst_space = dst_option == CLUSTER_OPTION
                         ? inst_t::tma_static_info_t::TMA_SHARED_CLUSTER
                         : inst_t::tma_static_info_t::TMA_SHARED_CTA,
        .src_space = inst_t::tma_static_info_t::TMA_GLOBAL,
        .tensor_dim = inst_dim,
    };
    pI->set_tma_static_info(tma_static_info);

    inst_t::tma_dyn_info_t tma_dyn_info{
        .dst_addr = dst_addr,
        .src_addr = tensormap_addr,
        .size_in_bytes = size_in_bytes,
        .mbar_addr = mbar_addr,
    };
    for (unsigned i = 0; i < 5; ++i)
      tma_dyn_info.coords[i] = coords[i];
    cache_tensormap_descriptor(tma_dyn_info, tensormap);
    pI->set_tma_dyn_info(thread->get_laneid(), tma_dyn_info);

    do_tma_transfer(tensormap, coords, shared_mem, global_mem, dst_addr, thread,
                    pI, true);

  } else if (dst_option == GLOBAL_OPTION && src_option == CTA_OPTION) {
    // Tensor store: global <- shared
    assert(completion_option == BULK_GROUP_OPTION &&
           "Only bulk_group completion option is supported for "
           "cp.async.bulk.tensor.Nd.shared::cta.global");

    auto tensormap_addr = get_operand_u64(thread, pI->dst());
    auto src_addr = get_operand_u32(thread, pI->src2());

    tensormap_descriptor_t tensormap;
    int32_t coords[5];
    uint32_t size_in_bytes;
    setup_tensor_tma(global_mem, tensormap_addr, inst_dim,
                     pI->get_operands()[1], thread, tensormap, coords,
                     size_in_bytes);

    GPPRINTF_INST_EXEC(
        TMA, "TMA tensor store Extracted coordinates: [%d, %d, %d, %d, %d]\n",
        coords[0], coords[1], coords[2], coords[3], coords[4]);

    inst_t::tma_static_info_t tma_static_info{
        .tma_type = inst_t::tma_static_info_t::TMA_TENSOR,
        .dst_space = inst_t::tma_static_info_t::TMA_GLOBAL,
        .src_space = inst_t::tma_static_info_t::TMA_SHARED_CTA,
        .tensor_dim = inst_dim,
    };
    pI->set_tma_static_info(tma_static_info);

    inst_t::tma_dyn_info_t tma_dyn_info{
        .dst_addr = tensormap_addr,
        .src_addr = src_addr,
        .size_in_bytes = size_in_bytes,
    };
    for (unsigned i = 0; i < 5; ++i)
      tma_dyn_info.coords[i] = coords[i];
    cache_tensormap_descriptor(tma_dyn_info, tensormap);
    pI->set_tma_dyn_info(thread->get_laneid(), tma_dyn_info);

    do_tma_transfer(tensormap, coords, shared_mem, global_mem, src_addr, thread,
                    pI, false);

    uint64_t base_dst_addr = tensormap.calculate_src_addr(coords);
    GPPRINTF_INST_EXEC(TMA,
                       "Functional Sim: TMA tensor store dst=0x%llx, src=0x%x, "
                       "size=%u, tensormap=0x%llx\n",
                       (unsigned long long)base_dst_addr, src_addr,
                       size_in_bytes, (unsigned long long)tensormap_addr);

  } else {
    reject_unsupported_tma("unsupported cp.async.bulk.tensor variant", pI);
  }
}

//=============================================================================
// Functional Simulation: Entry Point
//=============================================================================

void handle_tma_inst(const ptx_instruction *pIin, ptx_thread_info *thread) {
  using namespace flash_gpgpu_sim;

  ptx_instruction *pI = const_cast<ptx_instruction *>(pIin);

  if (pI->get_opcode() == TMA_PREFETCH_OP) {
    // cp.async.bulk.prefetch is a non-binding cache hint with no
    // architectural result. Accept it as a functional no-op; the
    // performance pipeline still issues it through the configured TMA unit
    // and accounts for the instruction latency.
    return;
  }

  bool is_commit_group = false;
  bool is_wait_group = false;
  bool is_tensor = false;
  for (auto op : pI->get_options()) {
    switch (op) {
    case COMMIT_GROUP_OPTION:
      is_commit_group = true;
      break;
    case WAIT_GROUP_OPTION:
      is_wait_group = true;
      break;
    case TENSOR_OPTION:
      is_tensor = true;
      break;
    default:
      break;
    }
  }

  if (!is_commit_group && !is_wait_group && !is_tensor) {
    // TMA copy instruction (non-tensor linear copy)
    handle_tma_copy(pI, thread);
  } else if (is_commit_group) {
    handle_tma_commit_group(pI, thread);
  } else if (is_wait_group) {
    handle_tma_wait_group(pI, thread);
  } else if (is_tensor) {
    handle_tma_tensor(pI, thread);
  } else {
    GPPRINTF_INST_EXEC(TMA, "Unrecognized TMA instruction%s\n", "");
    pI->print_insn();
    assert(false && "Unrecognized TMA instruction");
  }
}
