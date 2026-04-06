#include "tma.h"
#include "tensormap.h"
#include <atomic>
#include <cstdarg>
#include <cstring>

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

//=============================================================================
// Shared Helper Functions
//=============================================================================

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
                                uint32_t coords[5]) {
  // Initialize all coords to 0
  for (int i = 0; i < 5; i++)
    coords[i] = 0;

  if (coord_operand.is_vector()) {
    unsigned n_coords = coord_operand.get_vect_nelem();
    ptx_reg_t coord_regs[8];
    thread->get_vector_operand_values(coord_operand, coord_regs, n_coords);

    for (unsigned i = 0; i < n_coords && i < 5; i++) {
      coords[i] = coord_regs[i].u32;
    }
  } else {
    // Single coordinate (1D case with non-vector operand)
    coords[0] = thread
                    ->get_operand_value(coord_operand, coord_operand, U32_TYPE,
                                        thread, 0)
                    .u32;
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
  uint32_t start_coords[5] = {0};   // Starting coordinates of this tile
  uint64_t global_strides[5] = {0}; // Strides for each dimension
  uint32_t tile_coords[5] = {0};    // Current position within tile (odometer)
  uint64_t curr_row_addr = 0;       // Base physical address of current row
  uint32_t offset_in_row = 0;       // Byte offset within current row
  uint32_t row_bytes = 0;           // Total bytes in a row (dim0 extent)

  // Cached OOB state (avoids per-sector dimension traversal)
  bool row_is_oob = false;      // Higher-dim OOB: entire row is fill
  uint32_t valid_row_bytes = 0; // Dim0 valid bytes in row (constant per tile)

  // Linear mode state (simple 1D address range)
  uint64_t linear_addr = 0;      // Current address
  uint32_t linear_remaining = 0; // Remaining bytes
};

class tma_agu_unit_t {
public:
  void init_tensor(tma_agu_state_t &state, const tensormap_descriptor_t &tm,
                   const uint32_t start_coords[5]) {

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

    // Calculate strides: stride[0] = elem_size, stride[i] = globalStrides[i-1]
    state.global_strides[0] = state.elem_size;
    for (uint32_t d = 1; d < state.num_dims; d++) {
      state.global_strides[d] = tm.fields.globalStrides[d - 1];
    }

    // Initialize tile coordinates to (0,0,...,0)
    for (int i = 0; i < 5; i++) {
      state.tile_coords[i] = 0;
    }

    // Calculate initial row address: base + sum(start_coords[d] * stride[d])
    state.curr_row_addr = tm.fields.globalAddress;
    for (uint32_t d = 0; d < state.num_dims; d++) {
      state.curr_row_addr += start_coords[d] * state.global_strides[d];
    }

    state.row_bytes = state.box_dim[0] * state.elem_size;
    state.offset_in_row = 0;
    state.done = false;

    // Precompute dim0 valid bytes (constant for entire tile)
    uint32_t dim0_remaining = (start_coords[0] < state.global_dim[0])
                                  ? state.global_dim[0] - start_coords[0]
                                  : 0;
    state.valid_row_bytes =
        std::min(state.box_dim[0], dim0_remaining) * state.elem_size;

    // Precompute higher-dim OOB for first row (tile_coords all zero)
    state.row_is_oob = false;
    for (uint32_t d = 1; d < state.num_dims; d++) {
      if (start_coords[d] >= state.global_dim[d]) {
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
   *   - Breaks rows into 128B-aligned chunks
   *
   * For linear mode:
   *   - Simple sequential address generation
   *   - Aligns to 128B boundaries
   */
  bool gen_next_req(tma_agu_state_t &state, uint64_t &out_addr,
                    uint32_t &out_size) {
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
      // Size = min(SECTOR_SIZE, remaining, bytes to next sector boundary)
      // Issue at sector granularity (32B) to match L2 sector size and ensure
      // 1:1 correspondence between issued requests and received responses.
      uint32_t to_boundary = SECTOR_SIZE - (state.linear_addr % SECTOR_SIZE);
      out_size = std::min(
          {(uint32_t)SECTOR_SIZE, state.linear_remaining, to_boundary});

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
            (state.offset_in_row >= state.valid_row_bytes); // Dim0 boundary
      }

      // Calculate address: current_row_base + offset_within_row
      out_addr = state.curr_row_addr + state.offset_in_row;

      // Calculate size: min(SECTOR_SIZE, remaining in row, bytes to next sector
      // boundary). Issue at sector granularity (32B) to match L2 sector cache.
      uint32_t row_remaining = state.row_bytes - state.offset_in_row;
      uint32_t to_sector_boundary = SECTOR_SIZE - (out_addr % SECTOR_SIZE);
      out_size =
          std::min({(uint32_t)SECTOR_SIZE, row_remaining, to_sector_boundary});

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
      if (state.start_coords[d] + state.tile_coords[d] >= state.global_dim[d]) {
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

  struct tma_transaction_t {
    ptx_thread_info *m_thread = nullptr;
    ptx_instruction *m_inst = nullptr;
    inst_t::tma_static_info_t m_static_info;
    inst_t::tma_dyn_info_t m_dyn_info;
    uint32_t m_bytes_completed = 0;
    tma_agu_state_t agu_state; // AGU state for this transaction

    // Debug counters
    uint32_t m_mf_issued_count = 0;   // Number of mem_fetch requests issued
    uint32_t m_mf_received_count = 0; // Number of mem_fetch responses received
    uint32_t m_mf_tx_inflight = 0; // Currently in-flight for this transaction

    void reset() {
      m_thread = nullptr;
      m_inst = nullptr;
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

  // Helper function to finalize a completed transaction
  void finalize_transaction(unsigned tx_uid) {
    auto it = m_transactions.find(tx_uid);
    if (it == m_transactions.end())
      return;

    auto &tx = it->second;
    auto thread = tx.m_thread;
    unsigned cta_id = thread->get_hw_ctaid();
    unsigned warp_id = thread->get_hw_wid();

    bool is_write =
        (tx.m_static_info.dst_space == inst_t::tma_static_info_t::TMA_GLOBAL);
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
    auto num_transactions_before = issue_queue.size();
    for (unsigned laneid = 0; laneid < warp_size; laneid++) {
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
          assert((tensormap_addr < LOCAL_GENERIC_START ||
                  tensormap_addr >= GLOBAL_HEAP_START) &&
                 "TMA tensor: tensormap address must be in global memory, "
                 "not shared/local memory");
          global_mem->read(tensormap_addr, TENSORMAP_DESCRIPTOR_SIZE,
                           &tensormap);
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

        bool idealized =
            m_shader_ctx->get_config()->gpgpu_tma_idealized_memory != 0;

        // For TMA write operations, add transaction to bulk group
        bool is_write_op = (tma_static_info.dst_space ==
                            inst_t::tma_static_info_t::TMA_GLOBAL);
        if (is_write_op) {
          unsigned cta_id = thread->get_hw_ctaid();
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
    if (issue_queue.size() - num_transactions_before > 1) {
      printf("Error: Multiple active threads for TMA inst not supported\n");
      abort();
    }
  }

  void cycle() {

    // Process delayed arrive_tx notifications
    if (!m_pending_arrives.empty()) {
      for (auto &entry : m_pending_arrives) {
        entry.remaining--;
      }
      for (int i = m_pending_arrives.size() - 1; i >= 0; i--) {
        if (m_pending_arrives[i].remaining == 0) {
          auto &entry = m_pending_arrives[i];
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

    // Process pending TMA responses
    if (!m_response_fifo.empty()) {
      mem_fetch *mf = m_response_fifo.front();
      m_response_fifo.pop_front();

      mem_fetch *parent_mf = mf->get_original_mf() ? mf->get_original_mf() : mf;
      auto mf_it = m_mf_to_tx.find(parent_mf->get_request_uid());

      assert(mf_it != m_mf_to_tx.end());
      unsigned tx_uid = mf_it->second;
      auto tx_it = m_transactions.find(tx_uid);

      assert(tx_it != m_transactions.end());
      auto &tx = tx_it->second;
      tx.m_mf_received_count++;
      assert(m_mf_inflight > 0);
      m_mf_inflight--;
      assert(tx.m_mf_tx_inflight > 0);
      tx.m_mf_tx_inflight--;

      bool is_write =
          (tx.m_static_info.dst_space == inst_t::tma_static_info_t::TMA_GLOBAL);

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
        if (tx.m_static_info.dst_space ==
            inst_t::tma_static_info_t::TMA_SHARED_CLUSTER) {
          assert(false && "Unsupported TMA destination space: CLUSTER");
        } else if (tx.m_static_info.dst_space !=
                   inst_t::tma_static_info_t::TMA_SHARED_CTA) {
          assert(false && "Unrecognized TMA destination space");
        }
      }

      // Count bytes: use min(mf_size, parent_size) to handle L2 sector
      // subdivision
      unsigned mf_size = mf->get_data_size();
      unsigned parent_size = parent_mf->get_data_size();
      unsigned bytes_to_add = (mf_size > parent_size) ? parent_size : mf_size;
      tx.m_bytes_completed += bytes_to_add;

      // Check if transaction is complete
      if (tx.m_bytes_completed >= tx.m_dyn_info.size_in_bytes)
        finalize_transaction(tx_uid);

      delete mf;
    }

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

      unsigned tx_uid = issue_queue.front();

      auto it = m_transactions.find(tx_uid);
      assert(it != m_transactions.end());
      auto &tx = it->second;

      // Per-transaction quota: limit how many inflight requests a single
      // transaction can have, so multiple CTAs share bandwidth fairly.
      unsigned tx_quota = m_shader_ctx->get_config()->gpgpu_tma_tx_quota;
      if (tx_quota > 0 && tx.m_mf_tx_inflight >= tx_quota) {
        // This transaction hit its quota — try the next one in the queue.
        // Rotate: move current to back so other transactions get a chance.
        issue_queue.pop_front();
        issue_queue.push_back(tx_uid);
        return;
      }

      bool is_write =
          (tx.m_static_info.dst_space == inst_t::tma_static_info_t::TMA_GLOBAL);
      unsigned packet_size = is_write ? WRITE_PACKET_SIZE : READ_PACKET_SIZE;

      if (!m_icnt->full(packet_size, is_write)) {
        uint64_t addr;
        uint32_t size;
        bool oob_l2 = m_shader_ctx->get_config()->gpgpu_tma_oob_l2_traffic;

        // Loop: batch-consume consecutive skip-OOB requests in one cycle,
        // then issue at most one mem_fetch to L2 before breaking.
        while (m_agu.gen_next_req(tx.agu_state, addr, size)) {

          if (tx.m_mf_issued_count == 0) {
            GPPRINTF_TMA(
                TMA,
                "[TMA AGU] tx_uid=%u starting to issue %s mem_fetch requests\n",
                tx_uid, is_write ? "WRITE" : "READ");
            fflush(stdout);
          }
          tx.m_mf_issued_count++;

          // Determine if this OOB request should skip L2:
          //   - Writes: always skip OOB (must not write beyond tensor bounds)
          //   - Reads:  skip OOB only when gpgpu_tma_oob_l2_traffic is off
          bool skip_l2 = tx.agu_state.is_fill_request && (is_write || !oob_l2);

          if (skip_l2) {
            // No interconnect needed — just count bytes
            tx.m_bytes_completed += size;
            if (tx.m_bytes_completed >= tx.m_dyn_info.size_in_bytes) {
              finalize_transaction(tx_uid);
              break;
            }
            continue; // Consume next request without waiting for icnt
          }

          // Issue mem_fetch to interconnect → L2.
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

          m_icnt->push(mf);
          m_mf_inflight++;
          tx.m_mf_tx_inflight++;
          break; // One mem_fetch per cycle
        }
      }

      // Remove from issue queue when all requests have been issued
      if (tx.agu_state.done) {
        GPPRINTF_TMA(TMA,
                     "[TMA AGU DONE] tx_uid=%u issued %u %s mem_fetch requests "
                     "(total bytes: %u)\n",
                     tx_uid, tx.m_mf_issued_count, is_write ? "WRITE" : "READ",
                     tx.m_dyn_info.size_in_bytes);
        fflush(stdout);
        issue_queue.pop_front();
      }
    }
  }

  void fill(mem_fetch *mf) {
    mf->set_status(IN_TMA_RESPONSE_FIFO,
                   m_shader_ctx->get_gpu()->gpu_sim_cycle +
                       m_shader_ctx->get_gpu()->gpu_tot_sim_cycle);
    m_response_fifo.push_back(mf);
  }

  bool response_buffer_full() const {
    // ! assume infinite buffer for simplicity
    return false;
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

void tma_unit_t::cycle() { m_impl->cycle(); }

void tma_unit_t::fill(mem_fetch *mf) { m_impl->fill(mf); }

bool tma_unit_t::response_buffer_full() const {
  return m_impl->response_buffer_full();
}

} // namespace flash_gpgpu_sim

//=============================================================================
// Functional Simulation: Helper Functions
//=============================================================================

// Convert global address offset to tile-local offset
// Global tensor may have different row stride than tile (based on box
// dimensions) This function converts (global_addr - base_global_addr) to
// equivalent tile offset
static uint64_t global_to_tile_offset(uint64_t global_addr, uint64_t base_addr,
                                      const tensormap_descriptor_t &tensormap) {
  uint32_t elem_size = tensormap.get_element_size();
  uint32_t num_dims = tensormap.num_dims();

  // Calculate the byte offset in global memory space
  assert(global_addr >= base_addr && "global_addr must be >= base_addr");
  uint64_t global_byte_offset = global_addr - base_addr;

  if (num_dims == 1) {
    // 1D: tile offset equals global offset
    return global_byte_offset;
  }

  // For 2D and higher, decompose global offset into coordinates
  // and recompute using tile strides

  // Get global strides (dim 0 uses elem_size, higher dims use globalStrides)
  uint64_t global_strides[5];
  global_strides[0] = elem_size;
  for (uint32_t d = 1; d < num_dims; d++) {
    global_strides[d] = tensormap.fields.globalStrides[d - 1];
  }

  // Get tile strides (based on box dimensions)
  uint64_t tile_strides[5];
  tile_strides[0] = elem_size;
  for (uint32_t d = 1; d < num_dims; d++) {
    tile_strides[d] = tile_strides[d - 1] * tensormap.fields.boxDim[d - 1];
  }

  // Decompose global_byte_offset into coordinates (highest to lowest dimension)
  // Then recompute offset using tile strides
  uint64_t tile_offset = 0;
  uint64_t remaining = global_byte_offset;

  for (int d = num_dims - 1; d >= 0; d--) {
    uint64_t coord_in_tile = remaining / global_strides[d];
    remaining = remaining % global_strides[d];
    tile_offset += coord_in_tile * tile_strides[d];
  }

  return tile_offset;
}

// Apply TMA swizzle transformation to shared memory address
//
// Mask-based implementation matching Hopper/Blackwell hardware behavior.
// Assumes 128B row stride for row extraction (addr >> 7).
// Applies XOR permutation at 16B granularity (shift = 4).
static uint64_t apply_tma_swizzle(uint64_t linear_offset,
                                  uint32_t smem_base_addr,
                                  uint32_t swizzle_mode, uint32_t row_bytes) {
  if (swizzle_mode == TMA_SWIZZLE_NONE)
    return linear_offset;

  uint32_t mask = 0;
  constexpr uint32_t shift = 4; // only support 16B granularity for now

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
    printf("ERROR: TMA 96B swizzle mode is not yet implemented\n");
    abort();
  default:
    printf("ERROR: Unknown TMA swizzle mode %u\n", swizzle_mode);
    abort();
  }

  // Extract row information assuming 128B stride
  // In Hopper/Blackwell hardware, row extraction is based on 128B alignment
  uint32_t row_bits = (uint32_t)(linear_offset >> 7);

  // Apply XOR permutation: addr XOR ((row_bits & mask) << shift)
  return linear_offset ^ ((uint64_t)(row_bits & mask) << shift);
}

// Generate 128B-aligned memory fetch requests
// Splits a contiguous memory range into cache-line-aligned requests
static void
gen_aligned_req(uint64_t start_addr, uint32_t total_bytes,
                std::vector<std::pair<uint64_t, uint32_t>> &requests) {
  if (total_bytes == 0)
    return;

  constexpr uint32_t CACHE_LINE_SIZE = 128;
  uint64_t current_addr = start_addr;
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

    requests.push_back({current_addr, request_size});

    current_addr += request_size;
    remaining_bytes -= request_size;
  }
}

// Recursive helper to traverse tensor dimensions and generate memory requests
// dim: current dimension being processed (Rank-1 down to 0)
// current_coords: accumulated coordinates from higher dimensions
// base_addr: current physical address
// tensormap: tensor descriptor
// requests: output vector of (address, size) pairs
static void
traverse_tensor_dim(int dim, const uint32_t current_coords[5],
                    uint64_t base_addr, const tensormap_descriptor_t &tensormap,
                    std::vector<std::pair<uint64_t, uint32_t>> &requests) {
  uint32_t elem_size = tensormap.get_element_size();

  if (dim == 0) {
    // Base case: innermost dimension (contiguous)
    uint32_t start_x = current_coords[0];
    uint32_t box_width = tensormap.fields.boxDim[0];
    uint32_t global_width = tensormap.fields.globalDim[0];

    // Calculate valid range intersection: [start_x, start_x + box_width) ∩ [0,
    // global_width)
    uint32_t valid_start = start_x;
    uint32_t valid_end = start_x + box_width;

    // Check if completely out of bounds
    if (valid_start >= global_width || valid_end <= 0) {
      return; // No valid data, skip
    }

    // Clamp to valid tensor boundaries
    valid_start = std::max(valid_start, 0u);
    valid_end = std::min(valid_end, global_width);

    if (valid_start >= valid_end)
      return;

    // Calculate physical start address and total bytes
    uint64_t phys_start_addr = base_addr + (valid_start * elem_size);
    uint32_t valid_bytes = (valid_end - valid_start) * elem_size;

    // Generate aligned memory fetch requests
    gen_aligned_req(phys_start_addr, valid_bytes, requests);

  } else {
    // Recursive case: traverse higher dimensions
    uint32_t box_extent = tensormap.fields.boxDim[dim];
    uint32_t global_extent = tensormap.fields.globalDim[dim];
    uint64_t stride = tensormap.fields.globalStrides[dim - 1];

    for (uint32_t i = 0; i < box_extent; i++) {
      uint32_t current_coord = current_coords[dim] + i;

      // OOB check: skip if outside valid tensor range
      if (current_coord >= global_extent) {
        continue; // Skip this branch (zero-padding)
      }

      // Calculate address offset for this coordinate
      uint64_t next_addr = base_addr + (current_coord * stride);

      // Recurse to next lower dimension
      traverse_tensor_dim(dim - 1, current_coords, next_addr, tensormap,
                          requests);
    }
  }
}

// Generate TMA memory fetch requests for a tensor tile
// start_coords: starting coordinate for each dimension (e.g., {x, y, z, w, v})
// Returns: vector of (physical_address, size_in_bytes) pairs
static std::vector<std::pair<uint64_t, uint32_t>>
generate_tma_requests(const tensormap_descriptor_t &tensormap,
                      const uint32_t start_coords[5]) {
  std::vector<std::pair<uint64_t, uint32_t>> requests;

  if (!tensormap.is_valid() || tensormap.fields.tensorRank > 4) {
    return requests; // Empty result for invalid tensormap
  }

  // Start recursive traversal from highest dimension (0-based index)
  int highest_dim = static_cast<int>(tensormap.num_dims()) -
                    1; // highest dimension index (0-based)
  uint64_t base_addr = tensormap.fields.globalAddress;

  traverse_tensor_dim(highest_dim, start_coords, base_addr, tensormap,
                      requests);

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

// Execute TMA data transfer (load or store)
// is_load=true: global -> shared, is_load=false: shared -> global
static void do_tma_transfer(const tensormap_descriptor_t &tensormap,
                            const uint32_t coords[5], memory_space *shared_mem,
                            memory_space *global_mem, uint32_t smem_addr,
                            ptx_thread_info *thread, const ptx_instruction *pI,
                            bool is_load) {
  // For load operations, pre-fill the entire tile in shared memory with OOB
  // fill value
  if (is_load) {
    uint32_t tile_size_bytes = tensormap.get_tile_size_bytes();
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
  uint64_t base_global_addr = tensormap.calculate_src_addr(coords);

  uint32_t swizzle_mode = tensormap.fields.swizzle;
  uint32_t elem_size = tensormap.get_element_size();
  uint32_t row_bytes = tensormap.fields.boxDim[0] * elem_size;

  constexpr uint32_t SWIZZLE_GRANULARITY = 16;
  constexpr uint32_t LOCAL_BUF_SIZE = 128;
  alignas(16) unsigned char local_data_buf[LOCAL_BUF_SIZE];

  for (const auto &req : memory_requests) {
    uint64_t global_req_addr = req.first;
    uint32_t req_size = req.second;

    uint64_t tile_offset =
        global_to_tile_offset(global_req_addr, base_global_addr, tensormap);

    unsigned char *data_buffer = (req_size <= LOCAL_BUF_SIZE)
                                     ? local_data_buf
                                     : new unsigned char[req_size];

    if (is_load) {
      // Load: global -> shared
      global_mem->read(global_req_addr, req_size, data_buffer);

      GPPRINTF_INST_EXEC(
          TMA,
          "coord[%u,%u,%u,%u,%u] "
          "swizzle_mode %u "
          "gmem=0x%llx -> "
          "smem=0x%x, space=%p, size=%u, tile_offset=0x%llx, "
          "data[0..3]=0x%02x%02x%02x%02x fp32 %.3f\n",
          coords[0], coords[1], coords[2], coords[3], coords[4], swizzle_mode,
          (unsigned long long)global_req_addr, (unsigned)smem_addr, shared_mem,
          req_size, (unsigned long long)tile_offset,
          req_size > 0 ? data_buffer[0] : 0, req_size > 1 ? data_buffer[1] : 0,
          req_size > 2 ? data_buffer[2] : 0, req_size > 3 ? data_buffer[3] : 0,
          *reinterpret_cast<float *>(data_buffer));

      if (swizzle_mode != TMA_SWIZZLE_NONE) {
        for (uint32_t sub_offset = 0; sub_offset < req_size;
             sub_offset += SWIZZLE_GRANULARITY) {
          uint32_t sub_size =
              std::min(SWIZZLE_GRANULARITY, req_size - sub_offset);
          uint64_t logical_offset = tile_offset + sub_offset;
          uint64_t swizzled_offset = apply_tma_swizzle(
              logical_offset, smem_addr, swizzle_mode, row_bytes);

          shared_mem->write(smem_addr + swizzled_offset, sub_size,
                            data_buffer + sub_offset, thread, pI);
        }
      } else {
        // No swizzle - write contiguously
        shared_mem->write(smem_addr + tile_offset, req_size, data_buffer,
                          thread, pI);
      }

    } else {
      // Store: shared -> global (reverse swizzle)
      if (swizzle_mode != TMA_SWIZZLE_NONE) {
        for (uint32_t sub_offset = 0; sub_offset < req_size;
             sub_offset += SWIZZLE_GRANULARITY) {
          uint32_t sub_size =
              std::min(SWIZZLE_GRANULARITY, req_size - sub_offset);
          uint64_t logical_offset = tile_offset + sub_offset;
          uint64_t swizzled_offset = apply_tma_swizzle(
              logical_offset, smem_addr, swizzle_mode, row_bytes);

          shared_mem->read(smem_addr + swizzled_offset, sub_size,
                           data_buffer + sub_offset);
        }
      } else {
        // No swizzle - read contiguously
        shared_mem->read(smem_addr + tile_offset, req_size, data_buffer);
      }

      global_mem->write(global_req_addr, req_size, data_buffer, thread, pI);
    }

    if (req_size > LOCAL_BUF_SIZE)
      delete[] data_buffer;
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

// Read tensormap from global memory, validate, parse coordinates, compute size
static void
setup_tensor_tma(memory_space *global_mem, uint64_t tensormap_addr,
                 unsigned inst_dim, const operand_info &coord_operand,
                 ptx_thread_info *thread, tensormap_descriptor_t &out_tensormap,
                 uint32_t out_coords[5], uint32_t &out_size_in_bytes) {
  using namespace flash_gpgpu_sim;
  global_mem->read(tensormap_addr, TENSORMAP_DESCRIPTOR_SIZE, &out_tensormap);
  validate_tensormap(tensormap_addr, out_tensormap, inst_dim);

  out_size_in_bytes = out_tensormap.get_tile_size_bytes();
  assert(out_size_in_bytes > 0 && "TMA tensor transfer size cannot be zero");

  parse_tensor_coords(thread, coord_operand, out_coords);
}

// Handle TMA linear copy instruction (non-tensor)
static void handle_tma_copy(ptx_instruction *pI, ptx_thread_info *thread) {
  using namespace flash_gpgpu_sim;

  const auto &options = pI->get_options();
  if (options.size() != 3) {
    for (auto op : options) {
      GPPRINTF_INST_EXEC(TMA, "option: %d\n", op);
    }
  }
  assert(options.size() >= 3 &&
         "TMA copy must have at least dst, src, completion_mechanism.");

  auto option_iter = options.begin();
  auto dst_option = *option_iter++;
  auto src_option = *option_iter++;
  auto completion_option = *option_iter++;

  memory_space *global_mem = thread->get_global_memory();
  memory_space *shared_mem = thread->m_shared_mem;

  if (dst_option == CTA_OPTION && src_option == GLOBAL_OPTION &&
      completion_option == TMA_MBAR_COMPLETE_BYTES) {
    // shared::cta <- global with MBAR completion
    auto dst_addr = get_operand_u32(thread, pI->dst());
    auto src_addr = get_operand_u64(thread, pI->src1());
    auto size_in_bytes = get_operand_u32(thread, pI->src2());
    auto mbar_addr = get_operand_u32(thread, pI->src3());

    check_tma_alignment(dst_addr, src_addr, size_in_bytes);

    inst_t::tma_static_info_t tma_static_info{
        .tma_type = inst_t::tma_static_info_t::TMA_NORMAL,
        .dst_space = inst_t::tma_static_info_t::TMA_SHARED_CTA,
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
                       "TMA shared::cta <- global dst=0x%x, src=0x%llx, "
                       "size_in_bytes=%u, mbar=0x%x\n",
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

    copy_mem(shared_mem, src_addr, global_mem, dst_addr, size_in_bytes, thread,
             pI);

    GPPRINTF_INST_EXEC(TMA,
                       "Functional Sim: "
                       "TMA global <- shared::cta dst=0x%llx, src=0x%x, "
                       "size_in_bytes=%u\n",
                       (unsigned long long)dst_addr, src_addr, size_in_bytes);
  } else {
    assert(false && "Unsupported TMA copy instruction");
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

  inst_t::tma_static_info_t tma_static_info{
      .tma_type = inst_t::tma_static_info_t::TMA_BULK_WAIT,
      .dst_space = inst_t::tma_static_info_t::TMA_SPACE_INVALID,
      .src_space = inst_t::tma_static_info_t::TMA_SPACE_INVALID,
      .bulk_wait_num = group_num,
  };
  pI->set_tma_static_info(tma_static_info);
  GPPRINTF_INST_EXEC(TMA, "Functional Sim: cp.async.bulk.wait_group %u\n",
                     group_num);
}

// Handle cp.async.bulk.tensor.Nd (tensor load/store)
static void handle_tma_tensor(ptx_instruction *pI, ptx_thread_info *thread) {
  using namespace flash_gpgpu_sim;

  const auto &options = pI->get_options();
  if (options.size() != 5) {
    for (auto op : options) {
      GPPRINTF_INST_EXEC(TMA, "  option: %d\n", op);
    }
  }
  assert(
      options.size() >= 5 &&
      "TMA tensor copy must have: TENSOR_OPTION, dim, dst, src, completion.");

  auto option_iter = options.begin();
  ++option_iter; // Skip TENSOR_OPTION
  auto dim_option = *option_iter++;
  auto dst_option = *option_iter++;
  auto src_option = *option_iter++;
  auto completion_option = *option_iter++;

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
    uint32_t coords[5];
    uint32_t size_in_bytes;
    setup_tensor_tma(global_mem, tensormap_addr, inst_dim,
                     pI->get_operands()[2], thread, tensormap, coords,
                     size_in_bytes);

    inst_t::tma_static_info_t tma_static_info{
        .tma_type = inst_t::tma_static_info_t::TMA_TENSOR,
        .dst_space = inst_t::tma_static_info_t::TMA_SHARED_CTA,
        .src_space = inst_t::tma_static_info_t::TMA_GLOBAL,
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
    uint32_t coords[5];
    uint32_t size_in_bytes;
    setup_tensor_tma(global_mem, tensormap_addr, inst_dim,
                     pI->get_operands()[1], thread, tensormap, coords,
                     size_in_bytes);

    GPPRINTF_INST_EXEC(
        TMA, "TMA tensor store Extracted coordinates: [%u, %u, %u, %u, %u]\n",
        coords[0], coords[1], coords[2], coords[3], coords[4]);

    inst_t::tma_static_info_t tma_static_info{
        .tma_type = inst_t::tma_static_info_t::TMA_TENSOR,
        .dst_space = inst_t::tma_static_info_t::TMA_GLOBAL,
        .src_space = inst_t::tma_static_info_t::TMA_SHARED_CTA,
    };
    pI->set_tma_static_info(tma_static_info);

    inst_t::tma_dyn_info_t tma_dyn_info{
        .dst_addr = tensormap_addr,
        .src_addr = src_addr,
        .size_in_bytes = size_in_bytes,
    };
    for (unsigned i = 0; i < 5; ++i)
      tma_dyn_info.coords[i] = coords[i];
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
    GPPRINTF_INST_EXEC(
        TMA, "[STUB] Unsupported cp.async.bulk.tensor variant%s\n", "");
  }
}

//=============================================================================
// Functional Simulation: Entry Point
//=============================================================================

void handle_tma_inst(const ptx_instruction *pIin, ptx_thread_info *thread) {
  using namespace flash_gpgpu_sim;

  ptx_instruction *pI = const_cast<ptx_instruction *>(pIin);

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
