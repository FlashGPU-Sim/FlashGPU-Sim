#include "tma.h"
#include <atomic>

#include "../cuda-sim/ptx_ir.h"
#include "../gpu-sim.h"
#include "../shader.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../trace.h"
#include "ptx.tab.h"

#include <vector>

std::atomic<unsigned int> tma_next_tx_uid = 0;

namespace flash_gpgpu_sim {

class tma_unit_impl_t {
public:
  tma_unit_impl_t(shader_core_ctx *shader_ctx, barrier_set_t *barriers, 
                  mem_fetch_interface *icnt, shader_core_mem_fetch_allocator *mf_allocator)
      : m_shader_ctx(shader_ctx), m_barriers(barriers), m_icnt(icnt), m_mf_allocator(mf_allocator) {
      }

private:
  shader_core_ctx *m_shader_ctx;
  barrier_set_t *m_barriers;
  mem_fetch_interface *m_icnt;
  shader_core_mem_fetch_allocator *m_mf_allocator;

  struct tma_transaction_t {
    ptx_thread_info *m_thread = nullptr;
    ptx_instruction *m_inst = nullptr;
    inst_t::tma_static_info_t m_static_info;
    inst_t::tma_dyn_info_t m_dyn_info;
    uint32_t m_bytes_completed = 0;
    void reset() {
      m_thread = nullptr;
      m_inst = nullptr;
      m_bytes_completed = 0;
    }
    bool is_valid() const { return m_thread != nullptr; }
  };
  std::unordered_map<unsigned, tma_transaction_t>  m_transactions;

  std::list<unsigned> issue_queue;
  std::unordered_map<unsigned, unsigned> m_mf_to_tx;
  
  std::list<mem_fetch *> m_response_fifo;

public:
  void warp_reaches_tma(unsigned cta_id, unsigned warp_id, warp_inst_t *inst) {
    ptx_instruction *pI = dynamic_cast<ptx_instruction *>(inst);
    if (pI == nullptr) {
      printf("Error: TMA inst is not a ptx_inst\n");
      abort();
    }

    // Check if this is a tensor instruction
    bool is_tensor = false;
    for (auto op : pI->get_options()) {
      if (op == TENSOR_OPTION) {
        is_tensor = true;
        break;
      }
    }

    const auto &tma_static_info = pI->get_tma_static_info();

    // For tensor load operations (shared <- global with mbar), complete immediately
    if (is_tensor && 
        tma_static_info.dst_space == inst_t::tma_static_info_t::TMA_SHARED_CTA &&
        tma_static_info.src_space == inst_t::tma_static_info_t::TMA_GLOBAL) {
      
      const auto warp_size = m_shader_ctx->get_warp_size();
      for (int laneid = 0; laneid < warp_size; laneid++) {
        const auto &tma_dyn_info = pI->get_tma_dyn_info(laneid);
        if (tma_dyn_info.is_valid()) {
          DPRINTF_TMA(TMA,
                      "Tensor load immediate complete: "
                      "cta=%u, warp=%u, mbar=0x%x, size=%u\n",
                      cta_id, warp_id, tma_dyn_info.mbar_addr,
                      tma_dyn_info.size_in_bytes);
          
          // Immediately complete the transaction
          m_barriers->complete_tx(cta_id, warp_id, tma_dyn_info.mbar_addr,
                                  tma_dyn_info.size_in_bytes);
        }
      }
      return;
    }
    
    // For tensor store operations, nothing to do (no mbar completion)
    if (is_tensor) {
      DPRINTF_TMA(TMA, "Tensor store: no mbar completion needed%s\n", "");
      return;
    }

    // Regular TMA copy instruction - queue for async processing
    const auto warp_size = m_shader_ctx->get_warp_size();
    auto num_transactions_before = issue_queue.size();
    for (int laneid = 0; laneid < warp_size; laneid++) {
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

        unsigned tx_uid = tma_next_tx_uid.fetch_add(1, std::memory_order_relaxed);
        m_transactions.emplace(tx_uid, tx);
        issue_queue.push_back(tx_uid);

        DPRINTF_INST_EXEC(TMA,
                          "Start transaction dst=0x%llx, src=0x%llx, "
                          "size_in_bytes=%u, mbar=0x%x, tx_uid=%u\n",
                          tx.m_dyn_info.dst_addr, tx.m_dyn_info.src_addr,
                          tx.m_dyn_info.size_in_bytes, tx.m_dyn_info.mbar_addr, tx_uid);

      }
    }
    if (issue_queue.size() - num_transactions_before > 1) {
      printf("Error: Multiple active threads for TMA inst not supported\n");
      abort();
    }
  }

  void cycle() {

    // check response fifo
    if (!m_response_fifo.empty()) {
      mem_fetch *mf = m_response_fifo.front();
      m_response_fifo.pop_front();
      assert(mf->get_access_type() == TMA_ACC_R);

      auto mf_it = m_mf_to_tx.find(mf->get_original_mf()->get_request_uid());
      assert(mf_it != m_mf_to_tx.end());
      auto tx_uid = mf_it->second;

      auto tx_it = m_transactions.find(tx_uid);
      assert(tx_it != m_transactions.end());
      auto &tx = tx_it->second;

      DPRINTF_TMA(TMA, "TMA response received for mf uid=%u, tx_uid=%u, data_size=%u, response fifo size=%lu\n",
                 mf->get_request_uid(),  tx_uid, mf->get_data_size(), m_response_fifo.size());

      if (tx.m_static_info.dst_space == inst_t::tma_static_info_t::TMA_SHARED_CTA) {
        // TMA would attempt to write data to shared memory once receive mf response
        // ! assume no shared memory bank conflict for simplicity
        tx.m_bytes_completed += mf->get_data_size();

        if ( tx.m_bytes_completed >= tx.m_dyn_info.size_in_bytes ) {
          auto thread = tx.m_thread;
          unsigned cta_id = thread->get_hw_ctaid();
          unsigned warp_id = thread->get_hw_wid();
          
          DPRINTF_INST_EXEC(TMA,
                            "Complete transaction dst=0x%llx, src=0x%llx, "
                            "size_in_bytes=%u, mbar=0x%x, tx_uid=%u \n",
                            tx.m_dyn_info.dst_addr, tx.m_dyn_info.src_addr,
                            tx.m_dyn_info.size_in_bytes, tx.m_dyn_info.mbar_addr, tx_uid);

          m_barriers->complete_tx(
              cta_id, warp_id, tx.m_dyn_info.mbar_addr,
              tx.m_dyn_info.size_in_bytes);
      
          m_transactions.erase(tx_it);
          m_mf_to_tx.erase(mf_it);
        } 

        delete mf;

      } else if (tx.m_static_info.dst_space == inst_t::tma_static_info_t::TMA_SHARED_CLUSTER) {
        assert(false && "Unsupported TMA destination space");
      } else {
        assert(false && "Unrecognized TMA destination space");
      }
    }

    // issue memory requests
    if (!issue_queue.empty()) {
      unsigned tx_uid = issue_queue.front();

      auto it = m_transactions.find(tx_uid);
      assert(it != m_transactions.end());
      auto &tx = it->second;

      if (!m_icnt->full(READ_PACKET_SIZE, false)) {
        // TODO: deal with corner case, would need sector mask if size not aligned w. cacheline size
        unsigned size = std::min(MAX_MEMORY_ACCESS_SIZE, tx.m_dyn_info.size_in_bytes - tx.m_bytes_completed);

        mem_access_t access(TMA_ACC_R, tx.m_dyn_info.src_addr + tx.m_bytes_completed, 
                            size, false, m_shader_ctx->get_gpu()->gpgpu_ctx);
        mem_fetch *mf =
            m_mf_allocator->alloc(access, -1,
                                  m_shader_ctx->get_gpu()->gpu_sim_cycle +
                                  m_shader_ctx->get_gpu()->gpu_tot_sim_cycle);
        
        m_mf_to_tx.emplace(mf->get_request_uid(), tx_uid);
        tx.m_bytes_completed += size;

        m_icnt->push(mf);
      }

      if ( tx.m_bytes_completed >= tx.m_dyn_info.size_in_bytes ) {
        issue_queue.pop_front();
        tx.m_bytes_completed = 0;
      }
    }
  }

  void fill(mem_fetch *mf) {
    mf->set_status(
      IN_TMA_RESPONSE_FIFO,
      m_shader_ctx->get_gpu()->gpu_sim_cycle + m_shader_ctx->get_gpu()->gpu_tot_sim_cycle);
    m_response_fifo.push_back(mf);
  }

  bool response_buffer_full() const {
    // ! assume infinite buffer for simplicity
    return false;
  }
};

tma_unit_t::tma_unit_t(shader_core_ctx *shader_ctx, barrier_set_t *barriers,
                       mem_fetch_interface *icnt, shader_core_mem_fetch_allocator *mf_allocator)
    : m_impl(std::make_unique<tma_unit_impl_t>(shader_ctx, barriers, icnt, mf_allocator)) {}

tma_unit_t::~tma_unit_t() = default;

void tma_unit_t::warp_reaches_tma(unsigned cta_id, unsigned warp_id,
                                  warp_inst_t *inst) {
  m_impl->warp_reaches_tma(cta_id, warp_id, inst);
}

void tma_unit_t::cycle() {
  m_impl->cycle();
}

void tma_unit_t::fill(mem_fetch *mf) {
  m_impl->fill(mf);
}

bool tma_unit_t::response_buffer_full() const {
  return m_impl->response_buffer_full();
}

} // namespace flash_gpgpu_sim

void handle_tma_inst(const ptx_instruction *pIin, ptx_thread_info *thread) {
  // Currently no TMA instructions are defined.

  ptx_instruction *pI = const_cast<ptx_instruction *>(pIin);

  DPRINTF_INST_EXEC(TMA, "inst %s\n", pI->to_string().c_str());

  bool is_commit_group = false;
  bool is_wait_group = false;
  bool is_tensor = false;
  for (auto op : pI->get_options()) {
    switch (op) {
      case COMMIT_GROUP_OPTION: is_commit_group = true; break;
      case WAIT_GROUP_OPTION:   is_wait_group = true;   break;
      case TENSOR_OPTION:       is_tensor = true;       break;
      default: break;
    }
  }

  auto get_u32_value = [&](const operand_info &op) {
    ptx_reg_t reg = thread->get_operand_value(op, op, U32_TYPE, thread, 0);
    return reg.u32;
  };
  auto get_u64_value = [&](const operand_info &op) {
    ptx_reg_t reg = thread->get_operand_value(op, op, U64_TYPE, thread, 0);
    return reg.u64;
  };

  if (!is_commit_group && !is_wait_group && !is_tensor) {
    // This is TMA copy instruction.
    const auto &options = pI->get_options();
    if (options.size() != 3) {
      for (auto op : options) {
        DPRINTF_INST_EXEC(TMA, "option: %d\n", op);
      }
    }
    assert(options.size() >= 3 &&
           "TMA copy must have at least dst, src, completion_mechanism.");
    // ! Ignore cache hint for now.
    auto option_iter = options.begin();
    auto dst_option = *option_iter;
    ++option_iter;
    auto src_option = *option_iter;
    ++option_iter;
    auto completion_option = *option_iter;
    ++option_iter;

    if (dst_option == CTA_OPTION && src_option == GLOBAL_OPTION &&
        completion_option == TMA_MBAR_COMPLETE_BYTES) {
      // Handle shared::cta <- global TMA copy inst with MBAR completion.

      // Extract operands.
      auto dst_addr = get_u32_value(pI->dst());
      auto src_addr = get_u64_value(pI->src1());
      auto size_in_bytes = get_u32_value(pI->src2());
      auto mbar_addr = get_u32_value(pI->src3());

      // Check alignment to 16 bytes.
      if (dst_addr % 16 != 0 || src_addr % 16 != 0 || size_in_bytes % 16 != 0) {
        DPRINTF_INST_EXEC(
            TMA, "unaligned TMA copy dst=0x%x, src=0x%llx, size_in_bytes=%u\n",
            dst_addr, src_addr, size_in_bytes);
        abort();
      }

      inst_t::tma_static_info_t tma_static_info{
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
      auto laneid = thread->get_laneid();
      pI->set_tma_dyn_info(laneid, tma_dyn_info);

      // write data into shared memory
      memory_space *global_mem = thread->get_global_memory();
      memory_space *shared_mem = thread->m_shared_mem;

      unsigned char *data_buffer = new unsigned char[size_in_bytes];
      global_mem->read(src_addr, size_in_bytes, data_buffer);
      shared_mem->write(dst_addr, size_in_bytes, data_buffer, thread, pI);

      delete[] data_buffer;

      DPRINTF_INST_EXEC(TMA,
        "Functional Sim: "
        "TMA shared::cta <- global dst=0x%x, src=0x%llx, "
        "size_in_bytes=%u, mbar=0x%x\n",
        dst_addr, src_addr, size_in_bytes, mbar_addr);

    } else {
      assert(false && "Unsupported TMA copy instruction");
    }

  } else if (is_tensor) {
    // Handle TMA tensor instruction (cp.async.bulk.tensor.Nd).
    // Format: cp.async.bulk.tensor.Nd.dst.src.completion [dst], [tensormap, {coords}], [mbar]
    // For shared::cluster.global.mbarrier::complete_tx::bytes variant
    
    const auto &options = pI->get_options();
    
    // Check for direction: shared <- global (load) or global <- shared (store)
    bool is_load = false;  // shared <- global
    bool is_store = false; // global <- shared
    bool has_mbar_completion = false;
    
    for (auto op : options) {
      if (op == CTA_OPTION || op == CLUSTER_OPTION) {
        // First space option indicates destination
        if (!is_load && !is_store) {
          is_load = true;  // shared is destination
        }
      } else if (op == GLOBAL_OPTION) {
        if (!is_load && !is_store) {
          is_store = true; // global is destination
        }
      } else if (op == TMA_MBAR_COMPLETE_BYTES) {
        has_mbar_completion = true;
      }
    }
    
    if (is_load && has_mbar_completion) {
      // cp.async.bulk.tensor.Nd.shared::cta.global.mbarrier::complete_tx::bytes
      // Operands: [dst_shared], [tensormap, {coords...}], [mbar]
      
      auto dst_addr = get_u32_value(pI->dst());
      
      // Get tensormap address (src1 is the tensormap pointer in shared memory)
      auto tensormap_addr = get_u32_value(pI->src1());
      
      // Get mbar address from last operand
      auto num_operands = pI->get_num_operands();
      const auto &mbar_op = pI->get_operands()[num_operands - 1];
      auto mbar_addr = get_u32_value(mbar_op);
      
      // Read tensormap descriptor from shared memory
      memory_space *shared_mem = thread->m_shared_mem;
      flash_gpgpu_sim::tensormap_descriptor_t tensormap = 
          flash_gpgpu_sim::tensormap_descriptor_t::read_from_shared(shared_mem, tensormap_addr);
      
      // Calculate transfer size from tensormap
      uint32_t size_in_bytes = tensormap.get_tile_size_bytes();
      if (size_in_bytes == 0) {
        // Fallback if tensormap not properly initialized
        size_in_bytes = 65536;  // Placeholder
        DPRINTF_INST_EXEC(TMA, "Warning: tensormap not initialized, using placeholder size%s\n", "");
      }
      
      // Get coordinates from operands (between tensormap and mbar)
      // For 2D: coords are {coord0, coord1}
      uint32_t coords[5] = {0, 0, 0, 0, 0};
      // TODO: Parse coordinates from operands based on tensor dimension
      // For now, assume coords start at operand index 2
      for (unsigned i = 2; i < num_operands - 1 && i - 2 < tensormap.fields.tensorRank; i++) {
        coords[i - 2] = get_u32_value(pI->get_operands()[i]);
      }
      
      // Calculate source address from tensormap and coordinates
      uint64_t src_addr = tensormap.calculate_src_addr(coords);
      
      inst_t::tma_static_info_t tma_static_info{
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
      auto laneid = thread->get_laneid();
      pI->set_tma_dyn_info(laneid, tma_dyn_info);
      
      // Perform actual data transfer (functional simulation)
      if (tensormap.is_valid()) {
        memory_space *global_mem = thread->get_global_memory();
        unsigned char *data_buffer = new unsigned char[size_in_bytes];
        global_mem->read(src_addr, size_in_bytes, data_buffer);
        shared_mem->write(dst_addr, size_in_bytes, data_buffer, thread, pI);
        delete[] data_buffer;
        
        DPRINTF_INST_EXEC(TMA,
          "Functional Sim: TMA tensor load dst=0x%x, src=0x%llx, "
          "size=%u, mbar=0x%x, tensormap=0x%x\n",
          dst_addr, (unsigned long long)src_addr, size_in_bytes, mbar_addr, tensormap_addr);
      } else {
        DPRINTF_INST_EXEC(TMA,
          "Functional Sim: TMA tensor load (tensormap invalid) "
          "dst=0x%x, mbar=0x%x, size=%u\n",
          dst_addr, mbar_addr, size_in_bytes);
      }
        
    } else if (is_store) {
      // cp.async.bulk.tensor.Nd.global.shared::cta.bulk_group
      // This is a store operation, no mbarrier involved
      
      // Get tensormap address
      auto tensormap_addr = get_u32_value(pI->dst());  // dst is tensormap for store
      auto src_addr = get_u32_value(pI->src1());        // src is shared memory
      
      // Read tensormap descriptor
      memory_space *shared_mem = thread->m_shared_mem;
      flash_gpgpu_sim::tensormap_descriptor_t tensormap = 
          flash_gpgpu_sim::tensormap_descriptor_t::read_from_shared(shared_mem, tensormap_addr);
      
      uint32_t size_in_bytes = tensormap.get_tile_size_bytes();
      if (size_in_bytes == 0) size_in_bytes = 65536;  // Fallback
      
      inst_t::tma_static_info_t tma_static_info{
          .dst_space = inst_t::tma_static_info_t::TMA_GLOBAL,
          .src_space = inst_t::tma_static_info_t::TMA_SHARED_CTA,
      };
      pI->set_tma_static_info(tma_static_info);
      
      // TODO: Perform actual store (shared -> global) when needed
      DPRINTF_INST_EXEC(TMA,
        "Functional Sim: TMA tensor store (global <- shared) "
        "tensormap=0x%x, src=0x%x, size=%u\n",
        tensormap_addr, src_addr, size_in_bytes);
        
    } else {
      DPRINTF_INST_EXEC(TMA, 
        "[STUB] Unsupported cp.async.bulk.tensor variant%s\n", "");
    }

  } else if (is_commit_group) {
    // Handle TMA commit group instruction (cp.async.bulk.commit_group).
    // TODO: Implement commit group - for now treat as NOP
    DPRINTF_INST_EXEC(TMA, "[STUB] cp.async.bulk.commit_group not implemented (treated as NOP)%s\n", "");

  } else if (is_wait_group) {
    // Handle TMA wait group instruction (cp.async.bulk.wait_group).
    // TODO: Implement wait group - for now treat as NOP (assume all transfers complete immediately)
    DPRINTF_INST_EXEC(TMA, "[STUB] cp.async.bulk.wait_group not implemented (treated as NOP)%s\n", "");

  } else {
    DPRINTF_INST_EXEC(TMA, "Unrecognized TMA instruction%s\n", "");
    pI->print_insn();
    assert(false && "Unrecognized TMA instruction");
  }
}

//=============================================================================
// TensorMap Descriptor Implementation
//=============================================================================

namespace flash_gpgpu_sim {

uint32_t tensormap_descriptor_t::get_element_size() const {
  // Element type encoding based on CUDA spec
  switch (fields.tensorDataType) {
    case TMA_DTYPE_U8:   return 1;
    case TMA_DTYPE_U16:  return 2;
    case TMA_DTYPE_U32:  return 4;
    case TMA_DTYPE_U64:  return 8;
    case TMA_DTYPE_F16:  return 2;
    case TMA_DTYPE_F32:  return 4;
    case TMA_DTYPE_F64:  return 8;
    case TMA_DTYPE_BF16: return 2;
    default:             return 4; // default to 4 bytes
  }
}

uint32_t tensormap_descriptor_t::get_tile_size_bytes() const {
  if (fields.tensorRank == 0) return 0;
  
  uint32_t total_elements = 1;
  for (uint32_t i = 0; i < fields.tensorRank; i++) {
    total_elements *= fields.boxDim[i];
  }
  return total_elements * get_element_size();
}

uint64_t tensormap_descriptor_t::calculate_src_addr(const uint32_t coords[5]) const {
  // Calculate the source address based on coordinates and strides
  // This is a simplified linear address calculation
  uint64_t addr = fields.globalAddress;
  uint32_t elem_size = get_element_size();
  
  for (uint32_t i = 0; i < fields.tensorRank; i++) {
    // For dimension 0, use element size; for others, use stride
    if (i == 0) {
      addr += coords[i] * elem_size;
    } else {
      addr += coords[i] * fields.globalStrides[i - 1];
    }
  }
  return addr;
}

void tensormap_descriptor_t::print() const {
  printf("TensorMap Descriptor:\n");
  printf("  global_address: 0x%llx\n", (unsigned long long)fields.globalAddress);
  printf("  rank: %u\n", fields.tensorRank);
  printf("  elemtype: %u (size=%u bytes)\n", fields.tensorDataType, get_element_size());
  printf("  box_dim: [");
  for (uint32_t i = 0; i < fields.tensorRank; i++) printf("%u%s", fields.boxDim[i], i < fields.tensorRank-1 ? ", " : "");
  printf("]\n");
  printf("  global_dim: [");
  for (uint32_t i = 0; i < fields.tensorRank; i++) printf("%u%s", fields.globalDim[i], i < fields.tensorRank-1 ? ", " : "");
  printf("]\n");
  printf("  tile_size_bytes: %u\n", get_tile_size_bytes());
}

tensormap_descriptor_t tensormap_descriptor_t::read_from_shared(memory_space *shared_mem, uint32_t addr) {
  tensormap_descriptor_t desc;
  // Read entire 128-byte descriptor as raw bytes
  shared_mem->read(addr, 128, desc.raw_bytes);
  return desc;
}

void tensormap_descriptor_t::write_to_shared(memory_space *shared_mem, uint32_t addr) const {
  // Write entire 128-byte descriptor as raw bytes
  shared_mem->write(addr, 128, raw_bytes, nullptr, nullptr);
}

} // namespace flash_gpgpu_sim

//=============================================================================
// handle_tensormap_inst - Process tensormap PTX instructions
//=============================================================================

void handle_tensormap_inst(const ptx_instruction *pI, ptx_thread_info *thread) {
  using namespace flash_gpgpu_sim;
  // DPRINTF_INST_EXEC(TMA, "tensormap inst: %s\n", pI->to_string().c_str());
  
  const auto &options = pI->get_options();
  
  // Helper to get operand values
  auto get_u32_value = [&](const operand_info &op) {
    ptx_reg_t reg = thread->get_operand_value(op, op, U32_TYPE, thread, 0);
    return reg.u32;
  };
  auto get_u64_value = [&](const operand_info &op) {
    ptx_reg_t reg = thread->get_operand_value(op, op, U64_TYPE, thread, 0);
    return reg.u64;
  };
  
  // Check for tensormap instruction types
  bool is_replace = false;
  bool is_cp_fenceproxy = false;
  bool is_tile = false;
  
  // Field type for replace operations
  bool is_global_address = false;
  bool is_rank = false;
  bool is_box_dim = false;
  bool is_global_dim = false;
  bool is_global_stride = false;
  bool is_element_stride = false;
  bool is_elemtype = false;
  bool is_interleave_layout = false;
  bool is_swizzle_mode = false;
  bool is_fill_mode = false;
  
  for (auto op : options) {
    switch (op) {
      case REPLACE_OPTION: is_replace = true; break;
      case CP_FENCEPROXY_OPTION: is_cp_fenceproxy = true; break;
      case TILE_OPTION: is_tile = true; break;
      case GLOBAL_ADDRESS_OPTION: is_global_address = true; break;
      case RANK_OPTION: is_rank = true; break;
      case BOX_DIM_OPTION: is_box_dim = true; break;
      case GLOBAL_DIM_OPTION: is_global_dim = true; break;
      case GLOBAL_STRIDE_OPTION: is_global_stride = true; break;
      case ELEMENT_STRIDE_OPTION: is_element_stride = true; break;
      case ELEMTYPE_OPTION: is_elemtype = true; break;
      case INTERLEAVE_LAYOUT_OPTION: is_interleave_layout = true; break;
      case SWIZZLE_MODE_OPTION: is_swizzle_mode = true; break;
      case FILL_MODE_OPTION: is_fill_mode = true; break;
      default: break;
    }
  }
  
  memory_space *shared_mem = thread->m_shared_mem;
  
  if (is_replace && is_tile) {
    // tensormap.replace.tile.<field> [dst], value
    // or tensormap.replace.tile.<field> [dst], dim_idx, value
    
    // Get destination address (tensormap address in shared memory)
    uint32_t tensormap_addr = get_u32_value(pI->dst());
    
    // Read existing descriptor, modify field, write back
    tensormap_descriptor_t desc = tensormap_descriptor_t::read_from_shared(shared_mem, tensormap_addr);
    
    if (is_global_address) {
      // tensormap.replace.tile.global_address [dst], value
      uint64_t value = get_u64_value(pI->src1());
      desc.fields.globalAddress = value;
      DPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.global_address [0x%x] = 0x%llx\n", 
                        tensormap_addr, (unsigned long long)value);
                        
    } else if (is_rank) {
      // tensormap.replace.tile.rank [dst], value
      uint32_t value = get_u32_value(pI->src1());
      desc.fields.tensorRank = value;
      DPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.rank [0x%x] = %u\n", tensormap_addr, value);
      
    } else if (is_box_dim) {
      // tensormap.replace.tile.box_dim [dst], dim_idx, value
      uint32_t dim_idx = get_u32_value(pI->src1());
      uint32_t value = get_u32_value(pI->src2());
      if (dim_idx < 5) desc.fields.boxDim[dim_idx] = value;
      DPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.box_dim [0x%x], dim=%u, value=%u\n", 
                        tensormap_addr, dim_idx, value);
      
    } else if (is_global_dim) {
      // tensormap.replace.tile.global_dim [dst], dim_idx, value
      uint32_t dim_idx = get_u32_value(pI->src1());
      uint32_t value = get_u32_value(pI->src2());
      if (dim_idx < 5) desc.fields.globalDim[dim_idx] = value;
      DPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.global_dim [0x%x], dim=%u, value=%u\n", 
                        tensormap_addr, dim_idx, value);
      
    } else if (is_global_stride) {
      // tensormap.replace.tile.global_stride [dst], dim_idx, value
      uint32_t dim_idx = get_u32_value(pI->src1());
      uint64_t value = get_u64_value(pI->src2());
      if (dim_idx < 5) desc.fields.globalStrides[dim_idx] = value;
      DPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.global_stride [0x%x], dim=%u, value=%llu\n", 
                        tensormap_addr, dim_idx, (unsigned long long)value);
      
    } else if (is_element_stride) {
      // tensormap.replace.tile.element_stride [dst], dim_idx, value
      uint32_t dim_idx = get_u32_value(pI->src1());
      uint32_t value = get_u32_value(pI->src2());
      if (dim_idx < 5) desc.fields.elementStrides[dim_idx] = value;
      DPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.element_stride [0x%x], dim=%u, value=%u\n", 
                        tensormap_addr, dim_idx, value);
      
    } else if (is_elemtype) {
      // tensormap.replace.tile.elemtype [dst], value
      uint32_t value = get_u32_value(pI->src1());
      desc.fields.tensorDataType = value;
      DPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.elemtype [0x%x] = %u\n", tensormap_addr, value);
      
    } else if (is_interleave_layout) {
      // tensormap.replace.tile.interleave_layout [dst], value
      uint32_t value = get_u32_value(pI->src1());
      desc.fields.interleave = value;
      DPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.interleave_layout [0x%x] = %u\n", tensormap_addr, value);
      
    } else if (is_swizzle_mode) {
      // tensormap.replace.tile.swizzle_mode [dst], value
      uint32_t value = get_u32_value(pI->src1());
      desc.fields.swizzle = value;
      DPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.swizzle_mode [0x%x] = %u\n", tensormap_addr, value);
      
    } else if (is_fill_mode) {
      // tensormap.replace.tile.fill_mode [dst], value
      uint32_t value = get_u32_value(pI->src1());
      desc.fields.oobFill = value;
      DPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.fill_mode [0x%x] = %u\n", tensormap_addr, value);
      
    } else {
      DPRINTF_INST_EXEC(TMA, "[STUB] Unrecognized tensormap.replace.tile field%s\n", "");
    }
    
    // Write modified descriptor back to shared memory
    desc.write_to_shared(shared_mem, tensormap_addr);
    
  } else if (is_cp_fenceproxy) {
    // tensormap.cp_fenceproxy.global.shared::cta... [dst_global], [src_shared], size
    // Copy tensormap from shared to global memory with fence
    // For now, just perform the copy without actual fence semantics
    
    // TODO: Implement the actual copy from shared to global
    // The operands are: [dst_global], [src_shared], size
    DPRINTF_INST_EXEC(TMA, "[STUB] tensormap.cp_fenceproxy - copy tensormap to global%s\n", "");
    
  } else {
    DPRINTF_INST_EXEC(TMA, "[STUB] Unrecognized tensormap instruction variant%s\n", "");
  }
}