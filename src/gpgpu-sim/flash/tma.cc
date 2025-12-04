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
      // cp.async.bulk.tensor.Nd.shared::cluster.global.mbarrier::complete_tx::bytes
      // [dst_shared], [tensormap, {coords...}], [mbar]
      
      auto dst_addr = get_u32_value(pI->dst());
      // Skip tensormap and coords for now, just get mbar address from last operand
      auto num_operands = pI->get_num_operands();
      const auto &mbar_op = pI->get_operands()[num_operands - 1];
      auto mbar_addr = get_u32_value(mbar_op);
      
      // For tensor operations, we don't know the exact size without parsing tensormap
      // Use a placeholder size (will be filled by timing simulation)
      // The actual data transfer is done by functional sim reading from tensormap
      uint32_t size_in_bytes = 65536;  // Placeholder - should match expect_tx
      
      inst_t::tma_static_info_t tma_static_info{
          .dst_space = inst_t::tma_static_info_t::TMA_SHARED_CTA,
          .src_space = inst_t::tma_static_info_t::TMA_GLOBAL,
      };
      pI->set_tma_static_info(tma_static_info);
      
      inst_t::tma_dyn_info_t tma_dyn_info{
          .dst_addr = dst_addr,
          .src_addr = 0,  // Not needed for tensor ops
          .size_in_bytes = size_in_bytes,
          .mbar_addr = mbar_addr,
      };
      auto laneid = thread->get_laneid();
      pI->set_tma_dyn_info(laneid, tma_dyn_info);
      
      DPRINTF_INST_EXEC(TMA,
        "Functional Sim: "
        "TMA tensor load (shared <- global) dst=0x%x, mbar=0x%x, size=%u\n",
        dst_addr, mbar_addr, size_in_bytes);
        
    } else if (is_store) {
      // cp.async.bulk.tensor.Nd.global.shared::cta.bulk_group
      // This is a store operation, no mbarrier involved
      // Just mark as valid but no completion tracking needed
      
      inst_t::tma_static_info_t tma_static_info{
          .dst_space = inst_t::tma_static_info_t::TMA_GLOBAL,
          .src_space = inst_t::tma_static_info_t::TMA_SHARED_CTA,
      };
      pI->set_tma_static_info(tma_static_info);
      
      DPRINTF_INST_EXEC(TMA,
        "Functional Sim: "
        "TMA tensor store (global <- shared) - no mbar completion%s\n", "");
        
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