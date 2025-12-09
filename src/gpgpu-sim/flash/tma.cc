#include "tma.h"
#include <atomic>
#include <cstdarg>

#include "../cuda-sim/ptx_ir.h"
#include "../gpu-sim.h"
#include "../shader.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../trace.h"
#include "ptx.tab.h"

#include <vector>

std::atomic<unsigned int> tma_next_tx_uid = 0;

// Forward declaration for helper function defined later
namespace flash_gpgpu_sim {
static uint64_t global_to_tile_offset(uint64_t global_addr, uint64_t base_addr,
                                       const tensormap_descriptor_t& tensormap);
}

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
  
  ptx_instruction *pI = const_cast<ptx_instruction *>(pIin);

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
    // Options include: TENSOR_OPTION, DIM_xD_OPTION, dst_space, src_space, completion
    
    const auto &options = pI->get_options();
    if (options.size() != 5) {
      for (auto op : options) {
        DPRINTF_INST_EXEC(TMA, "  option: %d\n", op);
      }
    }
    assert(options.size() >= 5 &&
           "TMA tensor copy must have: TENSOR_OPTION, dim, dst, src, completion.");
    
    auto option_iter = options.begin();
    ++option_iter; // Skip TENSOR_OPTION (first option is always .tensor)
    auto dim_option = *option_iter;
    ++option_iter;
    auto dst_option = *option_iter;
    ++option_iter;
    auto src_option = *option_iter;
    ++option_iter;
    auto completion_option = *option_iter;
    ++option_iter;

    memory_space *shared_mem = thread->m_shared_mem;
    memory_space *global_mem = thread->get_global_memory();

    auto compute_inst_dim = [&](unsigned dim_option) -> unsigned {
      unsigned d = 0;
      switch (dim_option) {
        case DIM_1D_OPTION: d = 1; break;
        case DIM_2D_OPTION: d = 2; break;
        case DIM_3D_OPTION: d = 3; break;
        case DIM_4D_OPTION: d = 4; break;
        case DIM_5D_OPTION: d = 5; break;
        default:
          DPRINTF_INST_EXEC(TMA, "ERROR: Unknown dimension option %d\n", dim_option);
          abort();
      }
      return d;
    };

    if ((dst_option == CTA_OPTION || dst_option == CLUSTER_OPTION) && 
        src_option == GLOBAL_OPTION && completion_option == TMA_MBAR_COMPLETE_BYTES) {
      // cp.async.bulk.tensor.Nd.shared::cta.global.mbarrier::complete_tx::bytes
      // cp.async.bulk.tensor.Nd.shared::cluster.global.mbarrier::complete_tx::bytes
      // Operands: [dst_shared], [tensormap, {coords...}], [mbar], ignore cache-hint for now
      
      auto dst_addr = get_u32_value(pI->dst());
      uint64_t tensormap_addr = get_u64_value(pI->src1());      
      auto mbar_addr = get_u32_value(pI->src3());

      // Read tensormap descriptor from global memory
      flash_gpgpu_sim::tensormap_descriptor_t tensormap;
      global_mem->read(tensormap_addr, TENSORMAP_DESCRIPTOR_SIZE, &tensormap);
      if(!tensormap.is_valid()) {
        tensormap.print();
        fflush(stdout);
        exit(1);
      }
      unsigned tensormap_dim = tensormap.num_dims();
      unsigned inst_dim = compute_inst_dim(dim_option);
      if (tensormap_dim != inst_dim) {
        DPRINTF_INST_EXEC(TMA,
          "ERROR: Dimension mismatch - instruction is %uD but tensormap num_dims is %u (dim=%uD)\n",
          inst_dim, tensormap_dim, tensormap_dim);
        fflush(stdout);
        exit(1);
      }
      
      // Calculate transfer size from tensormap
      uint32_t size_in_bytes = tensormap.get_tile_size_bytes();
      assert(size_in_bytes > 0 && "TMA tensor load size cannot be zero");
      
      // Get coordinates from operands (between tensormap and mbar)
      // For 1D/2D/3D/4D/5D: coords are {coord0, coord1, ...}
      const auto &coord_operand = pI->get_operands()[2];

      uint32_t coords[5] = {0, 0, 0, 0, 0};
      if (coord_operand.is_vector()) {
        unsigned n_coords = coord_operand.get_vect_nelem();
        ptx_reg_t coord_regs[8];
        thread->get_vector_operand_values(coord_operand, coord_regs, n_coords);
        
        for (unsigned i = 0; i < n_coords && i < 5; i++) {
          coords[i] = coord_regs[i].u32;
        }
      } else {
        // Single coordinate (1D case with non-vector operand)
        coords[0] = thread->get_operand_value(coord_operand, coord_operand, U32_TYPE, thread, 0).u32;
      }
      
      // DPRINTF_INST_EXEC(TMA, "TMA tensor load Extracted coordinates: [%u, %u, %u, %u, %u]\n", 
      //           coords[0], coords[1], coords[2], coords[3], coords[4]);

      // Pass info to perf sim
      inst_t::tma_static_info_t tma_static_info{
          .dst_space = inst_t::tma_static_info_t::TMA_SHARED_CTA, // treat shared::cluster same as shared::cta
          .src_space = inst_t::tma_static_info_t::TMA_GLOBAL,
      };
      pI->set_tma_static_info(tma_static_info);
      
      inst_t::tma_dyn_info_t tma_dyn_info{
          .is_tensor = true, 
          .dst_addr = dst_addr,
          .src_addr = tensormap_addr,
          .size_in_bytes = size_in_bytes,
          .mbar_addr = mbar_addr,
      };
      for (unsigned i = 0; i < 5; ++i) tma_dyn_info.coords[i] = coords[i];

      auto laneid = thread->get_laneid();
      pI->set_tma_dyn_info(laneid, tma_dyn_info);
      
      // Functional simulation: Copy data from global to shared
      auto memory_requests = flash_gpgpu_sim::generate_tma_requests(tensormap, coords);
      
      // DPRINTF_INST_EXEC(TMA, "Generated %lu memory requests for TMA tensor load:\n", 
      //                   memory_requests.size());
      // for (size_t i = 0; i < memory_requests.size(); i++) {
      //   DPRINTF_INST_EXEC(TMA, "  Request[%lu]: addr=0x%llx, size=%u bytes\n", 
      //                     i, (unsigned long long)memory_requests[i].first, 
      //                     memory_requests[i].second);
      // }

      uint64_t base_src_addr = tensormap.calculate_src_addr(coords);
      for (const auto &req : memory_requests) {
        uint64_t req_addr = req.first;
        uint32_t req_size = req.second;
        unsigned char *data_buffer = new unsigned char[req_size];
        global_mem->read(req_addr, req_size, data_buffer);
        // Use tile-local offset (accounting for stride difference between global and tile)
        uint64_t tile_offset = flash_gpgpu_sim::global_to_tile_offset(req_addr, base_src_addr, tensormap);
        shared_mem->write(dst_addr + tile_offset, req_size, data_buffer, thread, pI);
        delete[] data_buffer;
      }
      
      // DPRINTF_INST_EXEC(TMA,
      //   "Functional Sim: TMA tensor load dst=0x%x, src=0x%llx, "
      //   "size=%u, mbar=0x%x, tensormap=0x%x\n",
      //   dst_addr, (unsigned long long)base_src_addr, size_in_bytes, mbar_addr, tensormap_addr);
        
    } else if (dst_option == GLOBAL_OPTION && src_option == CTA_OPTION) {
      // cp.async.bulk.tensor.Nd.global.shared::cta.bulk_group
      // This is a store operation, no mbarrier involved
      // Operands: [tensormap, {coords...}], [src_shared], ignore cache-hint for now
      
      // Get tensormap address
      auto tensormap_addr = get_u32_value(pI->dst());  // dst is tensormap for store
      auto src_addr = get_u32_value(pI->src2());       // src is shared memory
      
      // Read tensormap descriptor
      flash_gpgpu_sim::tensormap_descriptor_t tensormap;
      global_mem->read(tensormap_addr, TENSORMAP_DESCRIPTOR_SIZE, &tensormap);
      if(!tensormap.is_valid()) {
        tensormap.print();
        fflush(stdout);
        exit(1);
      }
      unsigned tensormap_dim = tensormap.num_dims();
      unsigned inst_dim = compute_inst_dim(dim_option);
      if (tensormap_dim != inst_dim) {
        DPRINTF_INST_EXEC(TMA,
          "ERROR: Dimension mismatch - instruction is %uD but tensormap num_dims is %u (dim=%uD)\n",
          inst_dim, tensormap_dim, tensormap_dim);
        fflush(stdout);
        exit(1);
      }
      
      // Calculate transfer size from tensormap
      uint32_t size_in_bytes = tensormap.get_tile_size_bytes();
      assert(size_in_bytes > 0 && "TMA tensor load size cannot be zero");
      
      // Get coordinates from operands (between tensormap and src_addr)
      // For 1D/2D/3D/4D/5D: coords are {coord0, coord1, ...}
      const auto &coord_operand = pI->get_operands()[1];

      uint32_t coords[5] = {0, 0, 0, 0, 0};
      if (coord_operand.is_vector()) {
        unsigned n_coords = coord_operand.get_vect_nelem();
        ptx_reg_t coord_regs[8];
        thread->get_vector_operand_values(coord_operand, coord_regs, n_coords);
        
        for (unsigned i = 0; i < n_coords && i < 5; i++) {
          coords[i] = coord_regs[i].u32;
        }
      } else {
        // Single coordinate (1D case with non-vector operand)
        coords[0] = thread->get_operand_value(coord_operand, coord_operand, U32_TYPE, thread, 0).u32;
      }
      
      DPRINTF_INST_EXEC(TMA, "TMA tensore store Extracted coordinates: [%u, %u, %u, %u, %u]\n", 
                coords[0], coords[1], coords[2], coords[3], coords[4]);
      
      inst_t::tma_static_info_t tma_static_info{
          .dst_space = inst_t::tma_static_info_t::TMA_GLOBAL,
          .src_space = inst_t::tma_static_info_t::TMA_SHARED_CTA,
      };
      pI->set_tma_static_info(tma_static_info);

      // TODO: set dyn info
      
      // Functional simulation: Copy data from shared to global
      // Use generate_tma_requests to get the global memory request layout
      auto memory_requests = flash_gpgpu_sim::generate_tma_requests(tensormap, coords);
      
      // base_dst_addr: the starting global address for this tile
      uint64_t base_dst_addr = tensormap.calculate_src_addr(coords);
      
      for (const auto &req : memory_requests) {
        uint64_t global_req_addr = req.first;
        uint32_t req_size = req.second;
        
        // Calculate the tile-local offset (accounting for stride difference between global and tile)
        uint64_t tile_offset = flash_gpgpu_sim::global_to_tile_offset(global_req_addr, base_dst_addr, tensormap);
        
        // Read from shared memory (src_addr + tile_offset)
        unsigned char *data_buffer = new unsigned char[req_size];
        shared_mem->read(src_addr + tile_offset, req_size, data_buffer);
        
        // Write to global memory at the calculated address
        global_mem->write(global_req_addr, req_size, data_buffer, thread, pI);
        
        delete[] data_buffer;
      }

      DPRINTF_INST_EXEC(TMA,
        "Functional Sim: TMA tensor store dst=0x%llx, src=0x%x, "
        "size=%u, tensormap=0x%x\n",
        (unsigned long long)base_dst_addr, src_addr, size_in_bytes, tensormap_addr);
        
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
// TMA Memory Request Generation
//=============================================================================

namespace flash_gpgpu_sim {

// Helper function to convert a global address offset to tile-local offset
// The global tensor may have a different row stride than the tile (based on box dimensions)
// For example: global tensor has 4096 cols, tile has 32 cols
// Global row stride = 4096 * elem_size, tile row stride = 32 * elem_size
// This function converts (global_addr - base_global_addr) to the equivalent tile offset
static uint64_t global_to_tile_offset(uint64_t global_addr, uint64_t base_addr,
                                       const tensormap_descriptor_t& tensormap) {
  uint32_t elem_size = tensormap.get_element_size();
  uint32_t num_dims = tensormap.num_dims();
  
  // Calculate the byte offset in global memory space
  uint64_t global_byte_offset = global_addr - base_addr;
  
  if (num_dims == 1) {
    // 1D: tile offset equals global offset
    return global_byte_offset;
  }
  
  // For 2D and higher, we need to decompose the global offset into coordinates
  // and then recompute using tile strides
  
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
  
  // Decompose global_byte_offset into coordinates (from highest to lowest dimension)
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

// Helper function to generate 128B-aligned memory fetch requests
// Splits a contiguous memory range into cache-line-aligned requests
static void gen_aligned_req(uint64_t start_addr, uint32_t total_bytes,
                            std::vector<std::pair<uint64_t, uint32_t>>& requests) {
  if (total_bytes == 0) return;
  
  constexpr uint32_t CACHE_LINE_SIZE = 128;
  uint64_t current_addr = start_addr;
  uint32_t remaining_bytes = total_bytes;
  
  while (remaining_bytes > 0) {
    // Calculate bytes to next cache line boundary
    uint64_t next_boundary = (current_addr + CACHE_LINE_SIZE) & ~(CACHE_LINE_SIZE - 1);
    uint32_t bytes_to_boundary = static_cast<uint32_t>(next_boundary - current_addr);
    
    // Determine request size: min(remaining, bytes_to_boundary, CACHE_LINE_SIZE)
    uint32_t request_size = std::min({remaining_bytes, bytes_to_boundary, CACHE_LINE_SIZE});
    
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
static void traverse_tensor_dim(int dim, 
                                const uint32_t current_coords[5],
                                uint64_t base_addr,
                                const tensormap_descriptor_t& tensormap,
                                std::vector<std::pair<uint64_t, uint32_t>>& requests) {
  uint32_t elem_size = tensormap.get_element_size();
  
  if (dim == 0) {
    // Base case: innermost dimension (contiguous)
    uint32_t start_x = current_coords[0];
    uint32_t box_width = tensormap.fields.boxDim[0];
    uint32_t global_width = tensormap.fields.globalDim[0];
    
    // Calculate valid range intersection: [start_x, start_x + box_width) ∩ [0, global_width)
    uint32_t valid_start = start_x;
    uint32_t valid_end = start_x + box_width;
    
    // Check if completely out of bounds
    if (valid_start >= global_width || valid_end <= 0) {
      return; // No valid data, skip
    }
    
    // Clamp to valid tensor boundaries
    valid_start = std::max(valid_start, 0u);
    valid_end = std::min(valid_end, global_width);
    
    if (valid_start >= valid_end) return;
    
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
      traverse_tensor_dim(dim - 1, current_coords, next_addr, tensormap, requests);
    }
  }
}

// Main entry point: Generate TMA memory fetch requests
// start_coords: starting coordinate for each dimension (e.g., {x, y, z, w, v})
// Returns: vector of (physical_address, size_in_bytes) pairs
std::vector<std::pair<uint64_t, uint32_t>> 
generate_tma_requests(const tensormap_descriptor_t& tensormap, 
                      const uint32_t start_coords[5]) {
  std::vector<std::pair<uint64_t, uint32_t>> requests;
  
  if (!tensormap.is_valid() || tensormap.fields.tensorRank > 4) {
    return requests; // Empty result for invalid tensormap
  }
  
  // Start recursive traversal from highest dimension (0-based index)
  int highest_dim = static_cast<int>(tensormap.num_dims()) - 1; // highest dimension index (0-based)
  uint64_t base_addr = tensormap.fields.globalAddress;
  
  traverse_tensor_dim(highest_dim, start_coords, base_addr, tensormap, requests);
  
  return requests;
}

} // namespace flash_gpgpu_sim

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
  if (fields.tensorRank > 4) return 0;
  
  uint32_t total_elements = 1;
  uint32_t dims = num_dims();
  for (uint32_t i = 0; i < dims; i++) {
    total_elements *= fields.boxDim[i];
  }
  return total_elements * get_element_size();
}

uint64_t tensormap_descriptor_t::calculate_src_addr(const uint32_t coords[5]) const {
  // Calculate the base address for the tile starting at given coordinates
  // This is used for simple address calculation (not for generating actual memory requests)
  // Assume fields.tensorRank is 0-based.
  uint64_t addr = fields.globalAddress;
  uint32_t elem_size = get_element_size();
  uint32_t dims = num_dims();
  
  for (uint32_t i = 0; i < dims; i++) {
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
  char buf[1024];
  size_t pos = 0;
  uint32_t dims = num_dims();

  auto append_checked = [&](const char *fmt, ...) {
    if (pos >= sizeof(buf)) return;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + pos, sizeof(buf) - pos, fmt, ap);
    va_end(ap);
    if (n < 0) { pos = sizeof(buf) - 1; buf[pos] = '\0'; return; }
    size_t rem = sizeof(buf) - pos;
    if ((size_t)n >= rem) { pos = sizeof(buf) - 1; buf[pos] = '\0'; }
    else pos += (size_t)n;
  };

  append_checked("TensorMap Descriptor:\n");
  append_checked("  global_address: 0x%llx\n", (unsigned long long)fields.globalAddress);
  append_checked("  rank: %u (num_dims=%u)\n", fields.tensorRank, dims);
  append_checked("  elemtype: %u (size=%u bytes)\n", fields.tensorDataType, get_element_size());

  append_checked("  box_dim: [");
  for (uint32_t i = 0; i < dims; i++) {
    if (i > 0) append_checked(", ");
    append_checked("%u", fields.boxDim[i]);
  }
  append_checked("]\n");

  append_checked("  global_dim: [");
  for (uint32_t i = 0; i < dims; i++) {
    if (i > 0) append_checked(", ");
    append_checked("%u", fields.globalDim[i]);
  }
  append_checked("]\n");

  append_checked("  global_strides: [");
  if (dims > 1) {
    for (uint32_t i = 0; i < dims - 1; i++) {
      if (i > 0) append_checked(", ");
      append_checked("0x%llx", (unsigned long long)fields.globalStrides[i]);
    }
  }
  append_checked("]\n");

  append_checked("  element_strides: [");
  for (uint32_t i = 0; i < dims; i++) {
    if (i > 0) append_checked(", ");
    append_checked("%u", fields.elementStrides[i]);
  }
  append_checked("]\n");

  // Map numeric mode values to human-readable macro names.
  auto interleave_to_string = [](uint32_t v) -> const char* {
    switch (v) {
      case TMA_INTERLEAVE_NONE: return "TMA_INTERLEAVE_NONE";
      case TMA_INTERLEAVE_16B:  return "TMA_INTERLEAVE_16B";
      case TMA_INTERLEAVE_32B:  return "TMA_INTERLEAVE_32B";
      default: return "TMA_INTERLEAVE_UNKNOWN";
    }
  };
  auto swizzle_to_string = [](uint32_t v) -> const char* {
    switch (v) {
      case TMA_SWIZZLE_NONE:  return "TMA_SWIZZLE_NONE";
      case TMA_SWIZZLE_32B:   return "TMA_SWIZZLE_32B";
      case TMA_SWIZZLE_64B:   return "TMA_SWIZZLE_64B";
      case TMA_SWIZZLE_128B:  return "TMA_SWIZZLE_128B";
      case TMA_SWIZZLE_96B:   return "TMA_SWIZZLE_96B";
      default: return "TMA_SWIZZLE_UNKNOWN";
    }
  };
  auto oobfill_to_string = [](uint32_t v) -> const char* {
    switch (v) {
      case TMA_OOB_ZERO: return "TMA_OOB_ZERO";
      case TMA_OOB_NAN:  return "TMA_OOB_NAN";
      default: return "TMA_OOB_UNKNOWN";
    }
  };

  append_checked("  interleave: %s (%u), swizzle: %s (%u), oobFill: %s (%u)\n",
                 interleave_to_string(fields.interleave), fields.interleave,
                 swizzle_to_string(fields.swizzle), fields.swizzle,
                 oobfill_to_string(fields.oobFill), fields.oobFill);

  append_checked("  tile_size_bytes: %u\n", get_tile_size_bytes());

  printf("%s", buf);
}

tensormap_descriptor_t tensormap_descriptor_t::read_from_shared(memory_space *shared_mem, uint32_t addr) {
  tensormap_descriptor_t desc;
  // Read entire 128-byte descriptor as raw bytes
  shared_mem->read(addr, 128, desc.raw_bytes);
  return desc;
}

void tensormap_descriptor_t::write_to_shared(memory_space *shared_mem, uint32_t addr,                                      
                                             ptx_thread_info *thd,
                                            const ptx_instruction *pI) const {
  // Write entire 128-byte descriptor as raw bytes
  shared_mem->write(addr, 128, raw_bytes, thd, pI);
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
    desc.write_to_shared(shared_mem, tensormap_addr, thread, pI);
    
  } else if (is_cp_fenceproxy) {
    // tensormap.cp_fenceproxy.global.shared::cta... [dst_global], [src_shared], size
    // Copy tensormap from shared to global memory with fence
    
    // TODO: Add option check
    uint64_t dst_addr = get_u64_value(pI->dst());
    uint32_t src_addr = get_u32_value(pI->src1());
    uint32_t size_in_bytes = get_u32_value(pI->src2());

    memory_space *global_mem = thread->get_global_memory();;
    
    tensormap_descriptor_t desc = tensormap_descriptor_t::read_from_shared(shared_mem, src_addr);
    global_mem->write(dst_addr, size_in_bytes, desc.raw_bytes, thread, pI);

    // DPRINTF_INST_EXEC(TMA, "tensormap.cp_fenceproxy: copy tensormap to global dst=0x%llx, src=0x%x, size=%u (fence ignored)\n", 
    //           (unsigned long long)dst_addr, src_addr, size_in_bytes);
    
  } else {
    DPRINTF_INST_EXEC(TMA, "[STUB] Unrecognized tensormap instruction variant%s\n", "");
  }
}