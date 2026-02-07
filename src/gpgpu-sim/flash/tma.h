#ifndef FLASH_GPGPU_SIM_TMA_H
#define FLASH_GPGPU_SIM_TMA_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <list>
#include <memory>
#include <vector>

#include "tensormap.h"

class shader_core_ctx;
class warp_inst_t;
class barrier_set_t;
class ptx_instruction;
class ptx_thread_info;
class mem_fetch_interface;
class shader_core_mem_fetch_allocator;
class mem_fetch;
class memory_space;

namespace flash_gpgpu_sim {

class tma_unit_impl_t;
class tma_unit_t {
public:
  tma_unit_t(shader_core_ctx *shader_ctx, barrier_set_t *barriers,
             mem_fetch_interface *icnt,
             shader_core_mem_fetch_allocator *mf_allocator);
  ~tma_unit_t();

  void warp_reaches_tma(unsigned cta_id, unsigned warp_id, warp_inst_t *inst);
  void cycle();

  void fill(mem_fetch *mf);
  bool response_buffer_full() const;

private:
  std::unique_ptr<tma_unit_impl_t> m_impl;
};

// Generate memory fetch requests for TMA tensor operations
// start_coords: starting coordinate for each dimension [x, y, z, w, v]
// Returns: vector of (physical_address, size_in_bytes) pairs for memory fetches
std::vector<std::pair<uint64_t, uint32_t>>
generate_tma_requests(const tensormap_descriptor_t &tensormap,
                      const uint32_t start_coords[5]);

} // namespace flash_gpgpu_sim

void handle_tma_inst(const ptx_instruction *pIin, ptx_thread_info *thread);

#endif