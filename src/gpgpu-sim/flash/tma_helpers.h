#ifndef FLASH_GPGPU_SIM_TMA_HELPERS_H
#define FLASH_GPGPU_SIM_TMA_HELPERS_H

#include <cstdint>
#include <utility>
#include <vector>

#include "tensormap.h"

namespace flash_gpgpu_sim {

//===========================================================================
// TMA AGU (Address Generation Unit)
//===========================================================================

// State tracked by the AGU for each in-flight TMA transaction.
struct tma_agu_state_t {
  bool is_tensor = false;
  bool done = true;
  bool is_fill_request = false;

  uint32_t num_dims = 0;
  uint32_t elem_size = 0;
  uint32_t box_dim[5] = {0};
  uint32_t global_dim[5] = {0};
  uint32_t start_coords[5] = {0};
  uint64_t global_strides[5] = {0};
  uint32_t tile_coords[5] = {0};
  uint64_t curr_row_addr = 0;
  uint32_t offset_in_row = 0;
  uint32_t row_bytes = 0;

  bool row_is_oob = false;
  uint32_t valid_row_bytes = 0;

  uint64_t linear_addr = 0;
  uint32_t linear_remaining = 0;
};

class tma_agu_unit_t {
public:
  void init_tensor(tma_agu_state_t &state, const tensormap_descriptor_t &tm,
                   const uint32_t start_coords[5]);

  void init_linear(tma_agu_state_t &state, uint64_t start_addr,
                   uint32_t total_bytes);

  bool gen_next_req(tma_agu_state_t &state, uint64_t &out_addr,
                    uint32_t &out_size);

private:
  void advance_to_next_row(tma_agu_state_t &state);
};

//===========================================================================
// Pure helper functions (no simulator dependencies)
//===========================================================================

// Convert global address to tile offset using tensor descriptor.
uint64_t global_to_tile_offset(uint64_t global_addr, uint64_t base_addr,
                               const tensormap_descriptor_t &tensormap);

// Apply TMA swizzle to a shared memory address.
uint64_t apply_tma_swizzle(uint64_t linear_offset, uint32_t smem_base_addr,
                           uint32_t swizzle_mode, uint32_t row_bytes);

// Generate 128B-aligned memory fetch requests from a contiguous range.
void gen_aligned_req(uint64_t start_addr, uint32_t total_bytes,
                     std::vector<std::pair<uint64_t, uint32_t>> &requests);

// Generate TMA memory fetch requests for a tensor tile.
std::vector<std::pair<uint64_t, uint32_t>>
generate_tma_requests(const tensormap_descriptor_t &tensormap,
                      const uint32_t start_coords[5]);

//===========================================================================
// OOB Fill Pattern Table
//===========================================================================

class tma_oob_fill_table_t {
public:
  static constexpr uint32_t CHUNK_SIZE = 128;
  static constexpr uint32_t NUM_DTYPES = 16;

  alignas(16) unsigned char patterns[2][NUM_DTYPES][CHUNK_SIZE];
  bool nan_supported[NUM_DTYPES];

  tma_oob_fill_table_t();

  const unsigned char *get_pattern(uint32_t oob_mode, uint32_t dtype) const;

  static const tma_oob_fill_table_t &instance();
};

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_TMA_HELPERS_H
