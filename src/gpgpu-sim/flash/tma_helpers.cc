#include "tma_helpers.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>

namespace flash_gpgpu_sim {

//=============================================================================
// tma_agu_unit_t Implementation
//=============================================================================

void tma_agu_unit_t::init_tensor(tma_agu_state_t &state,
                                 const tensormap_descriptor_t &tm,
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

void tma_agu_unit_t::init_linear(tma_agu_state_t &state, uint64_t start_addr,
                                 uint32_t total_bytes) {
  state = tma_agu_state_t(); // Reset to defaults
  state.is_tensor = false;
  state.linear_addr = start_addr;
  state.linear_remaining = total_bytes;
  state.done = (total_bytes == 0);
}

void tma_agu_unit_t::advance_to_next_row(tma_agu_state_t &state) {
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

static constexpr uint32_t SECTOR_SIZE = 32;

bool tma_agu_unit_t::gen_next_req(tma_agu_state_t &state, uint64_t &out_addr,
                                  uint32_t &out_size) {
  if (state.done)
    return false;

  if (!state.is_tensor) {
    // Linear mode: simple sequential address generation
    if (state.linear_remaining == 0) {
      state.done = true;
      return false;
    }

    state.is_fill_request = false;
    out_addr = state.linear_addr;
    uint32_t to_boundary = SECTOR_SIZE - (state.linear_addr % SECTOR_SIZE);
    out_size =
        std::min({(uint32_t)SECTOR_SIZE, state.linear_remaining, to_boundary});

    state.linear_addr += out_size;
    state.linear_remaining -= out_size;

    if (state.linear_remaining == 0) {
      state.done = true;
    }

  } else {
    // Tensor mode: odometer-style multi-dimensional traversal
    if (state.row_is_oob) {
      state.is_fill_request = true;
    } else {
      state.is_fill_request = (state.offset_in_row >= state.valid_row_bytes);
    }

    out_addr = state.curr_row_addr + state.offset_in_row;

    uint32_t row_remaining = state.row_bytes - state.offset_in_row;
    uint32_t to_sector_boundary = SECTOR_SIZE - (out_addr % SECTOR_SIZE);
    out_size =
        std::min({(uint32_t)SECTOR_SIZE, row_remaining, to_sector_boundary});

    state.offset_in_row += out_size;

    if (state.offset_in_row >= state.row_bytes) {
      advance_to_next_row(state);
    }
  }

  return true;
}

//=============================================================================
// Pure Helper Functions
//=============================================================================

uint64_t global_to_tile_offset(uint64_t global_addr, uint64_t base_addr,
                               const tensormap_descriptor_t &tensormap) {
  uint32_t elem_size = tensormap.get_element_size();
  uint32_t num_dims = tensormap.num_dims();

  assert(global_addr >= base_addr && "global_addr must be >= base_addr");
  uint64_t global_byte_offset = global_addr - base_addr;

  if (num_dims == 1) {
    return global_byte_offset;
  }

  uint64_t global_strides[5];
  global_strides[0] = elem_size;
  for (uint32_t d = 1; d < num_dims; d++) {
    global_strides[d] = tensormap.fields.globalStrides[d - 1];
  }

  uint64_t tile_strides[5];
  tile_strides[0] = elem_size;
  for (uint32_t d = 1; d < num_dims; d++) {
    tile_strides[d] = tile_strides[d - 1] * tensormap.fields.boxDim[d - 1];
  }

  uint64_t tile_offset = 0;
  uint64_t remaining = global_byte_offset;

  for (int d = num_dims - 1; d >= 0; d--) {
    uint64_t coord_in_tile = remaining / global_strides[d];
    remaining = remaining % global_strides[d];
    tile_offset += coord_in_tile * tile_strides[d];
  }

  return tile_offset;
}

uint64_t apply_tma_swizzle(uint64_t linear_offset, uint32_t smem_base_addr,
                           uint32_t swizzle_mode, uint32_t row_bytes) {
  (void)smem_base_addr; // reserved for future use
  (void)row_bytes;      // reserved for future use

  if (swizzle_mode == TMA_SWIZZLE_NONE)
    return linear_offset;

  uint32_t mask = 0;
  constexpr uint32_t shift = 4;

  switch (swizzle_mode) {
  case TMA_SWIZZLE_128B:
    mask = 0x7;
    break;
  case TMA_SWIZZLE_64B:
    mask = 0x3;
    break;
  case TMA_SWIZZLE_32B:
    mask = 0x1;
    break;
  case TMA_SWIZZLE_96B:
    printf("ERROR: TMA 96B swizzle mode is not yet implemented\n");
    abort();
  default:
    printf("ERROR: Unknown TMA swizzle mode %u\n", swizzle_mode);
    abort();
  }

  uint32_t row_bits = (uint32_t)(linear_offset >> 7);
  return linear_offset ^ ((uint64_t)(row_bits & mask) << shift);
}

void gen_aligned_req(uint64_t start_addr, uint32_t total_bytes,
                     std::vector<std::pair<uint64_t, uint32_t>> &requests) {
  if (total_bytes == 0)
    return;

  constexpr uint32_t CACHE_LINE_SIZE = 128;
  uint64_t current_addr = start_addr;
  uint32_t remaining_bytes = total_bytes;

  while (remaining_bytes > 0) {
    uint64_t next_boundary =
        (current_addr + CACHE_LINE_SIZE) & ~(CACHE_LINE_SIZE - 1);
    uint32_t bytes_to_boundary =
        static_cast<uint32_t>(next_boundary - current_addr);

    uint32_t request_size =
        std::min({remaining_bytes, bytes_to_boundary, CACHE_LINE_SIZE});

    requests.push_back({current_addr, request_size});

    current_addr += request_size;
    remaining_bytes -= request_size;
  }
}

// Recursive helper for generate_tma_requests
static void
traverse_tensor_dim(int dim, const uint32_t current_coords[5],
                    uint64_t base_addr, const tensormap_descriptor_t &tensormap,
                    std::vector<std::pair<uint64_t, uint32_t>> &requests) {
  uint32_t elem_size = tensormap.get_element_size();

  if (dim == 0) {
    uint32_t start_x = current_coords[0];
    uint32_t box_width = tensormap.fields.boxDim[0];
    uint32_t global_width = tensormap.fields.globalDim[0];

    uint32_t valid_start = start_x;
    uint32_t valid_end = start_x + box_width;

    if (valid_start >= global_width || valid_end <= 0) {
      return;
    }

    valid_start = std::max(valid_start, 0u);
    valid_end = std::min(valid_end, global_width);

    if (valid_start >= valid_end)
      return;

    uint64_t phys_start_addr = base_addr + (valid_start * elem_size);
    uint32_t valid_bytes = (valid_end - valid_start) * elem_size;

    gen_aligned_req(phys_start_addr, valid_bytes, requests);

  } else {
    uint32_t box_extent = tensormap.fields.boxDim[dim];
    uint32_t global_extent = tensormap.fields.globalDim[dim];
    uint64_t stride = tensormap.fields.globalStrides[dim - 1];

    for (uint32_t i = 0; i < box_extent; i++) {
      uint32_t current_coord = current_coords[dim] + i;

      if (current_coord >= global_extent) {
        continue;
      }

      uint64_t next_addr = base_addr + (current_coord * stride);
      traverse_tensor_dim(dim - 1, current_coords, next_addr, tensormap,
                          requests);
    }
  }
}

std::vector<std::pair<uint64_t, uint32_t>>
generate_tma_requests(const tensormap_descriptor_t &tensormap,
                      const uint32_t start_coords[5]) {
  std::vector<std::pair<uint64_t, uint32_t>> requests;

  if (!tensormap.is_valid() || tensormap.fields.tensorRank > 4) {
    return requests;
  }

  int highest_dim = static_cast<int>(tensormap.num_dims()) - 1;
  uint64_t base_addr = tensormap.fields.globalAddress;

  traverse_tensor_dim(highest_dim, start_coords, base_addr, tensormap,
                      requests);

  return requests;
}

//=============================================================================
// tma_oob_fill_table_t Implementation
//=============================================================================

tma_oob_fill_table_t::tma_oob_fill_table_t() {
  for (uint32_t dtype = 0; dtype < NUM_DTYPES; dtype++) {
    memset(patterns[TMA_OOB_ZERO][dtype], 0, CHUNK_SIZE);

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
      nan_supported[dtype] = true;
      memset(patterns[TMA_OOB_NAN][dtype], 0x7F, CHUNK_SIZE);
      break;
    }
    default:
      nan_supported[dtype] = false;
      memset(patterns[TMA_OOB_NAN][dtype], 0, CHUNK_SIZE);
      break;
    }
  }
}

const unsigned char *tma_oob_fill_table_t::get_pattern(uint32_t oob_mode,
                                                       uint32_t dtype) const {
  if (dtype >= NUM_DTYPES)
    dtype = 0;
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

const tma_oob_fill_table_t &tma_oob_fill_table_t::instance() {
  static tma_oob_fill_table_t g_instance;
  return g_instance;
}

} // namespace flash_gpgpu_sim
