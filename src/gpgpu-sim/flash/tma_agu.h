#pragma once
#include "tensormap.h"
#include "tma_gather4.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
namespace flash_gpgpu_sim {
struct tma_agu_state_t {
  // Mode and completion state
  bool is_gather4 = false;
  int32_t gather4_rows[4] = {};
  uint64_t gather4_column_base = 0;
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

    // Calculate strides: stride[0] = elem_size, stride[i] = globalStrides[i-1]
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
    int64_t byte_offset = 0;
    for (uint32_t d = 0; d < state.num_dims; d++) {
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

    state.row_bytes = state.box_dim[0] * state.elem_size;
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
      state.valid_row_start_bytes =
          static_cast<uint32_t>(valid_start - row_start) * state.elem_size;
      state.valid_row_end_bytes =
          static_cast<uint32_t>(valid_end - row_start) * state.elem_size;
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

  // Port of d6e05c05's gather4 AGU, retaining signed/OOB handling from
  // the current ordinary-tile implementation. Slot order is shared-memory
  // order; physical row indices determine only the global requests.
  void init_tensor_gather4(tma_agu_state_t &state,
                           const tensormap_descriptor_t &tm,
                           const int32_t coords[5]) {
    assert(tm.num_dims() == 2 && tm.fields.boxDim[1] == 1);
    assert(tm.fields.elementStrides[0] == 1);
    assert(tma_gather4_supported(tm));
    auto memory_map = tm;
    int32_t first[5] = {coords[0], coords[1], 0, 0, 0};
    if (tma_gather4_packed(tm)) {
      const unsigned bits =
          tm.fields.tensorDataType == TMA_DTYPE_16U6_ALIGN16B ? 6 : 4;
      const int64_t bit_origin = int64_t(coords[0]) * bits;
      const unsigned bit_offset = (bit_origin % 8 + 8) % 8;
      memory_map.fields.tensorDataType = TMA_DTYPE_U8;
      memory_map.fields.boxDim[0] =
          (tm.fields.boxDim[0] * bits + bit_offset + 7) / 8;
      memory_map.fields.globalDim[0] =
          uint64_t(tm.fields.globalDim[0]) * bits / 8;
      first[0] = (bit_origin - bit_offset) / 8;
    }
    init_tensor(state, memory_map, first);
    state.is_gather4 = true;
    state.box_dim[1] = 4;
    state.start_coords[1] = 0;
    state.gather4_column_base =
        tm.fields.globalAddress + int64_t(first[0]) * state.elem_size;
    for (unsigned row = 0; row < 4; ++row)
      state.gather4_rows[row] = coords[row + 1];
    set_gather4_row(state);
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
  void set_gather4_row(tma_agu_state_t &state) {
    const unsigned slot = state.tile_coords[1];
    if (slot == 4) {
      state.done = true;
      return;
    }
    const int32_t row = state.gather4_rows[slot];
    state.curr_row_addr = state.gather4_column_base +
                          static_cast<uint64_t>(static_cast<int64_t>(row)) *
                              state.global_strides[1];
    state.row_is_oob =
        row < 0 || static_cast<uint32_t>(row) >= state.global_dim[1];
  }

  // Advance to next row using odometer-style increment of higher dimensions.
  // Updates row_is_oob cache after advancing.
  void advance_to_next_row(tma_agu_state_t &state) {
    state.offset_in_row = 0;
    if (state.is_gather4) {
      if (++state.tile_coords[1] == 4)
        state.done = true;
      else
        set_gather4_row(state);
      return;
    }

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

} // namespace flash_gpgpu_sim
