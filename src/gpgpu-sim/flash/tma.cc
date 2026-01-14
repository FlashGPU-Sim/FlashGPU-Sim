#include "tma.h"
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

//=============================================================================
// TMA Swizzle Debug Configuration
//=============================================================================
// Set TMA_DEBUG_SWIZZLE to 1 to enable detailed 16B-level swizzle debug output
// This will print:
//   - For LOAD: gmem linear -> smem swizzled mapping with data values
//   - For STORE: smem swizzled -> gmem linear reverse mapping with data values
// Set to 0 to disable all swizzle debug output
#define TMA_DEBUG_SWIZZLE 1

std::atomic<unsigned int> tma_next_tx_uid = 0;

namespace flash_gpgpu_sim {

//=============================================================================
// Helper Functions
//=============================================================================

// Get a 32-bit unsigned value from an operand
static uint32_t get_operand_u32(ptx_thread_info *thread, const operand_info &op) {
  ptx_reg_t reg = thread->get_operand_value(op, op, U32_TYPE, thread, 0);
  return reg.u32;
}

// Get a 64-bit unsigned value from an operand
static uint64_t get_operand_u64(ptx_thread_info *thread, const operand_info &op) {
  ptx_reg_t reg = thread->get_operand_value(op, op, U64_TYPE, thread, 0);
  return reg.u64;
}

// Compute instruction dimension from option
static unsigned compute_inst_dim(unsigned dim_option) {
  switch (dim_option) {
    case DIM_1D_OPTION: return 1;
    case DIM_2D_OPTION: return 2;
    case DIM_3D_OPTION: return 3;
    case DIM_4D_OPTION: return 4;
    case DIM_5D_OPTION: return 5;
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
  for (int i = 0; i < 5; i++) coords[i] = 0;
  
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
}

// Validate tensormap and dimension match
// Returns true if valid, false otherwise (also prints error and exits on failure)
static bool validate_tensormap(const tensormap_descriptor_t &tensormap,
                               unsigned inst_dim) {
  if (!tensormap.is_valid()) {
    tensormap.print();
    fflush(stdout);
    exit(1);
  }
  
  unsigned tensormap_dim = tensormap.num_dims();
  if (tensormap_dim != inst_dim) {
    printf("TMA ERROR: Dimension mismatch - instruction is %uD but tensormap num_dims is %u (dim=%uD)\n",
      inst_dim, tensormap_dim, tensormap_dim);
    fflush(stdout);
    exit(1);
  }
  return true;
}

// Convert global address offset to tile-local offset
// Global tensor may have different row stride than tile (based on box dimensions)
// This function converts (global_addr - base_global_addr) to equivalent tile offset
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
static uint64_t apply_tma_swizzle(uint64_t linear_offset, uint32_t smem_base_addr,
                                   uint32_t swizzle_mode, uint32_t row_bytes) {
  if (swizzle_mode == TMA_SWIZZLE_NONE) return linear_offset;

  uint32_t mask = 0;
  constexpr uint32_t shift = 4;  // only support 16B granularity for now

  switch (swizzle_mode) {
    case TMA_SWIZZLE_128B: mask = 0x7; break;  // 3 bits, cycle of 8
    case TMA_SWIZZLE_64B:  mask = 0x3; break;  // 2 bits, cycle of 4
    case TMA_SWIZZLE_32B:  mask = 0x1; break;  // 1 bit, cycle of 2
    case TMA_SWIZZLE_96B:  mask = 0x1; break;  // 1 bit, cycle of 2
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
//=============================================================================
// Static OOB Fill Pattern Table (precomputed at startup)
//=============================================================================

class tma_oob_fill_table_t {
public:
  static constexpr uint32_t CHUNK_SIZE = 128;  // Cache line size
  static constexpr uint32_t NUM_DTYPES = 16;   // Max data type index + 1
  
  // Fill patterns: [oob_mode][dtype] -> 128-byte pattern
  // oob_mode: 0 = ZERO, 1 = NAN
  alignas(16) unsigned char patterns[2][NUM_DTYPES][CHUNK_SIZE];
  bool nan_supported[NUM_DTYPES];  // Whether NaN is valid for this dtype
  
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
            memcpy(patterns[TMA_OOB_NAN][dtype] + i * sizeof(uint32_t), 
                   &nan_val, sizeof(uint32_t));
          }
          break;
        }
        case TMA_DTYPE_F64: {
          nan_supported[dtype] = true;
          uint64_t nan_val = 0x7FFFFFFFFFFFFFFFULL;
          for (uint32_t i = 0; i < CHUNK_SIZE / sizeof(uint64_t); i++) {
            memcpy(patterns[TMA_OOB_NAN][dtype] + i * sizeof(uint64_t), 
                   &nan_val, sizeof(uint64_t));
          }
          break;
        }
        case TMA_DTYPE_F16:
        case TMA_DTYPE_BF16: {
          nan_supported[dtype] = true;
          uint16_t nan_val = 0x7FFF;
          for (uint32_t i = 0; i < CHUNK_SIZE / sizeof(uint16_t); i++) {
            memcpy(patterns[TMA_OOB_NAN][dtype] + i * sizeof(uint16_t), 
                   &nan_val, sizeof(uint16_t));
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
  const unsigned char* get_pattern(uint32_t oob_mode, uint32_t dtype) const {
    if (dtype >= NUM_DTYPES) dtype = 0;  // Safety bound
    if (oob_mode == TMA_OOB_NAN && !nan_supported[dtype]) {
      static bool warned[NUM_DTYPES] = {false};
      if (!warned[dtype]) {
        printf("TMA ERROR: OOB_NAN not supported for data type %u, using zero fill\n", dtype);
        warned[dtype] = true;
      }
      return patterns[TMA_OOB_ZERO][dtype];
    }
    return patterns[oob_mode][dtype];
  }
};

// Global static instance (initialized at startup)
static tma_oob_fill_table_t g_oob_fill_table;

// Execute TMA data transfer (load or store)
// is_load=true: global -> shared, is_load=false: shared -> global
static void do_tma_transfer(const tensormap_descriptor_t &tensormap,
                            const uint32_t coords[5],
                            memory_space *shared_mem,
                            memory_space *global_mem,
                            uint32_t smem_addr,
                            ptx_thread_info *thread,
                            const ptx_instruction *pI,
                            bool is_load) {
  // For load operations, pre-fill the entire tile in shared memory with OOB fill value
  if (is_load) {
    uint32_t tile_size_bytes = tensormap.get_tile_size_bytes();
    const unsigned char* fill_pattern = g_oob_fill_table.get_pattern(
        tensormap.fields.oobFill, tensormap.fields.tensorDataType);
    
    // Write fill pattern to shared memory in chunks
    constexpr uint32_t CHUNK_SIZE = tma_oob_fill_table_t::CHUNK_SIZE;
    uint32_t offset = 0;
    while (offset < tile_size_bytes) {
      uint32_t chunk_size = std::min(tile_size_bytes - offset, CHUNK_SIZE);
      shared_mem->write(smem_addr + offset, chunk_size, fill_pattern, thread, pI);
      offset += chunk_size;
    }
  }

  // Generate memory requests based on tensormap and coordinates
  auto memory_requests = generate_tma_requests(tensormap, coords);

  // Calculate base address in global memory for this tile
  uint64_t base_global_addr = tensormap.calculate_src_addr(coords);

  // Get swizzle mode and row stride for swizzle calculation
  uint32_t swizzle_mode = tensormap.fields.swizzle;
  uint32_t elem_size = tensormap.get_element_size();
  uint32_t row_bytes = tensormap.fields.boxDim[0] * elem_size;

  // Swizzle granularity: 16 bytes (minimum addressable unit for swizzle)
  constexpr uint32_t SWIZZLE_GRANULARITY = 16;

  // Stack-allocated buffer for typical TMA request sizes (avoid malloc/free in hot path)
  constexpr uint32_t LOCAL_BUF_SIZE = 128;  // TMA requests are typically ≤128B
  alignas(16) unsigned char local_data_buf[LOCAL_BUF_SIZE];
  
  for (const auto &req : memory_requests) {
    uint64_t global_req_addr = req.first;
    uint32_t req_size = req.second;

    // Calculate tile-local offset (accounting for stride difference)
    uint64_t tile_offset = global_to_tile_offset(global_req_addr, base_global_addr, tensormap);

    // Use stack buffer for small requests, heap only for large (rare)
    unsigned char *data_buffer = (req_size <= LOCAL_BUF_SIZE) 
        ? local_data_buf 
        : new unsigned char[req_size];

    if (is_load) {
      // Load: global -> shared
      global_mem->read(global_req_addr, req_size, data_buffer);
      
      // Apply swizzle at 16-byte granularity when writing to shared memory
      // Each 16-byte sub-block may be written to a different swizzled location
      if (swizzle_mode != TMA_SWIZZLE_NONE) {
#if TMA_DEBUG_SWIZZLE
        // Debug: Print detailed data layout for first 5 tiles
        static bool printed_first_tile = false;
        static int tiles_printed = 0;
        bool print_data = (!printed_first_tile && tile_offset < 640);  // 5 rows * 128B
        
        if (print_data && tiles_printed < 5) {
          if (tiles_printed == 0) {
            printf("\n=== TMA LOAD Debug: Detailed Swizzle Analysis ===\n");
            printf("Global addr: 0x%lx, Smem addr: 0x%x\n", (unsigned long)global_req_addr, smem_addr);
            printf("Swizzle mode: %u, Row bytes: %u\n\n", swizzle_mode, row_bytes);
          }
          
          uint32_t row = tile_offset / row_bytes;
          printf("--- Tile at offset 0x%lx (Row %u, size %u bytes) ---\n", 
                 (unsigned long)tile_offset, row, req_size);
          
          // Show data pattern for this row (first 16 FP16 values)
          printf("Data pattern (first 16 FP16 values from global): ");
          for (uint32_t i = 0; i < std::min(req_size, 32u) && i < 32; i += 2) {
            uint16_t val = *(uint16_t*)(data_buffer + i);
            printf("%04x ", val);
          }
          printf("\n");
          
          tiles_printed++;
          if (tiles_printed >= 5) printed_first_tile = true;
        }
#endif
        
        // Apply swizzle with detailed per-block debug
        for (uint32_t sub_offset = 0; sub_offset < req_size; sub_offset += SWIZZLE_GRANULARITY) {
          uint32_t sub_size = std::min(SWIZZLE_GRANULARITY, req_size - sub_offset);
          uint64_t logical_offset = tile_offset + sub_offset;
          uint64_t swizzled_offset = apply_tma_swizzle(logical_offset, smem_addr, swizzle_mode, row_bytes);
          
#if TMA_DEBUG_SWIZZLE
          // Debug: Show detailed 16B block swizzle mapping
          if (print_data && sub_offset < 128) {
            uint32_t logical_row = logical_offset / row_bytes;
            uint32_t logical_block = (logical_offset % row_bytes) / 16;
            uint32_t swizzled_row = swizzled_offset / row_bytes;
            uint32_t swizzled_block = (swizzled_offset % row_bytes) / 16;
            
            // Extract first FP16 value from this 16B block
            uint16_t val = *(uint16_t*)(data_buffer + sub_offset);
            
            printf("  16B[%3u]: Logical(R%u,B%u) 0x%03lx -> Swizzled(R%u,B%u) 0x%03lx, Data[0]=0x%04x\n",
                   sub_offset / 16,
                   logical_row, logical_block, (unsigned long)logical_offset,
                   swizzled_row, swizzled_block, (unsigned long)swizzled_offset,
                   val);
          }
#endif
          
          shared_mem->write(smem_addr + swizzled_offset, sub_size, 
                           data_buffer + sub_offset, thread, pI);
        }
#if TMA_DEBUG_SWIZZLE
        if (print_data) {
          printf("\n");
        }
#endif
      } else {
        // No swizzle - write contiguously
        shared_mem->write(smem_addr + tile_offset, req_size, data_buffer, thread, pI);
      }

    } else {
      // Store: shared -> global
      // **REVERSE SWIZZLE**: Read from swizzled smem, write to linear gmem
      
#if TMA_DEBUG_SWIZZLE
      // Debug: Print detailed data layout for first 5 tiles
      static bool printed_first_store_tile = false;
      static int store_tiles_printed = 0;
      bool print_store_data = (!printed_first_store_tile && tile_offset < 640);  // 5 rows * 128B
      
      if (print_store_data && store_tiles_printed < 5) {
        if (store_tiles_printed == 0) {
          printf("\n=== TMA STORE Debug: Detailed Reverse Swizzle Analysis ===\n");
          printf("Global addr: 0x%lx, Smem addr: 0x%x\n", (unsigned long)global_req_addr, smem_addr);
          printf("Swizzle mode: %u, Row bytes: %u\n", swizzle_mode, row_bytes);
          printf("NOTE: REVERSE swizzle - reading FROM swizzled smem, writing TO linear gmem\n\n");
        }
        
        uint32_t row = tile_offset / row_bytes;
        printf("--- Tile at offset 0x%lx (Row %u, size %u bytes) ---\n", 
               (unsigned long)tile_offset, row, req_size);
        
        store_tiles_printed++;
        if (store_tiles_printed >= 5) printed_first_store_tile = true;
      }
#endif
      
      // Apply reverse swizzle at 16-byte granularity when reading from shared memory
      if (swizzle_mode != TMA_SWIZZLE_NONE) {
        for (uint32_t sub_offset = 0; sub_offset < req_size; sub_offset += SWIZZLE_GRANULARITY) {
          uint32_t sub_size = std::min(SWIZZLE_GRANULARITY, req_size - sub_offset);
          uint64_t logical_offset = tile_offset + sub_offset;  // Linear position in gmem
          uint64_t swizzled_offset = apply_tma_swizzle(logical_offset, smem_addr, swizzle_mode, row_bytes);  // Swizzled position in smem
          
          // Read from swizzled smem address
          shared_mem->read(smem_addr + swizzled_offset, sub_size, 
                          data_buffer + sub_offset);
          
#if TMA_DEBUG_SWIZZLE
          // Debug: Show detailed 16B block reverse swizzle mapping
          if (print_store_data && sub_offset < 128) {
            uint32_t logical_row = logical_offset / row_bytes;
            uint32_t logical_block = (logical_offset % row_bytes) / 16;
            uint32_t swizzled_row = swizzled_offset / row_bytes;
            uint32_t swizzled_block = (swizzled_offset % row_bytes) / 16;
            
            // Extract first FP16 value from this 16B block
            uint16_t val = *(uint16_t*)(data_buffer + sub_offset);
            
            printf("  16B[%3u]: Gmem(R%u,B%u) 0x%03lx <- Smem(R%u,B%u) 0x%03lx, Data[0]=0x%04x\n",
                   sub_offset / 16,
                   logical_row, logical_block, (unsigned long)logical_offset,
                   swizzled_row, swizzled_block, (unsigned long)swizzled_offset,
                   val);
          }
#endif
        }
        
#if TMA_DEBUG_SWIZZLE
        // Show data pattern being written (after gathering from swizzled smem)
        if (print_store_data) {
          printf("Data pattern (first 16 FP16 values to write to gmem): ");
          for (uint32_t i = 0; i < std::min(req_size, 32u) && i < 32; i += 2) {
            uint16_t val = *(uint16_t*)(data_buffer + i);
            printf("%04x ", val);
          }
          printf("\n\n");
        }
#endif
      } else {
        // No swizzle - read contiguously
        shared_mem->read(smem_addr + tile_offset, req_size, data_buffer);
      }

      
      // Write contiguous (linear) data to global memory
      global_mem->write(global_req_addr, req_size, data_buffer, thread, pI);
    }

    if (req_size > LOCAL_BUF_SIZE) delete[] data_buffer;
  }
}


//=============================================================================
// TMA AGU Unit (Address Generation Unit)
//=============================================================================

struct tma_agu_state_t {
  // Mode and completion state
  bool is_tensor = false;          // true=tensor mode, false=linear mode
  bool done = true;                // true=iteration complete
  bool is_fill_request = false;    // true=current request is OOB fill

  // Tensor mode state (multi-dimensional tile traversal)
  uint32_t num_dims = 0;           // Number of dimensions (1-5)
  uint32_t elem_size = 0;          // Element size in bytes
  uint32_t box_dim[5] = {0};       // Tile dimensions [d0, d1, d2, d3, d4]
  uint32_t global_dim[5] = {0};    // Global tensor dimensions
  uint32_t start_coords[5] = {0};  // Starting coordinates of this tile
  uint64_t global_strides[5] = {0}; // Strides for each dimension
  uint32_t tile_coords[5] = {0};   // Current position within tile (odometer)
  uint64_t curr_row_addr = 0;      // Base physical address of current row
  uint32_t offset_in_row = 0;      // Byte offset within current row
  uint32_t row_bytes = 0;          // Total bytes in a row (dim0 extent)

  // Linear mode state (simple 1D address range)
  uint64_t linear_addr = 0;        // Current address
  uint32_t linear_remaining = 0;   // Remaining bytes
};

class tma_agu_unit_t {
public:

  void init_tensor(tma_agu_state_t &state,
                   const tensormap_descriptor_t &tm,
                   const uint32_t start_coords[5]) {

    if (!tm.is_valid()) {
      assert(false && "TMA AGU ERROR: Invalid tensormap in init_tensor");
    }

    state = tma_agu_state_t();  // Reset to defaults
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
  }
  
  void init_linear(tma_agu_state_t &state,
                   uint64_t start_addr,
                   uint32_t total_bytes) {
    state = tma_agu_state_t();  // Reset to defaults
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
  bool gen_next_req(tma_agu_state_t &state,
                    uint64_t &out_addr,
                    uint32_t &out_size) {
    if (state.done) return false;

    if (!state.is_tensor) {
      // Linear mode: simple sequential address generation
      if (state.linear_remaining == 0) {
        state.done = true;
        return false;
      }

      state.is_fill_request = false;  // Linear mode never fills
      out_addr = state.linear_addr;
      // Size = min(128B, remaining, bytes to next 128B boundary)
      uint32_t to_boundary = 128 - (state.linear_addr % 128);
      out_size = std::min({(uint32_t)128, state.linear_remaining, to_boundary});

      state.linear_addr += out_size;
      state.linear_remaining -= out_size;

      if (state.linear_remaining == 0) {
        state.done = true;
      }

    } else {
      // Tensor mode: odometer-style multi-dimensional traversal
      // Check if current position is valid (OOB check)
      state.is_fill_request = !is_position_valid(state);

      // Calculate address: current_row_base + offset_within_row
      out_addr = state.curr_row_addr + state.offset_in_row;

      // Calculate size: min(128B, remaining in row)
      uint32_t row_remaining = state.row_bytes - state.offset_in_row;
      out_size = std::min((uint32_t)128, row_remaining);

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
  // Check if current position is within valid tensor bounds (not OOB)
  bool is_position_valid(const tma_agu_state_t &state) const {
    if (!state.is_tensor) return true;
    for (uint32_t d = 0; d < state.num_dims; d++) {
      uint32_t global_coord = state.start_coords[d] + state.tile_coords[d];
      if (global_coord >= state.global_dim[d]) {
        return false;  // Out of bounds
      }
    }
    return true;
  }
  
  // Advance to next row using odometer-style increment of higher dimensions
  void advance_to_next_row(tma_agu_state_t &state) {
    state.offset_in_row = 0;

    // Increment coordinates starting from dimension 1 (dimension 0 is row)
    for (uint32_t d = 1; d < state.num_dims; d++) {
      state.tile_coords[d]++;

      state.curr_row_addr += state.global_strides[d];

      if (state.tile_coords[d] < state.box_dim[d]) {
        // No carry - we're done
        return;
      }

      // Carry to next dimension: reset this dimension
      // We've added stride[d] a total of box_dim[d] times (for coords 0 through box_dim[d]-1, plus the current increment)
      // Subtract the full extent to get back to the base address for this dimension
      state.curr_row_addr -= state.box_dim[d] * state.global_strides[d];
      state.tile_coords[d] = 0;
    }

    // All dimensions wrapped - iteration complete
    state.done = true;
  }
};

//=============================================================================
// TMA Unit (Performance Simulation)
//=============================================================================

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
  
  tma_agu_unit_t m_agu;

  struct tma_transaction_t {
    ptx_thread_info *m_thread = nullptr;
    ptx_instruction *m_inst = nullptr;
    inst_t::tma_static_info_t m_static_info;
    inst_t::tma_dyn_info_t m_dyn_info;
    uint32_t m_bytes_completed = 0;
    tma_agu_state_t agu_state;       // AGU state for this transaction
    
    // Debug counters
    uint32_t m_mf_issued_count = 0;   // Number of mem_fetch requests issued
    uint32_t m_mf_received_count = 0; // Number of mem_fetch responses received

    void reset() {
      m_thread = nullptr;
      m_inst = nullptr;
      m_bytes_completed = 0;
      agu_state = tma_agu_state_t();  // Reset to default state
      m_mf_issued_count = 0;
      m_mf_received_count = 0;
    }
    
    bool is_valid() const { return m_thread != nullptr; }
  };
  std::unordered_map<unsigned, tma_transaction_t>  m_transactions;

  std::list<unsigned> issue_queue;
  std::unordered_map<unsigned, unsigned> m_mf_to_tx;
  
  std::list<mem_fetch *> m_response_fifo;

  // Helper function to finalize a completed transaction
  void finalize_transaction(unsigned tx_uid) {
    auto it = m_transactions.find(tx_uid);
    if (it == m_transactions.end()) return;
    
    auto &tx = it->second;
    auto thread = tx.m_thread;
    unsigned cta_id = thread->get_hw_ctaid();
    unsigned warp_id = thread->get_hw_wid();

    GPPRINTF_TMA(TMA, "[TMA COMPLETE] tx_uid=%u, cta_id=%u, warp_id=%u, mbar=0x%x, issued_mf=%u, received_mf=%u, bytes_completed=%u/%u\n",
           tx_uid, cta_id, warp_id, tx.m_dyn_info.mbar_addr,
           tx.m_mf_issued_count, tx.m_mf_received_count,
           tx.m_bytes_completed, tx.m_dyn_info.size_in_bytes);
    fflush(stdout);

    GPPRINTF_TMA(TMA,
                 "Complete transaction dst=0x%llx, src=0x%llx, "
                 "size_in_bytes=%u, mbar=0x%x, tx_uid=%u\n",
                 tx.m_dyn_info.dst_addr, tx.m_dyn_info.src_addr,
                 tx.m_dyn_info.size_in_bytes, tx.m_dyn_info.mbar_addr, tx_uid);

    m_barriers->complete_tx(cta_id, warp_id, tx.m_dyn_info.mbar_addr,
                            tx.m_dyn_info.size_in_bytes);

    m_transactions.erase(it);

    // Remove all mappings for this transaction
    for (auto map_it = m_mf_to_tx.begin(); map_it != m_mf_to_tx.end(); ) {
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

    // Check if this is a tensor instruction
    bool is_tensor = false;
    for (auto op : pI->get_options()) {
      if (op == TENSOR_OPTION) {
        is_tensor = true;
        break;
      }
    }

    const auto &tma_static_info = pI->get_tma_static_info();

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
        
        // Initialize address generator
        if (tma_static_info.tma_type == inst_t::tma_static_info_t::TMA_TENSOR) {
          
          memory_space *global_mem = thread->get_global_memory();
          tensormap_descriptor_t tensormap;
          global_mem->read(tma_dyn_info.src_addr, TENSORMAP_DESCRIPTOR_SIZE, &tensormap);
          m_agu.init_tensor(tx.agu_state, tensormap, tma_dyn_info.coords);
          
        } else {
          m_agu.init_linear(tx.agu_state, tma_dyn_info.src_addr, tma_dyn_info.size_in_bytes);
        }

        unsigned tx_uid = tma_next_tx_uid.fetch_add(1, std::memory_order_relaxed);
        m_transactions.emplace(tx_uid, tx);
        issue_queue.push_back(tx_uid);

        GPPRINTF_INST_EXEC(TMA, "[TMA START] cta_id=%u, warp_id=%u, lane=%d, tid=%u, tx_uid=%u, dst=0x%llx, src=0x%llx, size=%u, mbar=0x%x\n",
               thread->get_hw_ctaid(), warp_id, laneid, tid, tx_uid,
               tma_dyn_info.dst_addr, tma_dyn_info.src_addr,
               tma_dyn_info.size_in_bytes, tma_dyn_info.mbar_addr);
        fflush(stdout);

        GPPRINTF_INST_EXEC(TMA,
                          "Start transaction dst=0x%llx, src=0x%llx, "
                          "size_in_bytes=%u, mbar=0x%x, tx_uid=%u, tma_type=%d\n",
                          tma_dyn_info.dst_addr, tma_dyn_info.src_addr,
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

    // Process pending TMA responses
    if (!m_response_fifo.empty()) {
      mem_fetch *mf = m_response_fifo.front();
      m_response_fifo.pop_front();
      assert(mf->get_access_type() == TMA_ACC_R);

      mem_fetch *parent_mf = mf->get_original_mf() ? mf->get_original_mf() : mf;
      auto mf_it = m_mf_to_tx.find(parent_mf->get_request_uid());

      assert(mf_it != m_mf_to_tx.end());
      unsigned tx_uid = mf_it->second;
      auto tx_it = m_transactions.find(tx_uid);

      assert(tx_it != m_transactions.end());
      auto &tx = tx_it->second;
      tx.m_mf_received_count++;

      GPPRINTF_TMA(TMA, "TMA response received for mf uid=%u, tx_uid=%u, data_size=%u, response fifo size=%lu\n",
            mf->get_request_uid(),  tx_uid, mf->get_data_size(), m_response_fifo.size());

      // Validate destination space
      if (tx.m_static_info.dst_space == inst_t::tma_static_info_t::TMA_SHARED_CLUSTER) {
        assert(false && "Unsupported TMA destination space: CLUSTER");
      } else if (tx.m_static_info.dst_space != inst_t::tma_static_info_t::TMA_SHARED_CTA) {
        assert(false && "Unrecognized TMA destination space");
      }
      // Count bytes: use min(mf_size, parent_size) to handle L2 sector subdivision
      unsigned mf_size = mf->get_data_size();
      unsigned parent_size = parent_mf->get_data_size();
      unsigned bytes_to_add = (mf_size > parent_size) ? parent_size : mf_size;
      tx.m_bytes_completed += bytes_to_add;

      // Check if transaction is complete
      if (tx.m_bytes_completed >= tx.m_dyn_info.size_in_bytes) finalize_transaction(tx_uid);

      delete mf;
    }

    // issue memory requests using shadow stride accumulation
    if (!issue_queue.empty()) {
      unsigned tx_uid = issue_queue.front();

      auto it = m_transactions.find(tx_uid);
      assert(it != m_transactions.end());
      auto &tx = it->second;

      if (!m_icnt->full(READ_PACKET_SIZE, false)) {
        uint64_t addr;  uint32_t size;

        // Use AGU Unit to generate next address
        if (m_agu.gen_next_req(tx.agu_state, addr, size)) {

          // debug info: Count mem_fetch issued for this transaction
          if (tx.m_mf_issued_count == 0) {
            GPPRINTF_TMA(TMA, "[TMA AGU] tx_uid=%u starting to issue mem_fetch requests\n", tx_uid);
            fflush(stdout);
          }
          tx.m_mf_issued_count++;

          // Check if this is a fill request (OOB region)
          if (tx.agu_state.is_fill_request) {
            // Fill request: just count the bytes, no actual memory transfer needed
            // The functional simulation already handled the zero-fill
            tx.m_bytes_completed += size;

            // Check if transaction is complete
            if (tx.m_bytes_completed >= tx.m_dyn_info.size_in_bytes) finalize_transaction(tx_uid);

          } else {
            // Normal memory request: issue mem_fetch to interconnect
            // Compute byte mask and sector mask for this TMA request
            // TMA requests are up to 128 bytes (MAX_MEMORY_ACCESS_SIZE)
            // Each sector is 32 bytes (SECTOR_SIZE), 4 sectors total (SECTOR_CHUNCK_SIZE)
            mem_access_byte_mask_t byte_mask;
            mem_access_sector_mask_t sector_mask;

            // Set byte mask for the requested bytes
            unsigned start_byte = addr % MAX_MEMORY_ACCESS_SIZE;
            for (unsigned i = 0; i < size; i++) {
              byte_mask.set((start_byte + i) % MAX_MEMORY_ACCESS_SIZE);
            }

            // Set sector mask based on which 32-byte sectors are accessed
            unsigned start_sector = start_byte / SECTOR_SIZE;
            unsigned end_sector = (start_byte + size - 1) / SECTOR_SIZE;
            for (unsigned i = start_sector; i <= end_sector && i < SECTOR_CHUNCK_SIZE; i++) {
              sector_mask.set(i);
            }

            active_mask_t active_mask; // Empty mask for TMA (not warp-based)

            mem_access_t access(TMA_ACC_R, addr, size, false,
                                active_mask, byte_mask, sector_mask,
                                m_shader_ctx->get_gpu()->gpgpu_ctx);

            mem_fetch *mf =
                m_mf_allocator->alloc(access, -1,
                                      m_shader_ctx->get_gpu()->gpu_sim_cycle +
                                      m_shader_ctx->get_gpu()->gpu_tot_sim_cycle);

            m_mf_to_tx.emplace(mf->get_request_uid(), tx_uid);

            m_icnt->push(mf);
          }
        }
      }

      // Remove from issue queue when all requests have been issued
      if (tx.agu_state.done) {
        GPPRINTF_TMA(TMA, "[TMA AGU DONE] tx_uid=%u issued %u mem_fetch requests (total bytes: %u)\n",
               tx_uid, tx.m_mf_issued_count, tx.m_dyn_info.size_in_bytes);
        fflush(stdout);
        issue_queue.pop_front();
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

//=============================================================================
// TensorMap Descriptor
//=============================================================================

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
  // fields.tensorRank is 0-based.
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
// TMA Instruction Functional Simulation
//=============================================================================

void handle_tma_inst(const ptx_instruction *pIin, ptx_thread_info *thread) {
  using namespace flash_gpgpu_sim;

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

  if (!is_commit_group && !is_wait_group && !is_tensor) {
    // This is TMA copy instruction.
    const auto &options = pI->get_options();
    if (options.size() != 3) {
      for (auto op : options) {
        GPPRINTF_INST_EXEC(TMA, "option: %d\n", op);
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
      auto dst_addr = get_operand_u32(thread, pI->dst());
      auto src_addr = get_operand_u64(thread, pI->src1());
      auto size_in_bytes = get_operand_u32(thread, pI->src2());
      auto mbar_addr = get_operand_u32(thread, pI->src3());

      // Check alignment to 16 bytes.
      if (dst_addr % 16 != 0 || src_addr % 16 != 0 || size_in_bytes % 16 != 0) {
        GPPRINTF_INST_EXEC(
            TMA, "unaligned TMA copy dst=0x%x, src=0x%llx, size_in_bytes=%u\n",
            dst_addr, src_addr, size_in_bytes);
        abort();
      }

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
      auto laneid = thread->get_laneid();
      pI->set_tma_dyn_info(laneid, tma_dyn_info);

      // write data into shared memory
      memory_space *global_mem = thread->get_global_memory();
      memory_space *shared_mem = thread->m_shared_mem;

      unsigned char *data_buffer = new unsigned char[size_in_bytes];
      global_mem->read(src_addr, size_in_bytes, data_buffer);
      shared_mem->write(dst_addr, size_in_bytes, data_buffer, thread, pI);

      delete[] data_buffer;

      GPPRINTF_INST_EXEC(TMA,
        "Functional Sim: "
        "TMA shared::cta <- global dst=0x%x, src=0x%llx, "
        "size_in_bytes=%u, mbar=0x%x\n",
        dst_addr, src_addr, size_in_bytes, mbar_addr);

    } else {
      assert(false && "Unsupported TMA copy instruction");
    }

  } else if (is_tensor) {
    // Handle TMA tensor instruction (cp.async.bulk.tensor.Nd)
    // Format: cp.async.bulk.tensor.Nd.dst.src.completion [dst], [tensormap, {coords}], [mbar]
    // Options include: TENSOR_OPTION, DIM_xD_OPTION, dst_space, src_space, completion
    
    const auto &options = pI->get_options();
    if (options.size() != 5) {
      for (auto op : options) {
        GPPRINTF_INST_EXEC(TMA, "  option: %d\n", op);
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

    // Compute instruction dimension
    unsigned inst_dim = compute_inst_dim(dim_option);

    if ((dst_option == CTA_OPTION || dst_option == CLUSTER_OPTION) && 
        src_option == GLOBAL_OPTION && completion_option == TMA_MBAR_COMPLETE_BYTES) {
      // cp.async.bulk.tensor.Nd.shared::cta.global.mbarrier::complete_tx::bytes
      // cp.async.bulk.tensor.Nd.shared::cluster.global.mbarrier::complete_tx::bytes
      // Operands: [dst_shared], [tensormap, {coords...}], [mbar], ignore cache-hint for now
      
      auto dst_addr = get_operand_u32(thread, pI->dst());
      uint64_t tensormap_addr = get_operand_u64(thread, pI->src1());      
      auto mbar_addr = get_operand_u32(thread, pI->src3());

      // Read tensormap descriptor from global memory
      tensormap_descriptor_t tensormap;
      global_mem->read(tensormap_addr, TENSORMAP_DESCRIPTOR_SIZE, &tensormap);
      validate_tensormap(tensormap, inst_dim);
      
      // Calculate transfer size from tensormap
      uint32_t size_in_bytes = tensormap.get_tile_size_bytes();
      assert(size_in_bytes > 0 && "TMA tensor load size cannot be zero");
      
      // Parse coordinates from operands
      const auto &coord_operand = pI->get_operands()[2];
      uint32_t coords[5];
      parse_tensor_coords(thread, coord_operand, coords);

      // Pass info to perf sim
      inst_t::tma_static_info_t tma_static_info{
          .tma_type = inst_t::tma_static_info_t::TMA_TENSOR,
          .dst_space = inst_t::tma_static_info_t::TMA_SHARED_CTA, // treat shared::cluster same as shared::cta
          .src_space = inst_t::tma_static_info_t::TMA_GLOBAL,
      };
      pI->set_tma_static_info(tma_static_info);
      
      inst_t::tma_dyn_info_t tma_dyn_info{
          .dst_addr = dst_addr,
          .src_addr = tensormap_addr,
          .size_in_bytes = size_in_bytes,
          .mbar_addr = mbar_addr,
      };
      for (unsigned i = 0; i < 5; ++i) tma_dyn_info.coords[i] = coords[i];

      auto laneid = thread->get_laneid();
      pI->set_tma_dyn_info(laneid, tma_dyn_info);
      
      // Functional simulation: Copy data from global to shared
      do_tma_transfer(tensormap, coords, shared_mem, global_mem, dst_addr, thread, pI, true);
        
    } else if (dst_option == GLOBAL_OPTION && src_option == CTA_OPTION) {
      // cp.async.bulk.tensor.Nd.global.shared::cta.bulk_group
      // This is a store operation, no mbarrier involved
      // Operands: [tensormap, {coords...}], [src_shared], ignore cache-hint for now
      assert( completion_option == BULK_GROUP_OPTION && "Only bulk_group completion option is supported for cp.async.bulk.tensor.Nd.shared::cta.global" );
      
      // Get tensormap address
      auto tensormap_addr = get_operand_u32(thread, pI->dst());  // dst is tensormap for store
      auto src_addr = get_operand_u32(thread, pI->src2());       // src is shared memory
      
      // Read tensormap descriptor
      tensormap_descriptor_t tensormap;
      global_mem->read(tensormap_addr, TENSORMAP_DESCRIPTOR_SIZE, &tensormap);
      validate_tensormap(tensormap, inst_dim);
      
      // Calculate transfer size from tensormap
      uint32_t size_in_bytes = tensormap.get_tile_size_bytes();
      assert(size_in_bytes > 0 && "TMA tensor store size cannot be zero");
      
      // Parse coordinates from operands
      const auto &coord_operand = pI->get_operands()[1];
      uint32_t coords[5];
      parse_tensor_coords(thread, coord_operand, coords);
      
      GPPRINTF_INST_EXEC(TMA, "TMA tensor store Extracted coordinates: [%u, %u, %u, %u, %u]\n", 
                coords[0], coords[1], coords[2], coords[3], coords[4]);
      
      // Pass info to perf sim
      inst_t::tma_static_info_t tma_static_info{
          .tma_type = inst_t::tma_static_info_t::TMA_TENSOR,
          .dst_space = inst_t::tma_static_info_t::TMA_SHARED_CTA, // treat shared::cluster same as shared::cta
          .src_space = inst_t::tma_static_info_t::TMA_GLOBAL,
      };
      pI->set_tma_static_info(tma_static_info);
      
      inst_t::tma_dyn_info_t tma_dyn_info{
          .dst_addr = tensormap_addr,
          .src_addr = src_addr,
          .size_in_bytes = size_in_bytes,
      };
      for (unsigned i = 0; i < 5; ++i) tma_dyn_info.coords[i] = coords[i];

      auto laneid = thread->get_laneid();
      pI->set_tma_dyn_info(laneid, tma_dyn_info);
      
      // Functional simulation: Copy data from shared to global
      do_tma_transfer(tensormap, coords, shared_mem, global_mem, src_addr, thread, pI, false);

      uint64_t base_dst_addr = tensormap.calculate_src_addr(coords);
      GPPRINTF_INST_EXEC(TMA,
        "Functional Sim: TMA tensor store dst=0x%llx, src=0x%x, "
        "size=%u, tensormap=0x%x\n",
        (unsigned long long)base_dst_addr, src_addr, size_in_bytes, tensormap_addr);
        
    } else {
      GPPRINTF_INST_EXEC(TMA, 
        "[STUB] Unsupported cp.async.bulk.tensor variant%s\n", "");
    }

  } else if (is_commit_group) {
    // Handle TMA commit group instruction (cp.async.bulk.commit_group).
    // TODO: Implement commit group - for now treat as NOP
    GPPRINTF_INST_EXEC(TMA, "[STUB] cp.async.bulk.commit_group not implemented (treated as NOP)%s\n", "");

  } else if (is_wait_group) {
    // Handle TMA wait group instruction (cp.async.bulk.wait_group).
    // TODO: Implement wait group - for now treat as NOP (assume all transfers complete immediately)
    GPPRINTF_INST_EXEC(TMA, "[STUB] cp.async.bulk.wait_group not implemented (treated as NOP)%s\n", "");

  } else {
    GPPRINTF_INST_EXEC(TMA, "Unrecognized TMA instruction%s\n", "");
    pI->print_insn();
    assert(false && "Unrecognized TMA instruction");
  }
}

//=============================================================================
// handle_tensormap_inst - Process tensormap PTX instructions
//=============================================================================

void handle_tensormap_inst(const ptx_instruction *pI, ptx_thread_info *thread) {
  using namespace flash_gpgpu_sim;
  
  const auto &options = pI->get_options();
  
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
  bool is_swizzle_atomicity = false;
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
      case SWIZZLE_ATOMICITY_OPTION: is_swizzle_atomicity = true; break;
      case FILL_MODE_OPTION: is_fill_mode = true; break;
      default: break;
    }
  }
  
  memory_space *shared_mem = thread->m_shared_mem;
  
  if (is_replace && is_tile) {
    // tensormap.replace.tile.<field> [dst], value
    // or tensormap.replace.tile.<field> [dst], dim_idx, value
    
    // Get destination address (tensormap address in shared memory)
    uint32_t tensormap_addr = get_operand_u32(thread, pI->dst());
    
    // Read existing descriptor, modify field, write back
    tensormap_descriptor_t desc = tensormap_descriptor_t::read_from_shared(shared_mem, tensormap_addr);
    
    if (is_global_address) {
      // tensormap.replace.tile.global_address [dst], value
      uint64_t value = get_operand_u64(thread, pI->src1());
      desc.fields.globalAddress = value;
      GPPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.global_address [0x%x] = 0x%llx\n", 
                        tensormap_addr, (unsigned long long)value);
                        
    } else if (is_rank) {
      // tensormap.replace.tile.rank [dst], value
      uint32_t value = get_operand_u32(thread, pI->src1());
      desc.fields.tensorRank = value;
      GPPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.rank [0x%x] = %u\n", tensormap_addr, value);
      
    } else if (is_box_dim) {
      // tensormap.replace.tile.box_dim [dst], dim_idx, value
      uint32_t dim_idx = get_operand_u32(thread, pI->src1());
      uint32_t value = get_operand_u32(thread, pI->src2());
      if (dim_idx < 5) desc.fields.boxDim[dim_idx] = value;
      GPPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.box_dim [0x%x], dim=%u, value=%u\n", 
                        tensormap_addr, dim_idx, value);
      
    } else if (is_global_dim) {
      // tensormap.replace.tile.global_dim [dst], dim_idx, value
      uint32_t dim_idx = get_operand_u32(thread, pI->src1());
      uint32_t value = get_operand_u32(thread, pI->src2());
      if (dim_idx < 5) desc.fields.globalDim[dim_idx] = value;
      GPPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.global_dim [0x%x], dim=%u, value=%u\n", 
                        tensormap_addr, dim_idx, value);
      
    } else if (is_global_stride) {
      // tensormap.replace.tile.global_stride [dst], dim_idx, value
      uint32_t dim_idx = get_operand_u32(thread, pI->src1());
      uint64_t value = get_operand_u64(thread, pI->src2());
      if (dim_idx < 5) desc.fields.globalStrides[dim_idx] = value;
      GPPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.global_stride [0x%x], dim=%u, value=%llu\n", 
                        tensormap_addr, dim_idx, (unsigned long long)value);
      
    } else if (is_element_stride) {
      // tensormap.replace.tile.element_stride [dst], dim_idx, value
      uint32_t dim_idx = get_operand_u32(thread, pI->src1());
      uint32_t value = get_operand_u32(thread, pI->src2());
      if (dim_idx < 5) desc.fields.elementStrides[dim_idx] = value;
      GPPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.element_stride [0x%x], dim=%u, value=%u\n", 
                        tensormap_addr, dim_idx, value);
      
    } else if (is_elemtype) {
      // tensormap.replace.tile.elemtype [dst], value
      uint32_t value = get_operand_u32(thread, pI->src1());
      desc.fields.tensorDataType = value;
      GPPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.elemtype [0x%x] = %u\n", tensormap_addr, value);
      
    } else if (is_interleave_layout) {
      // tensormap.replace.tile.interleave_layout [dst], value
      uint32_t value = get_operand_u32(thread, pI->src1());
      desc.fields.interleave = value;
      GPPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.interleave_layout [0x%x] = %u\n", tensormap_addr, value);
      assert( value == 0 && "Only TMA_INTERLEAVE_NONE (0) is currently supported");
      
    } else if (is_swizzle_mode) {
      // tensormap.replace.tile.swizzle_mode [dst], value
      uint32_t value = get_operand_u32(thread, pI->src1());
      desc.fields.swizzle = value;
      GPPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.swizzle_mode [0x%x] = %u\n", tensormap_addr, value);
      
    } else if (is_swizzle_atomicity) {
      // tensormap.replace.tile.swizzle_atomicity [dst], value
      uint32_t value = get_operand_u32(thread, pI->src1());
      assert(value == 0 && "Only 16B swizzle atomicity (0x0) is currently supported");
      GPPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.swizzle_atomicity [0x%x] = %u (16B)\n", tensormap_addr, value);
      
    } else if (is_fill_mode) {
      // tensormap.replace.tile.fill_mode [dst], value
      uint32_t value = get_operand_u32(thread, pI->src1());
      desc.fields.oobFill = value;
      GPPRINTF_INST_EXEC(TMA, "tensormap.replace.tile.fill_mode [0x%x] = %u\n", tensormap_addr, value);
      
    } else {
      GPPRINTF_INST_EXEC(TMA, "[STUB] Unrecognized tensormap.replace.tile field%s\n", "");
    }
    
    // Write modified descriptor back to shared memory
    desc.write_to_shared(shared_mem, tensormap_addr, thread, pI);
    
  } else if (is_cp_fenceproxy) {
    // tensormap.cp_fenceproxy.global.shared::cta... [dst_global], [src_shared], size
    // Copy tensormap from shared to global memory with fence
    
    // TODO: Add option check
    uint64_t dst_addr = get_operand_u64(thread, pI->dst());
    uint32_t src_addr = get_operand_u32(thread, pI->src1());
    uint32_t size_in_bytes = get_operand_u32(thread, pI->src2());

    memory_space *global_mem = thread->get_global_memory();;
    
    tensormap_descriptor_t desc = tensormap_descriptor_t::read_from_shared(shared_mem, src_addr);
    global_mem->write(dst_addr, size_in_bytes, desc.raw_bytes, thread, pI);
    
  } else {
    GPPRINTF_INST_EXEC(TMA, "[STUB] Unrecognized tensormap instruction variant%s\n", "");
  }
}