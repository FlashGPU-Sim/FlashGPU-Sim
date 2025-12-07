#ifndef FLASH_GPGPU_SIM_TMA_H
#define FLASH_GPGPU_SIM_TMA_H

#include <memory>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

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
             mem_fetch_interface *icnt, shader_core_mem_fetch_allocator *mf_allocator);
  ~tma_unit_t();
  
  void warp_reaches_tma(unsigned cta_id, unsigned warp_id, warp_inst_t *inst);
  void cycle();

  void fill(mem_fetch *mf);
  bool response_buffer_full() const;

private:
  std::unique_ptr<tma_unit_impl_t> m_impl;
};

// -----------------------------------------------------------------------------
// TensorMap descriptor (aligned to 128B, mirrors CUDA Driver API layout)
// -----------------------------------------------------------------------------

#define TENSORMAP_DESCRIPTOR_SIZE 128u

// Element data type encoding
#define TMA_DTYPE_U8    0u
#define TMA_DTYPE_U16   1u
#define TMA_DTYPE_U32   2u
#define TMA_DTYPE_U64   4u
#define TMA_DTYPE_F16   6u
#define TMA_DTYPE_F32   7u
#define TMA_DTYPE_F64   9u
#define TMA_DTYPE_BF16  10u

// Interleave layout modes (in bytes)
#define TMA_INTERLEAVE_NONE   0u
#define TMA_INTERLEAVE_16B    1u
#define TMA_INTERLEAVE_32B    2u

// Swizzle modes (in bytes)
#define TMA_SWIZZLE_NONE      0u
#define TMA_SWIZZLE_32B       1u
#define TMA_SWIZZLE_64B       2u
#define TMA_SWIZZLE_128B      3u
#define TMA_SWIZZLE_96B       4u

// Out-of-bound fill modes
#define TMA_OOB_ZERO          0u
#define TMA_OOB_NAN           1u

typedef union __attribute__((aligned(128))) tensormap_descriptor_t {
  // 1) Raw views
  uint8_t  raw_bytes[128];
  uint64_t raw_u64[16];

  // 2) Structured view (packed to maintain exact offsets)
  struct __attribute__((packed)) {
    // Core addressing
    uint64_t globalAddress;          // [0-7]

    // Shape
    uint32_t tensorRank;             // [8-11]
    uint32_t boxDim[5];              // [12-31]
    uint32_t globalDim[5];           // [32-51]

    // Strides
    uint64_t globalStrides[5];       // [52-91]
    uint32_t elementStrides[5];      // [92-111]

    // Format / control
    uint32_t tensorDataType;         // [112-115]
    uint32_t interleave;             // [116-119]
    uint32_t swizzle;                // [120-123]
    uint32_t oobFill;                // [124-127]
  } fields;

  // Helpers
  uint32_t get_element_size() const;
  uint32_t get_tile_size_bytes() const;
  uint64_t calculate_src_addr(const uint32_t coords[5]) const;
  uint32_t num_dims() const { return fields.tensorRank + 1u; }
  bool is_valid() const { return fields.tensorRank <= 4 && fields.globalAddress != 0; }
  void print() const;

  static tensormap_descriptor_t read_from_shared(memory_space *shared_mem, uint32_t addr);
  void write_to_shared(memory_space *shared_mem, uint32_t addr, ptx_thread_info *thd, const ptx_instruction *pI) const;
  
} tensormap_descriptor_t;

// Generate memory fetch requests for TMA tensor operations
// start_coords: starting coordinate for each dimension [x, y, z, w, v]
// Returns: vector of (physical_address, size_in_bytes) pairs for memory fetches
std::vector<std::pair<uint64_t, uint32_t>> 
generate_tma_requests(const tensormap_descriptor_t& tensormap, 
                      const uint32_t start_coords[5]);

} // namespace flash_gpgpu_sim

void handle_tma_inst(const ptx_instruction *pIin, ptx_thread_info *thread);
void handle_tensormap_inst(const ptx_instruction *pI, ptx_thread_info *thread);

#endif