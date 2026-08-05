# MMA Implementation

Per-data-type implementation files for PTX MMA (Matrix Multiply-Accumulate) instructions.

## Purpose

This subdirectory contains the split implementation of MMA tensor core operations, organized by data type for improved maintainability and clarity. Each file focuses on a specific data type's implementation, while shared helpers and the main dispatcher remain in `tensor_mma.cc`.

## Files

### Interface and Documentation

- **`tensor_mma.h`** - Public interface for all MMA operations
  - Function declarations for MMA compute, load, and store operations
  - Enum definitions for shapes (M16N8K8, M16N8K4, etc.) and layouts (ROW_COL, etc.)
  - Helper function prototypes for type conversion and saturation

- **`tensor_mma.md`** - Interface documentation
  - Detailed API documentation for all public functions
  - Fragment distribution patterns for each MMA shape
  - Type conversion specifications
  - Usage examples and integration notes

### Implementation Files

- **`tensor_mma.cc`** (520 LOC) - Main dispatcher and shared helpers
  - `tensor_mma_impl()` - Main dispatcher routing to type-specific implementations
  - `tensor_mma_ld_impl()` - Load matrix fragments from memory
  - `tensor_mma_st_impl()` - Store matrix fragments to memory
  - Shared helper functions:
    - Type conversions: `mma_f16_to_f32()`, `mma_bf16_to_f32()`, `mma_tf32_round()`
    - Saturation: `mma_saturate_s8()`, `mma_saturate_u8()`, `mma_saturate_s4()`, `mma_saturate_u4()`
    - Thread mapping: `mma_thread_to_element_offset()`

- **`mma_f16.cc`** (382 LOC) - F16/BF16 floating-point implementations
  - `tensor_mma_f16_impl()` - F16/BF16 MMA for M16N8K8/K16 shapes
  - `tensor_mma_f16_m8n8k4_impl()` - Specialized F16 M8N8K4 implementation (4-computation architecture)

- **`mma_tf32.cc`** (178 LOC) - TF32 floating-point implementation
  - `tensor_mma_tf32_impl()` - TensorFloat-32 MMA for M16N8K4/K8 shapes

- **`mma_s8.cc`** (399 LOC) - S8/U8 integer implementations
  - `tensor_mma_s8_impl()` - Signed/unsigned 8-bit integer MMA for M16N8K16/K32 shapes
  - `tensor_mma_s8_m8n8k16_impl()` - Specialized S8/U8 M8N8K16 implementation

## Integration

This implementation integrates with the GPGPU-Sim PTX parser and execution pipeline:

1. **PTX Parser** (`src/cuda-sim/`) recognizes `mma.sync.aligned` instructions
2. **Opcode Dispatcher** routes `TENSOR_MMA_OP` to `flash_gpgpu_sim::tensor_mma_impl()`
3. **Type Dispatcher** (`tensor_mma.cc`) routes to appropriate type-specific function
4. **Type Implementation** (this directory) performs functional simulation

## Testing

Integration tests are located in `test/src/integration/mma/`:
- F16 tests: `cuda_mma_f16_test.cc`
- BF16 tests: `cuda_mma_bf16_test.cc`
- TF32 tests: `cuda_mma_tf32_test.cc`
- S8/U8 tests: `cuda_mma_s8_test.cc`

Run MMA tests with:
```bash
./test/run_tests.sh -c SM120_RTX5090 run "*MMA*"
```

## References

- Parent directory documentation: `../README.md`
- Integration tests: `../../../../test/src/integration/mma/README.md`
- PTX ISA: [MMA Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma)
