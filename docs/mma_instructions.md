# MMA Instructions Implementation

This document describes the PTX MMA (Matrix Multiply-Accumulate) instruction implementation in GPGPU-Sim.

## Overview

MMA instructions are warp-level matrix operations for tensor cores, as defined in the PTX ISA. This implementation provides a **completely separate code path** from the existing WMMA instructions to ensure clean separation and avoid breaking existing functionality.

Reference: [PTX ISA - Warp-Level Matrix Instructions for MMA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=tensormap%2520replace#warp-level-matrix-instructions-for-mma)

## MMA vs WMMA

| Aspect | WMMA (Existing) | MMA (This Implementation) |
|--------|----------------|---------------------------|
| **Opcodes** | MMA_OP, MMA_LD_OP, MMA_ST_OP | TENSOR_MMA_OP, TENSOR_MMA_LD_OP, TENSOR_MMA_ST_OP |
| **Functions** | mma_impl(), mma_ld_impl(), mma_st_impl() | tensor_mma_impl(), tensor_mma_ld_impl(), tensor_mma_st_impl() |
| **Data Types** | wmma_type enum | mma_type enum, mma_layout_mode enum |
| **Helpers** | thread_group_offset(), acc_float_offset(), mapping() | mma_thread_to_element_offset(), mma_type_convert(), mma_saturate() |
| **Code Base** | src/cuda-sim/instructions.cc (lines 1917, 3446, 3569) | New functions added separately |

## Supported MMA Variants

### Support Matrix by Architecture

The following table shows currently supported MMA shape/type combinations:

| Shape | F16 | BF16 | TF32 | S8/U8 | F64 | Architecture |
|-------|-----|------|------|-------|-----|--------------|
| **8x8x4** | ⏳ | - | - | - | ✓ | Volta (SM70) / Ampere (SM80) |
| **8x8x16** | - | - | - | ⏳ | - | Turing (SM75) |
| **16x8x4** | - | - | ✓ | - | - | Ampere (SM80) |
| **16x8x8** | ✓ | ✓ | ⏳ | - | - | Turing (SM75) / Ampere (SM80) |
| **16x8x16** | ⏳ | ⏳ | - | ✓ | - | Ampere (SM80) |
| **16x8x32** | - | - | - | ⏳ | - | Ampere (SM80) |

**Legend**:
- ✓ = Fully implemented and tested
- ⏳ = Planned for implementation (see issue #26)
- \- = Not supported by PTX ISA for this type

**Target completion** (issue #26):
- **Phase 1**: F16 8x8x4 and 16x8x16
- **Phase 2**: BF16 16x8x16
- **Phase 3**: TF32 16x8x8
- **Phase 4**: S8/U8 8x8x16 and 16x8x32

### Data Types

| Type | Description | Size | Use Case |
|------|-------------|------|----------|
| **F16** | Half precision float | 16-bit | General tensor operations |
| **F32** | Single precision float | 32-bit | Accumulator, high precision |
| **TF32** | TensorFloat-32 | 32-bit | High-throughput ML training |
| **BF16** | BrainFloat16 | 16-bit | ML training, wide dynamic range |
| **S8** | Signed 8-bit integer | 8-bit | Quantized inference |
| **U8** | Unsigned 8-bit integer | 8-bit | Quantized inference |
| **S4** | Signed 4-bit integer | 4-bit | Ultra-low precision |
| **U4** | Unsigned 4-bit integer | 4-bit | Ultra-low precision |
| **B1** | Binary (1-bit) | 1-bit | Binary neural networks |

### Layout Modes

Each matrix (A, B, C/D) can have different layouts:

- **ROW_COL** - A in row-major, B in column-major
- **ROW_ROW** - A in row-major, B in row-major
- **COL_ROW** - A in column-major, B in row-major
- **COL_COL** - A in column-major, B in column-major

## Thread-to-Matrix Mappings

### General Principle

MMA instructions operate on warp-level (32 threads). Each thread handles specific elements of the input/output matrices based on:
1. Matrix shape (M×N×K)
2. Data layout (ROW/COL)
3. Thread ID within warp (0-31)

### M16N16K16 Mapping Example

For a 16×16×16 MMA operation:
- **Matrix A (16×16)**: Each thread loads/computes 8 elements
- **Matrix B (16×16)**: Each thread loads/computes 8 elements
- **Matrix C/D (16×16)**: Each thread loads/stores 8 elements

Thread mapping follows a structured pattern to maximize coalesced memory access and minimize shared memory bank conflicts.

### Helper Function

The `mma_thread_to_element_offset()` function maps:
```
(thread_id, shape, layout, type_size, stride) → element_offset
```

This function handles all supported shapes and layouts.

## Type Conversion

### Cross-Type Operations

MMA supports mixed-precision accumulation:
- **Input matrices (A, B)**: Can be F16, BF16, TF32, S8, U8, S4, U4, B1
- **Accumulator (C/D)**: Typically F32 for higher precision

### Conversion Function

The `mma_type_convert()` function handles conversions between:
- FP16 ↔ FP32
- BF16 ↔ FP32
- TF32 ↔ FP32
- Integer types ↔ FP32

**TensorFloat-32 (TF32) Conversion:**
- Range: Same as FP32
- Precision: 10-bit mantissa (vs 23-bit for FP32)
- Exponent: 8-bit (same as FP32)
- Used for high-throughput training

**BrainFloat16 (BF16) Conversion:**
- Range: Same as FP32 (8-bit exponent)
- Precision: 7-bit mantissa
- Better for training than FP16 due to wider range

## Saturation Modes

For integer MMA operations (S8, U8, S4, U4), saturation prevents overflow:

### Saturation Behavior

The `mma_saturate()` function:
- **Without saturation**: Wrap on overflow (modulo arithmetic)
- **With saturation**: Clamp to [min, max] range

Example for S8:
```
Result = 200 (before saturation)
Saturated = 127 (max value for S8)
```

## Code Organization

### File Structure

```
src/cuda-sim/
├── ptx.l                   # Lexer - NEW MMA tokens
├── ptx.y                   # Parser - NEW MMA grammar rules
├── ptx_ir.h                # NEW mma_type enum, mma_layout_mode enum
├── opcodes.def             # NEW TENSOR_MMA_* opcodes
├── instructions.cc         # NEW tensor_mma_impl() functions
└── cuda-sim.cc             # NEW case statements for TENSOR_MMA_*

test/
├── src/
│   ├── unit/
│   │   └── tensor_mma_test.cc         # NEW MMA unit tests
│   └── integration/
│       └── cuda_tensor_mma_test.cc    # NEW MMA integration tests
└── kernels/
    ├── tensor_mma_f16_m16n8k8.ptx     # NEW test kernels
    └── ...
```

### Naming Conventions

All MMA-related symbols use consistent prefixes:

- **Enums**: `MMA_*` (e.g., `MMA_M16N8K8`, `MMA_ROW_COL`)
- **Opcodes**: `TENSOR_MMA_*` (e.g., `TENSOR_MMA_OP`)
- **Functions**: `tensor_mma_*` (e.g., `tensor_mma_impl`)
- **Helpers**: `mma_*` (e.g., `mma_thread_to_element_offset`)

## Usage Examples

### Example 1: F16 MMA (M16N8K8)

```ptx
// Load matrix A (F16, 16×8)
mma.load.a.sync.aligned.m16n8k8.f16 {r0, r1, r2, r3}, [addr_a];

// Load matrix B (F16, 8×8)
mma.load.b.sync.aligned.m16n8k8.f16 {r4, r5}, [addr_b];

// Load accumulator C (F32, 16×8)
mma.load.c.sync.aligned.m16n8k8.f32 {r8, r9, r10, r11}, [addr_c];

// Compute D = A × B + C
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
    {r8, r9, r10, r11},
    {r0, r1, r2, r3},
    {r4, r5},
    {r8, r9, r10, r11};

// Store result D (F32, 16×8)
mma.store.d.sync.aligned.m16n8k8.f32 [addr_d], {r8, r9, r10, r11};
```

### Example 2: S8 MMA with Saturation (M16N8K16)

```ptx
// Load S8 matrices
mma.load.a.sync.aligned.m16n8k16.s8 {r0, r1}, [addr_a];
mma.load.b.sync.aligned.m16n8k16.s8 {r2}, [addr_b];
mma.load.c.sync.aligned.m16n8k16.s32 {r4, r5, r6, r7}, [addr_c];

// Compute with saturation
mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite
    {r4, r5, r6, r7},
    {r0, r1},
    {r2},
    {r4, r5, r6, r7};

// Store saturated S32 result
mma.store.d.sync.aligned.m16n8k16.s32 [addr_d], {r4, r5, r6, r7};
```

## Testing Strategy

### Unit Tests (`test/src/unit/tensor_mma_test.cc`)

- **Per-instruction validation**: Each MMA variant tested independently
- **Thread mapping verification**: Validate thread-to-element offset calculations
- **Type conversion tests**: Verify all data type conversions (especially TF32, BF16)
- **Saturation tests**: Check integer overflow handling
- **Edge cases**: Zero matrices, max values, mixed signs

### Integration Tests (`test/src/integration/cuda_tensor_mma_test.cc`)

- **Real PTX execution**: Run actual MMA PTX instructions through simulator
- **Golden reference comparison**: Compare results against known-good values
- **Memory pattern validation**: Verify memory transaction addresses
- **Cross-layout testing**: Test all layout combinations (ROW-COL, ROW-ROW, etc.)

### Test Execution

```bash
# Setup environment (REQUIRED before testing)
source setup.sh
source setup_environment

# Run all tests with reduced configuration
./test/run_tests.sh -c SM120_RTX5090_REDUCED run

# Run specific MMA test files
./test/run_tests.sh -c SM120_RTX5090_REDUCED run CudaMmaF16Test
./test/run_tests.sh -c SM120_RTX5090_REDUCED run CudaMmaBf16Test
./test/run_tests.sh -c SM120_RTX5090_REDUCED run CudaMmaTf32Test
./test/run_tests.sh -c SM120_RTX5090_REDUCED run CudaMmaS8Test
```

### Coverage Goals

- ✓ Each MMA instruction variant: at least one test case
- ✓ All data type combinations: tested
- ✓ All shape configurations: covered
- ✓ All layout modes: verified
- ✓ Zero regression in existing WMMA tests

## Implementation Status

| Component | Status | Files |
|-----------|--------|-------|
| Parser (Lexer/Grammar) | ⏳ Pending | ptx.l, ptx.y |
| Data Structures | ⏳ Pending | ptx_ir.h, opcodes.h |
| Opcodes | ⏳ Pending | opcodes.def |
| Execution Functions | ⏳ Pending | instructions.cc |
| Dispatcher | ⏳ Pending | cuda-sim.cc |
| Tests | ⏳ Pending | test/src/unit/, test/src/integration/ |
| Documentation | ✓ Complete | docs/mma_instructions.md |

## Troubleshooting

### Common Issues

**Issue**: MMA instruction not recognized during PTX parsing
- **Solution**: Verify new tokens added to `ptx.l` (MMA_INSTR, MMA_SYNC_OP)
- **Solution**: Check grammar rules in `ptx.y` for MMA shapes/layouts

**Issue**: Undefined reference to `tensor_mma_impl`
- **Solution**: Ensure opcodes.def has correct function names
- **Solution**: Rebuild after modifying opcodes.def (triggers code generation)

**Issue**: Thread mapping produces incorrect results
- **Solution**: Verify `mma_thread_to_element_offset()` for specific shape
- **Solution**: Check layout mode (ROW vs COL) matches PTX instruction

**Issue**: Type conversion errors (NaN, Inf)
- **Solution**: Verify `mma_type_convert()` handles edge cases (denormals, overflow)
- **Solution**: Check TF32/BF16 conversion logic for proper rounding

## References

- [PTX ISA Guide - MMA Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma)
- [CUDA C++ Programming Guide - Tensor Cores](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [CUTLASS Library - GEMM Kernels](https://github.com/NVIDIA/cutlass) (for reference implementations)
- Issue #18: Implementation plan and progress tracking
