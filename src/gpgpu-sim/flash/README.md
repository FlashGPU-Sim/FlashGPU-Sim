# Flash GPGPU-Sim Extensions

This folder contains extended functionality for GPGPU-Sim, including tensor core implementations and advanced GPU features.

## Purpose

The `flash/` namespace provides:
- **Tensor Core Support**: PTX MMA (Matrix Multiply-Accumulate) instruction implementations
- **Advanced GPU Features**: Implementations for modern GPU architectures (SM75+)
- **Isolated Extensions**: Clear separation from core GPGPU-Sim to avoid breaking existing functionality

## Key Files

### Tensor Core MMA Implementation

Located in `flash/mma/` subdirectory:

- **mma/tensor_mma.h**: Public interface for MMA instruction support
  - Function declarations for MMA compute, load, and store operations
  - Enum definitions for shapes (M16N8K8, M16N8K4, etc.) and layouts (ROW_COL, etc.)
  - Helper function prototypes for type conversion and saturation

- **mma/tensor_mma.cc**: Main dispatcher and shared helpers
  - Main dispatcher: `tensor_mma_impl()` - routes to type-specific implementations
  - Shared helper functions: Type conversions (F16, BF16, TF32), saturation (S8, U8, S4, U4)
  - Load/store implementations: `tensor_mma_ld_impl()`, `tensor_mma_st_impl()`

- **mma/mma_f16.cc**: F16/BF16 floating-point MMA implementations
  - `tensor_mma_f16_impl()` - F16/BF16 M16N8K8 implementation
  - `tensor_mma_f16_m8n8k4_impl()` - F16 M8N8K4 implementation

- **mma/mma_tf32.cc**: TF32 floating-point MMA implementation
  - `tensor_mma_tf32_impl()` - TF32 M16N8K4 and M16N8K8 implementations

- **mma/mma_s8.cc**: S8/U8 integer MMA implementations
  - `tensor_mma_s8_impl()` - S8/U8 M16N8K16/K32 implementations
  - `tensor_mma_s8_m8n8k16_impl()` - S8/U8 M8N8K16 implementation

- **mma/tensor_mma.md**: Interface documentation (moved from flash/)
  - Detailed API documentation for all public functions
  - Fragment distribution patterns for each MMA shape
  - Type conversion specifications
  - Usage examples and integration notes

## Architecture Integration

### Relationship to Core GPGPU-Sim

The flash namespace integrates with GPGPU-Sim through:

1. **PTX Parser Integration** (`src/cuda-sim/`):
   - `ptx.l`: Lexer recognizes `mma.sync` tokens
   - `ptx.y`: Parser handles MMA instruction grammar
   - `ptx_ir.h`: PTX IR stores MMA shape/layout metadata
   - `opcodes.def`: Defines TENSOR_MMA_OP, TENSOR_MMA_LD_OP, TENSOR_MMA_ST_OP

2. **Opcode Dispatch** (`src/cuda-sim/cuda-sim.cc`):
   - TENSOR_MMA_* opcodes route to flash::tensor_mma_impl()
   - Separate from existing WMMA (MMA_OP) to avoid conflicts

3. **Execution** (`src/cuda-sim/instructions.cc`):
   - `using flash_gpgpu_sim::tensor_mma_impl` imports implementation from `flash/mma/tensor_mma.h`
   - Warp-level collective execution across 32 threads

### Separation from WMMA

**IMPORTANT**: MMA and WMMA are separate instruction families:

| Aspect | WMMA (Existing) | MMA (This Implementation) |
|--------|----------------|---------------------------|
| PTX Instruction | `wmma.mma.sync` | `mma.sync.aligned` |
| Opcodes | MMA_OP | TENSOR_MMA_OP |
| Implementation | instructions.cc:mma_impl() | flash/mma/tensor_mma.cc:tensor_mma_impl() |
| Namespace | Global | flash_gpgpu_sim |

This separation ensures:
- No breaking changes to existing WMMA code
- Clear distinction between instruction families
- Independent evolution of MMA and WMMA features

## Supported Features

### MMA Shapes (M×N×K)
- M16N8K8 (SM75+): F16, BF16, TF32
- M16N8K4 (SM80+): TF32
- M16N8K16 (SM80+): F16, BF16, S8, U8
- M16N8K32 (SM80+): S8, U8
- M16N8K64 (SM80+): S4, U4
- M8N8K4 (SM80+): F64

### Data Types
- **Floating-point**: F16, BF16, TF32, F32, F64
- **Integer**: S8, U8, S4, U4, B1

### Layout Modes
- ROW_COL, ROW_ROW, COL_ROW, COL_COL

## Testing

Integration tests are located in `test/src/mma/`:
- `mma_f16_test.cu` - F16 M16N8K8 tests
- `mma_bf16_test.cu` - BF16 M16N8K8 tests
- `mma_tf32_test.cu` - TF32 M16N8K4 tests
- `mma_s8_test.cu` - S8 M16N8K16 tests

See `test/src/mma/README.md` for test execution instructions.

## Code Formatting

The Flash module uses LLVM-style formatting enforced by pre-commit hooks.

**Quick setup:**
1. Ensure `clang-format` is installed
2. Format all files: `./scripts/format-flash.sh`
3. Commits will be blocked if staged Flash files are not formatted

See `docs/workflows/code-formatting.md` for details.

## References

- PTX ISA: [MMA Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma)
- Implementation overview: [`mma/README.md`](mma/README.md)
- Interface documentation: [`mma/tensor_mma.md`](mma/tensor_mma.md)
- Issue tracking: Issue #18
