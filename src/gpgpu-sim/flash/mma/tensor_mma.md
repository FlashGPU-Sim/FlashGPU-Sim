# Tensor MMA Implementation Interface

This document describes the C++ implementation interface for PTX MMA (Matrix Multiply-Accumulate) instructions in GPGPU-Sim. For PTX instruction semantics and formats, see `docs/mma_instructions.md`.

## External Interface

### Main Instruction Implementations

#### `tensor_mma_impl()`

**Signature**:
```cpp
void tensor_mma_impl(const ptx_instruction *pI, core_t *core, warp_inst_t inst)
```

**Purpose**: Execute PTX MMA instruction performing warp-level matrix multiply-accumulate operation (D = A × B + C).

**Parameters**:
- `pI`: PTX instruction descriptor containing:
  - MMA shape (M×N×K dimensions)
  - Data types for A, B, C/D matrices
  - Layout modes (row-major/column-major)
- `core`: GPU core context providing access to thread/warp state and register file
- `inst`: Warp instruction containing source and destination register operands

**Behavior**:
- Performs distributed matrix multiplication across 32 threads in a warp
- Each thread computes a subset of output matrix elements
- Supports multiple shapes: M16N8K8, M16N8K16, M16N16K16, M8N8K4, M8N8K32, M8N8K128
- Handles data types: F16 (half), BF16 (bfloat16), TF32 (tensorfloat-32), S8 (signed int8), U8 (unsigned int8)
- Layout modes: ROW-COL, ROW-ROW, COL-ROW, COL-COL

**Thread Distribution**: Uses `mma_thread_to_element_offset()` to map each thread ID to its assigned matrix elements based on shape and layout.

**Error Handling**: Assumes valid instruction from PTX parser; no runtime validation.

---

#### `tensor_mma_ld_impl()`

**Signature**:
```cpp
void tensor_mma_ld_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst)
```

**Purpose**: Load matrix fragments from memory into registers for MMA computation.

**Parameters**:
- `pI`: PTX instruction with memory addressing information
- `core`: GPU core context
- `inst`: Warp instruction (passed by reference for memory operation setup)

**Behavior**:
- Loads matrix fragments in distributed fashion across warp threads
- Each thread loads its assigned portion of the matrix
- Supports both row-major and column-major layouts
- Handles stride and base address calculations

**Memory Access Pattern**: Coalesced access when possible, determined by layout mode and shape.

---

#### `tensor_mma_st_impl()`

**Signature**:
```cpp
void tensor_mma_st_impl(const ptx_instruction *pI, core_t *core, warp_inst_t &inst)
```

**Purpose**: Store computed matrix fragments from registers to memory.

**Parameters**:
- `pI`: PTX instruction with memory addressing information
- `core`: GPU core context
- `inst`: Warp instruction (passed by reference for memory operation setup)

**Behavior**:
- Stores result matrix fragments in distributed fashion
- Each thread stores its computed portion
- Maintains layout mode specified in instruction

---

## Internal Helpers

### Type Conversion Functions

#### `mma_f16_to_f32()`

**Signature**: `float mma_f16_to_f32(uint16_t f16)`

**Purpose**: Convert IEEE 754 half-precision (float16) to single-precision (float32).

**Input**: 16-bit half-precision float (1 sign bit, 5 exponent bits, 10 mantissa bits)

**Output**: 32-bit single-precision float

**Special Cases**:
- Denormals: Handled with proper scaling
- Infinity: Preserved
- NaN: Preserved

---

#### `mma_f32_to_f16()`

**Signature**: `uint16_t mma_f32_to_f16(float f32)`

**Purpose**: Convert float32 to float16 with rounding.

**Input**: 32-bit single-precision float

**Output**: 16-bit half-precision float

**Rounding**: Rounds to nearest even

**Special Cases**:
- Underflow: Flush to zero
- Overflow: Saturate to infinity

---

#### `mma_bf16_to_f32()`

**Signature**: `float mma_bf16_to_f32(uint16_t bf16)`

**Purpose**: Convert BFloat16 to float32.

**Input**: 16-bit bfloat16 (1 sign bit, 8 exponent bits, 7 mantissa bits)

**Output**: 32-bit single-precision float

**Implementation**: Simple bit shift (BF16 is truncated FP32)

---

#### `mma_f32_to_bf16()`

**Signature**: `uint16_t mma_f32_to_bf16(float f32)`

**Purpose**: Convert float32 to BFloat16 with truncation.

**Input**: 32-bit single-precision float

**Output**: 16-bit bfloat16

**Implementation**: Truncates mantissa (no rounding)

---

#### `mma_tf32_round()`

**Signature**: `float mma_tf32_round(float f32)`

**Purpose**: Round float32 to TensorFloat-32 precision.

**Input**: 32-bit single-precision float

**Output**: 32-bit float rounded to TF32 precision (1 sign, 8 exponent, 10 mantissa bits)

**Rounding Mode**: **Truncation** (round-toward-zero)
- Implementation: Clear lower 13 bits of FP32 mantissa
- Applied to input matrices A and B before multiplication
- Accumulation uses full FP32 precision (no intermediate rounding)

**Use Case**: Simulates TF32 arithmetic precision for Ampere+ tensor cores

**Implementation Note**: Both M16N8K4 and M16N8K8 TF32 MMA variants use the same rounding behavior:
- K=4: 4 TF32 multiply-accumulate operations per output element
- K=8: 8 TF32 multiply-accumulate operations per output element
- Rounding applied consistently to inputs in both paths

---

### Saturation Functions

#### `mma_saturate_s8()`

**Signature**: `int8_t mma_saturate_s8(int32_t val)`

**Purpose**: Saturate int32 value to signed 8-bit range.

**Range**: [-128, 127]

**Behavior**: Clamps values outside range to nearest boundary

---

#### `mma_saturate_u8()`

**Signature**: `uint8_t mma_saturate_u8(int32_t val)`

**Purpose**: Saturate int32 value to unsigned 8-bit range.

**Range**: [0, 255]

**Behavior**: Clamps negative values to 0, values > 255 to 255

---

#### `mma_saturate_s4()`

**Signature**: `int8_t mma_saturate_s4(int32_t val)`

**Purpose**: Saturate int32 value to signed 4-bit range.

**Range**: [-8, 7]

**Use Case**: Sub-byte precision integer MMA operations

---

#### `mma_saturate_u4()`

**Signature**: `uint8_t mma_saturate_u4(int32_t val)`

**Purpose**: Saturate int32 value to unsigned 4-bit range.

**Range**: [0, 15]

**Use Case**: Sub-byte precision integer MMA operations

---

### Thread-to-Element Mapping

#### `mma_thread_to_element_offset()`

**Signature**:
```cpp
unsigned mma_thread_to_element_offset(unsigned thread_id,
                                      mma_shape_type shape,
                                      mma_layout_mode layout,
                                      unsigned char type_size,
                                      unsigned stride)
```

**Purpose**: Maps warp thread ID to byte offset in matrix fragment based on MMA shape and layout.

**Parameters**:
- `thread_id`: Warp-relative thread ID (0-31)
- `shape`: MMA shape (e.g., M16N8K8, M16N16K16)
- `layout`: Matrix layout mode (ROW-COL, ROW-ROW, COL-ROW, COL-COL)
- `type_size`: Element size in bytes (2 for F16/BF16, 4 for F32, 1 for S8/U8)
- `stride`: Row/column stride in elements (NOT bytes) for layout calculation

**Returns**: Byte offset into matrix fragment for the given thread

**Thread Distribution Example** (M16N8K8):
- Thread 0 handles elements at specific (row, col) positions
- Thread mapping ensures efficient memory coalescing
- Different shapes have different distribution patterns

**Use Case**: Called during MMA execution to determine which matrix elements each thread processes.

---

## Data Structures

### `mma_shape_type` enum

Defines supported MMA shapes (M×N×K):
- `MMA_M16N8K8`: 16×8 output, K=8 accumulation
- `MMA_M16N8K16`: 16×8 output, K=16 accumulation
- `MMA_M16N8K32`: 16×8 output, K=32 accumulation (int4/int1)
- `MMA_M16N8K64`: 16×8 output, K=64 accumulation (int1)
- `MMA_M16N16K8`: 16×16 output, K=8 accumulation
- `MMA_M16N16K16`: 16×16 output, K=16 accumulation
- `MMA_M8N8K4`: 8×8 output, K=4 accumulation (fp64)
- `MMA_M8N8K16`: 8×8 output, K=16 accumulation
- `MMA_M8N8K32`: 8×8 output, K=32 accumulation
- `MMA_M8N8K128`: 8×8 output, K=128 accumulation (int1)

### `mma_layout_mode` enum

Defines matrix layout combinations:
- `MMA_ROW_COL`: A is row-major, B is column-major
- `MMA_ROW_ROW`: A is row-major, B is row-major
- `MMA_COL_ROW`: A is column-major, B is row-major
- `MMA_COL_COL`: A is column-major, B is column-major

**Note**: C and D matrices always use same layout; only A and B layouts vary.

---

## Integration with GPGPU-Sim

### Namespace

All MMA implementation code is wrapped in `namespace flash_gpgpu_sim` to:
- Separate from existing WMMA implementations
- Group with other modern GPU features (TMA, mbarrier)
- Maintain clear architectural boundaries

### Usage in `instructions.cc`

The main instruction decoder calls MMA implementations:
```cpp
#include "../gpgpu-sim/flash/mma/tensor_mma.h"

// In instruction execution:
case TENSOR_MMA_OP:
  flash_gpgpu_sim::tensor_mma_impl(pI, core, inst);
  break;
```

### PTX Parser Integration

MMA opcodes defined in `opcodes.def` and parsed in `ptx.l`:
- Lexer recognizes `mma.sync.aligned.*` syntax
- Parser extracts shape, types, layouts from instruction string
- Stores configuration in `ptx_instruction` structure

---

## Testing

See `test/src/integration/mma/cuda_mma_*_test.cc` for:
- Integration tests for full MMA instruction execution
- Test coverage for all supported shapes and data types (F16, BF16, TF32, S8/U8)
