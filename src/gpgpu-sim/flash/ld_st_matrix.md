# ldmatrix/stmatrix Implementation

This document describes FlashGPU-Sim's implementation of the PTX `ldmatrix`
and `stmatrix` instruction handlers. Refer to the
[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
for the normative instruction syntax and fragment layout.

## Supported and Validated Variants

The integration tests validate both load and store operations for:

- `m8n8.b16` without `.trans`: `x1`, `x2`, and `x4`
- `m8n8.b16` with `.trans`: `x1`

The transpose implementation is shared by all matrix-count variants, but
`.trans.x2` and `.trans.x4` do not currently have integration coverage and are
not part of the validated support set. Other shapes and element types are also
not part of the validated support set; the handler aborts for shapes other than
`m8n8`.

## Overview

The ldmatrix/stmatrix implementation provides warp-level cooperative memory operations for tensor core workflows. The implementation uses a unified template-based handler to minimize code duplication while maintaining symmetry between load and store operations.

## External Interface

### Primary Entry Points

#### `handle_ldmatrix_inst()`

```cpp
void handle_ldmatrix_inst(const ptx_instruction *pI, core_t *core, warp_inst_t &inst);
```

Entry point for executing ldmatrix (load matrix) PTX instructions.

**Parameters:**
- `pI`: PTX instruction object containing parsed instruction metadata (opcode, options, operands)
- `core`: Shader core context providing thread information and warp state
- `inst`: Warp-level instruction representation for timing simulation

**Behavior:**
- Parses instruction options (shape, num matrices, transpose, scalar type)
- Validates required options, operand count, and register-vector size
- Loads matrix fragments from shared memory to thread registers
- Distributes matrix elements across warp threads according to fragment layout

**Error Conditions:**
- Aborts on unsupported matrix shapes (currently only m8n8 supported)
- Asserts on invalid option combinations (e.g., missing required options)
- Asserts on operand count mismatch

**Thread Requirements:**
- Must be called in warp-level context (all 32 threads participate)
- Requires the mandatory PTX `.sync` and `.aligned` qualifiers

---

#### `handle_stmatrix_inst()`

```cpp
void handle_stmatrix_inst(const ptx_instruction *pI, core_t *core, warp_inst_t &inst);
```

Entry point for executing stmatrix (store matrix) PTX instructions.

**Parameters:**
- `pI`: PTX instruction object containing parsed instruction metadata
- `core`: Shader core context providing thread information
- `inst`: Warp-level instruction representation

**Behavior:**
- Parses instruction options (shape, num matrices, transpose, scalar type)
- Validates required options, operand count, and register-vector size
- Stores matrix fragments from thread registers to shared memory
- Uses same fragment distribution as ldmatrix (symmetric operation)

**Error Conditions:**
- Same as `handle_ldmatrix_inst()` (unified implementation)

**Thread Requirements:**
- Must be called in warp-level context
- Requires the mandatory PTX `.sync` and `.aligned` qualifiers

---

## Internal Helpers

### Shape Specification

#### `LdMatrixShapeSpec` Struct

```cpp
struct LdMatrixShapeSpec {
  int rows;            // M dimension (matrix height)
  int cols;            // N dimension (matrix width)
  int lanes_per_row;   // Threads per row
  int cols_per_lane;   // Elements per thread (column-wise)
};
```

Encapsulates fragment distribution parameters for a matrix shape.

**Purpose:**
- Defines thread-to-matrix-element mapping for warp-level operations
- Enables parametric dispatch without shape-specific code paths
- Documents fragment layout explicitly

**Field Semantics:**
- `rows`: Number of rows in matrix tile (M dimension)
- `cols`: Number of columns in matrix tile (N dimension)
- `lanes_per_row`: How many threads contribute to each row (32 threads / rows)
- `cols_per_lane`: How many consecutive columns each thread handles

**Example (m8n8):**
```cpp
{8, 8, 4, 2}  // 8x8 matrix, 4 threads/row, 2 elements/thread
```

**Fragment Distribution Calculation:**
```cpp
row_id = lane_id / lanes_per_row;
col_id = (lane_id % lanes_per_row) * cols_per_lane;
// Thread loads/stores elements at [row_id, col_id] and [row_id, col_id+1]
```

---

#### `get_shape_spec()`

```cpp
static LdMatrixShapeSpec get_shape_spec(int matrix_shape);
```

Maps PTX matrix shape option to shape specification struct.

**Parameters:**
- `matrix_shape`: PTX option constant (e.g., `M8N8_OPTION`)

**Returns:**
- `LdMatrixShapeSpec` struct with distribution parameters

**Supported Shapes:**
- `M8N8_OPTION`: Returns `{8, 8, 4, 2}` for 8x8 matrix

**Error Handling:**
- Prints error message and aborts for unsupported shapes
- This is intentional to catch implementation gaps during development

**Usage Pattern:**
```cpp
LdMatrixShapeSpec spec = get_shape_spec(matrix_shape);
// Use spec.rows, spec.cols, etc. to drive fragment distribution
```

---

### Unified Implementation

#### `handle_ld_st_matrix_inst_impl<Direction>()`

```cpp
template <MatrixDirection Direction>
static void handle_ld_st_matrix_inst_impl(
    const ptx_instruction *pI,
    core_t *core,
    warp_inst_t &inst
);
```

Templated implementation handling both ldmatrix and stmatrix.

**Template Parameters:**
- `Direction`: `MatrixDirection::LOAD` for ldmatrix, `MatrixDirection::STORE` for stmatrix

**Compile-Time Branching:**
Uses `if constexpr (is_load)` to select load vs. store behavior without runtime overhead.

**Algorithm Overview:**

1. **Parse Options:**
   - Extract shape, num matrices, transpose flag, scalar type
   - Validate all required options present and compatible

2. **Get Shape Specification:**
   - Call `get_shape_spec()` to obtain fragment distribution parameters

3. **Iterate Over Warp Threads:**
   - Calculate each thread's fragment position using shape-spec
   - Determine address provider threads based on fragment distribution

4. **Handle Transpose:**
   - **Non-transpose**: Load/store contiguous elements (single memory access)
   - **Transpose**: Load/store strided elements (separate accesses per element)

5. **Memory Operations:**
   - **Load**: Read from shared memory to thread registers
   - **Store**: Write from thread registers to shared memory

**Fragment Distribution Logic:**
```cpp
// Map lane_id to matrix position
matrix_row_id = lane_id / spec.lanes_per_row;
matrix_col_id = (lane_id % spec.lanes_per_row) * spec.cols_per_lane;

// Non-transpose: access [row, col] and [row, col+1]
// Transpose: access [col, row] and [col+1, row]
```

**Transpose Semantics:**
- **Non-transpose**: Thread accesses consecutive columns from the row base
  supplied by the address-provider lane
  - Address: `row_base + col * element_size`
  - Single contiguous read/write of `cols_per_lane * element_size` bytes

- **Transpose**: Each element selects a different address-provider row
  (`target_row = col + element`) and applies the fragment row as its offset
  - Address: `target_row_base + row * element_size`
  - Separate read/write per element (elements are strided, not contiguous)

**Symmetry Guarantee:**
The template design ensures ldmatrix and stmatrix use identical fragment distribution, preventing load/store mismatches.

---

## Implementation Notes

### Design Rationale

**Unified Handler:**
- Reduces code duplication between load and store variants
- Enforces load/store symmetry (same fragment distribution)
- Simplifies maintenance (single code path to debug/optimize)

**Shape-Spec Table:**
- Avoids variant explosion (no separate function per shape)
- Makes distribution parameters explicit and documentable
- Enables future shape additions with minimal code change
- Alternative would require 3 shapes × 3 num variants × 2 directions = 18 functions

**Template Metaprogramming:**
- `if constexpr` eliminates runtime overhead of direction checks
- Compiler generates optimized load-specific and store-specific code
- Type safety: Cannot accidentally mix load/store logic

### Thread Synchronization

**Implicit Warp Barrier:**
- PTX `.sync` modifier provides warp-level synchronization
- All 32 threads execute instruction in lockstep (SIMT model)
- No explicit barrier needed in implementation

**Address Provider Threads:**
- For m8n8: Threads providing row addresses depend on matrix_id and num matrices
- The lane providing row `R` of matrix `M` is `M * 8 + R`
- Other threads use these addresses to calculate element positions

### Memory Access Patterns

**Alignment Requirements:**
- The PTX `.aligned` qualifier is a warp-convergence requirement; it does not
  mean that the address must be aligned to 128 bytes.
- For `m8n8.b16`, each group of four lanes transfers a 16-byte row fragment,
  which must be naturally aligned.
- The handler does not validate address alignment. Integration tests use
  16-byte-aligned shared storage.

**Access granularity:**
- Non-transpose accesses transfer two adjacent `b16` elements at once.
- Transpose accesses transfer the two `b16` elements separately.

## Related Documentation

- **Test Coverage**: `test/src/integration/cuda_ld_st_matrix_test.cc`
- **Flash Module Overview**: `src/gpgpu-sim/flash/README.md`
