# ldmatrix/stmatrix Instructions

This document describes the PTX ldmatrix (load matrix) and stmatrix (store matrix) instructions used for efficient tensor core memory operations in GPGPU-Sim.

## Overview

The `ldmatrix` and `stmatrix` instructions provide cooperative, warp-level memory operations for loading from and storing to shared memory. They are specifically designed for tensor core workflows where 32 threads in a warp collaboratively move matrix tile data between shared memory and registers.

**Key characteristics:**
- **Warp-level**: All 32 threads participate in each operation
- **Fragment distribution**: Each thread loads/stores specific matrix elements based on a fixed pattern
- **Synchronized**: Implicit warp-level synchronization (`.sync` modifier)
- **Aligned**: Memory accesses must be 128-byte aligned (`.aligned` modifier)

## PTX Syntax

### ldmatrix (Load Matrix)

```ptx
ldmatrix.sync.aligned.shape.num{.trans}{.ss}.type r, [p]
```

**Modifiers:**
- `.shape`: Matrix tile dimensions
  - `.m8n8`: 8×8 tile (64 elements) - **16-bit elements only**
  - `.m16n16`: 16×16 tile (256 elements) - **8-bit, 6-bit, or 4-bit elements**
- `.num`: Number of matrices to load per warp
  - `.x1`: 1 matrix
  - `.x2`: 2 matrices
  - `.x4`: 4 matrices
- `.trans` (optional): Interpret memory layout as transposed
- `.ss`: State space (always `.shared` for ldmatrix)
- `.type`: Element data type
  - `.b16`: 16-bit elements (2 bytes per element) - **m8n8 only**
  - `.b8`: 8-bit elements (1 byte per element) - **m16n16 only** (not yet implemented)

**Example:**
```ptx
ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%r0, %r1}, [%ptr];
```

### stmatrix (Store Matrix)

```ptx
stmatrix.sync.aligned.shape.num{.trans}{.ss}.type [p], r
```

**Modifiers:** Same as ldmatrix (shape, num, trans, type)

**Example:**
```ptx
stmatrix.sync.aligned.m8n8.x1.shared.b16 [%ptr], {%r0, %r1};
```

## Supported Variants

**Currently Implemented:**

| Shape | Num Variants | Data Types | Transpose | Total Variants |
|-------|--------------|------------|-----------|----------------|
| m8n8  | x1, x2, x4   | b16        | yes       | 6 (3 load + 3 store) |
| **Total** | | | | **6 non-transpose + 2 transpose = 8** |

**Valid PTX 9.1 Combinations** (basic variant without src_fmt/dst_fmt):

| Shape | Valid Data Types | Notes |
|-------|-----------------|-------|
| m8n8  | .b16 only | 16-bit elements (fully implemented with transpose support) |

**Note:**
- The shape m16n8 does NOT exist in PTX ISA
- The m16n16 shape requires src_fmt/dst_fmt modifiers (advanced variant not currently supported)
- Only m8n8 with .b16 is implemented for the basic ldmatrix/stmatrix variant

## Fragment Distribution Mapping

Fragment distribution defines which matrix elements each thread loads/stores. The mapping is **shape-specific** and follows PTX ISA semantics.

### Shape Specification Table

Each shape has fixed distribution parameters:

```c
struct LdMatrixShapeSpec {
    int rows;            // M dimension (height)
    int cols;            // N dimension (width)
    int lanes_per_row;   // Threads per row
    int cols_per_lane;   // Elements per thread (column-wise)
};
```

| Shape | M (rows) | N (cols) | Lanes/Row | Cols/Lane | Elements/Thread |
|-------|----------|----------|-----------|-----------|-----------------|
| m8n8  | 8        | 8        | 4         | 2         | 2 |

### Thread-to-Element Mapping

**For non-transposed loads/stores:**

Each thread `lane_id` (0-31) maps to matrix elements:

```c
// Shape m8n8
row = lane_id / 4  // 0-7 (8 rows)
col = (lane_id % 4) * 2  // {0,2,4,6} (2 elements per thread)
// Loads/stores: [row, col] and [row, col+1]
```

**Example: m8n8 thread mapping**
```
Thread  | Row | Cols Loaded
--------|-----|-------------
0       | 0   | 0, 1
1       | 0   | 2, 3
2       | 0   | 4, 5
3       | 0   | 6, 7
4       | 1   | 0, 1
...     |     |
28      | 7   | 0, 1
31      | 7   | 6, 7
```

### Transpose Modifier Semantics

When `.trans` is specified:
- **Memory interpretation changes**: Load/store accesses memory as if the tile is transposed
- **Fragment distribution unchanged**: Thread-to-element mapping remains the same
- **Address calculation**: Row/column indices are swapped during address computation

**Implementation detail:**
```c
if (is_transpose) {
    // Swap row/col when computing shared memory address
    addr = base + (col * M + row) * element_size;
} else {
    // Normal row-major addressing
    addr = base + (row * N + col) * element_size;
}
```

### Data Type Packing

**b16 (16-bit elements):**
- 2 bytes per element
- Each 32-bit register holds 2 elements
- Thread loading 2 elements → 1 register

**b8 (8-bit elements):**
- 1 byte per element
- Each 32-bit register holds 4 elements
- Thread loading 2 elements → 1 register (with packing)

## Test Matrix

Test coverage for ldmatrix/stmatrix variants (8 tests total):

### Non-Transpose Tests (6 tests)

| Shape | Num | Type | Load Test | Store Test |
|-------|-----|------|-----------|------------|
| m8n8  | x1  | b16  | ✅ LdMatrixX1Test | ✅ StMatrixX1Test |
| m8n8  | x2  | b16  | ✅ LdMatrixX2Test | ✅ StMatrixX2Test |
| m8n8  | x4  | b16  | ✅ LdMatrixX4Test | ✅ StMatrixX4Test |

### Transpose Tests (2 tests)

| Shape | Num | Type | Load+Trans Test | Store+Trans Test |
|-------|-----|------|-----------------|------------------|
| m8n8  | x1  | b16  | ✅ LdMatrixM8N8TransTest | ✅ StMatrixM8N8TransTest |

**Note:** Transpose tests validate the `.trans` modifier which swaps row/column in address calculation, effectively transposing the memory layout interpretation.

### Removed Tests (invalid shape/type combinations)

The following combinations are **invalid per PTX specification** and have been removed:
- ❌ m8n8 with .b8 (m8n8 only supports .b16)
- ❌ m16n16 without src_fmt/dst_fmt (requires advanced variant)
- ❌ m16n8 (shape does not exist in PTX ISA)

## Implementation Notes

### Unified Handler Architecture

The implementation uses a **single templated function** for both load and store:

```c
template<Direction dir>
void handle_ld_st_matrix_inst_impl(const ptx_instruction* pI, ptx_thread_info* thread);
```

**Direction parameter:**
- `Direction::LOAD` → ldmatrix behavior
- `Direction::STORE` → stmatrix behavior

**Key advantages:**
- Fragment distribution logic is identical for load/store
- Reduces code duplication
- Ensures symmetry between load and store operations

### Shape-Spec Table Approach

Instead of 36 specialized handlers (one per variant), the implementation uses **parametric dispatch**:

```c
// Get shape spec based on parsed option
LdMatrixShapeSpec spec = get_shape_spec(pI->get_options());

// Use spec to drive fragment distribution
for (int lane_id = 0; lane_id < 32; lane_id++) {
    int row = lane_id / spec.lanes_per_row;
    int col = (lane_id % spec.lanes_per_row) * spec.cols_per_lane;
    // Load/store elements [row, col] and [row, col+1]
}
```

This approach:
- Avoids variant explosion
- Makes shape parameters explicit
- Simplifies testing (change spec, not code)

## Usage Example

**Kernel using ldmatrix/stmatrix:**

```cuda
__global__ void tensor_copy_kernel() {
    __shared__ half smem_in[8 * 8];
    __shared__ half smem_out[8 * 8];

    // Load 8×8 matrix from shared memory (2 matrices, b16 type)
    uint32_t regs[2];
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
        : "=r"(regs[0]), "=r"(regs[1])
        : "l"((uintptr_t)smem_in)
    );

    // Store back to different shared memory location
    asm volatile(
        "stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};"
        :
        : "l"((uintptr_t)smem_out),
          "r"(regs[0]), "r"(regs[1])
    );
}
```

## Testing Instructions

Run ldmatrix/stmatrix tests:

```bash
# Setup environment
source setup.sh && source setup_environment

# Build tests
./test/run_tests.sh build

# Run all ldmatrix/stmatrix tests (8 tests total: 6 non-transpose + 2 transpose)
./test/run_tests.sh -c SM120_RTX5090 run "*LdMatrix*"
./test/run_tests.sh -c SM120_RTX5090 run "*StMatrix*"

# Run only transpose tests (2 tests)
./test/run_tests.sh -c SM120_RTX5090 run "*Trans*"
```

## References

- **PTX ISA 9.1**: Section 9.7.13.8 (ldmatrix) and 9.7.13.9 (stmatrix)
- **Related instructions**: See `docs/mma_instructions.md` for tensor core compute operations
- **Test implementation**: `test/src/integration/cuda_ld_st_matrix_test.cc`
- **Handler implementation**: `src/gpgpu-sim/flash/ld_st_matrix.cc`
