# TMA (Tensor Memory Accelerator) Implementation Interface

This document describes the C++ implementation interface for PTX TMA (Tensor
Memory Accelerator) instructions in GPGPU-Sim. See the
[test framework guide](../../../tests/README.md) for test execution.

## External Interface

### Main Instruction Handlers

#### `handle_tma_inst()`

**Signature**:
```cpp
void handle_tma_inst(const ptx_instruction *pI, ptx_thread_info *thread);
```

**Purpose**: Execute PTX `cp.async.bulk` and `cp.async.bulk.tensor` instructions for asynchronous bulk memory transfers via the TMA hardware unit.

**PTX Syntax**:
```ptx
cp.async.bulk.dst.src.mbarrier::complete_tx::bytes [dst], [src], size, [smem_bar];
cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes [addr], [map, {c0}], [smem_bar];
cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes [addr], [map, {c0, c1}], [smem_bar];
...
cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes [addr], [map, {c0, c1, c2, c3, c4}], [smem_bar];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group N;
```

**Parameters**:
- `pI`: PTX instruction containing:
  - Instruction type (linear bulk copy or tensor operation)
  - Dimension (1D-5D for tensor operations)
  - Memory space qualifiers (shared/global)
  - Mbarrier address for completion notification
- `thread`: Current thread context for execution

**Behavior**:
- **Linear bulk copy**: Transfers contiguous memory block from global to shared memory
- **Tensor operation**: Transfers multi-dimensional tensor tile using tensormap descriptor
  - Supports 1D through 5D tensors
  - Parses coordinate vectors: `{c0}` (1D), `{c0, c1}` (2D), ..., `{c0, c1, c2, c3, c4}` (5D)
  - Reads tensormap descriptor from memory to obtain tile dimensions and strides
  - Calculates source address based on coordinates and descriptor parameters
- **Commit group** (stubbed): Treated as NOP with debug logging
- **Wait group** (stubbed): Treated as NOP with debug logging

**Functional Simulation**: Performs immediate memory copy between global and shared memory spaces.

**Timing Simulation**: Populates `tma_static_info_t` and `tma_dyn_info_t` structures for performance simulation via `tma_unit_impl_t`.

**Multi-Dimensional Support**: All dimensions (1D-5D) fully supported with proper L2 cache integration and response handling (verified in issue #31).

**Error Conditions**:
- Invalid tensormap descriptor → prints error and exits
- Dimension mismatch between instruction and tensormap → prints error and exits
- Unsupported instruction variant → assertion failure

---

#### `handle_tensormap_inst()`

**Signature**:
```cpp
void handle_tensormap_inst(const ptx_instruction *pI, ptx_thread_info *thread);
```

**Purpose**: Execute PTX `tensormap.replace` and `tensormap.cp_fenceproxy` instructions for modifying tensormap descriptors.

**PTX Syntax**:
```ptx
tensormap.replace.tile.global_address.b1024.b64 [desc], new_addr;
tensormap.replace.tile.rank.b1024.b32 [desc], new_rank;
tensormap.replace.tile.box_dim.b1024.b32 [desc], ord, new_box_dim;
...
tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.sem.scope [dst], [src], size;
```

**Parameters**:
- `pI`: PTX instruction with field type to modify (global_address, rank, box_dim, etc.)
- `thread`: Current thread context

**Behavior**:
- **tensormap.replace**: Modifies specific field in tensormap descriptor (global address, rank, dimensions, strides, etc.)
- **tensormap.cp_fenceproxy**: Copies tensormap with fence semantics (stubbed as NOP)

**Functional Simulation**: Reads descriptor, modifies specified field, writes back to memory.

**Error Conditions**: Unsupported field type → prints stub message

---

### TMA Unit Interface

#### `tma_unit_impl_t::warp_reaches_tma()`

**Signature**:
```cpp
void warp_reaches_tma(unsigned cta_id, unsigned warp_id, warp_inst_t *inst);
```

**Purpose**: Enqueue TMA transaction for asynchronous processing when warp executes TMA instruction.

**Parameters**:
- `cta_id`: CTA (thread block) ID
- `warp_id`: Warp ID within CTA
- `inst`: Warp instruction containing TMA operation details

**Behavior**:
- Creates `tma_transaction_t` for each participating thread (lane) in warp
- Initializes AGU (Address Generation Unit) state for tensor or linear mode
- Assigns unique transaction ID and enqueues to issue queue
- Registers transaction with mbarrier for completion notification

**Transaction Lifecycle**: Transaction remains active until all memory fetches complete and bytes transferred equals expected size.

---

#### `tma_unit_impl_t::cycle()`

**Signature**:
```cpp
void cycle();
```

**Purpose**: Execute one simulation cycle for TMA unit, processing responses and issuing memory requests.

**Behavior (executed each cycle)**:
1. **Response Processing**:
   - Dequeue `mem_fetch` responses from response FIFO
   - Handle both direct responses and L2-subdivided sector responses (using `get_original_mf()`)
   - Accumulate bytes completed for transaction
   - When transaction complete: notify mbarrier, remove transaction, clean up mappings
   - Gracefully handle late responses (responses arriving after transaction marked complete)

2. **Request Issuance**:
   - For each transaction in issue queue with available interconnect credits
   - Use AGU to generate next memory request address and size
   - Compute byte_mask and sector_mask for L2 cache integration
   - Create `mem_fetch` with TMA_ACC_R access type
   - Send to interconnect via `m_icnt->push()`
   - Track mapping from `mem_fetch` UID to transaction UID

**AGU Integration**: Uses `tma_agu_unit_t` to generate cache-line-aligned requests covering entire tensor tile.

**L2 Cache Integration**: Properly sets byte_mask and sector_mask to enable L2 cache subdivision of large requests into 32-byte sectors.

**Completion Detection**: Transaction completes when `m_bytes_completed >= size_in_bytes`.

---

## Internal Helpers

### Operand Extraction

#### `get_operand_u32()`

**Signature**: `static uint32_t get_operand_u32(ptx_thread_info *thread, const operand_info &op)`

**Purpose**: Extract 32-bit unsigned value from PTX operand.

**Returns**: Value of operand interpreted as U32 type.

---

#### `get_operand_u64()`

**Signature**: `static uint64_t get_operand_u64(ptx_thread_info *thread, const operand_info &op)`

**Purpose**: Extract 64-bit unsigned value from PTX operand.

**Returns**: Value of operand interpreted as U64 type.

---

### Dimension and Coordinate Parsing

#### `compute_inst_dim()`

**Signature**: `static unsigned compute_inst_dim(unsigned dim_option)`

**Purpose**: Convert PTX dimension option to numeric dimension value.

**Input**: Dimension option constant (DIM_1D_OPTION through DIM_5D_OPTION)

**Output**: Dimension value (1-5)

**Error Handling**: Unknown dimension → prints error and aborts.

---

#### `parse_tensor_coords()`

**Signature**: `static void parse_tensor_coords(ptx_thread_info *thread, const operand_info &coord_operand, int32_t coords[5])`

**Purpose**: Parse tensor coordinates from vector operand into coordinate array.

**Parameters**:
- `thread`: Thread context for operand value extraction
- `coord_operand`: Vector operand containing coordinates (e.g., `{c0, c1, c2}` for 3D)
- `coords`: Output array (pre-allocated, size 5), initialized to all zeros

**Behavior**:
- If operand is vector: extracts each coordinate element (up to 5 elements)
- If operand is scalar: treats as 1D coordinate (coords[0] only)
- Unused dimensions remain zero

---

### Validation

#### `validate_tensormap()`

**Signature**: `static bool validate_tensormap(const tensormap_descriptor_t &tensormap, unsigned inst_dim)`

**Purpose**: Verify tensormap descriptor validity and dimension consistency.

**Checks**:
1. Tensormap descriptor is valid (via `is_valid()`)
2. Tensormap dimension matches instruction dimension

**Error Handling**: Validation failure → prints descriptor, flushes output, exits.

**Returns**: True if valid (function exits on failure, so return is always true).

---

### Address Calculation

#### `global_to_tile_offset()`

**Signature**: `static uint64_t global_to_tile_offset(uint64_t global_addr, uint64_t base_addr, const tensormap_descriptor_t& tensormap)`

**Purpose**: Convert global memory address offset to tile-local offset accounting for different stride patterns.

**Problem**: Global tensor may have different row stride than tile (based on box dimensions). A linear offset in global memory does not map to the same linear offset in the tile.

**Algorithm**:
1. Compute global byte offset: `global_addr - base_addr`
2. Decompose offset into coordinates using global strides
3. Recompute offset using tile strides (based on box dimensions)

**Special Cases**:
- 1D: Tile offset equals global offset (no stride difference)
- 2D+: Requires coordinate decomposition and stride conversion

**Returns**: Tile-local offset in bytes.

---

#### `gen_aligned_req()`

**Signature**: `static void gen_aligned_req(uint64_t start_addr, uint32_t total_bytes, std::vector<std::pair<uint64_t, uint32_t>>& requests)`

**Purpose**: Split contiguous memory range into cache-line-aligned memory requests.

**Parameters**:
- `start_addr`: Starting address (may be unaligned)
- `total_bytes`: Total bytes to transfer
- `requests`: Output vector of (address, size) pairs

**Algorithm**:
1. First request: From start_addr to next 128-byte boundary (or total_bytes if smaller)
2. Middle requests: Full 128-byte cache lines
3. Last request: Remaining bytes (if any)

**Alignment**: Ensures all but first request are 128-byte aligned for optimal cache performance.

---

### TMA Transfer Execution

#### `do_tma_transfer()`

**Signature**: `static void do_tma_transfer(const tensormap_descriptor_t &tensormap, const int32_t coords[5], memory_space *shared_mem, memory_space *global_mem, uint64_t shared_addr, ptx_thread_info *thread, const ptx_instruction *pI, bool is_load, tma_reduction_op_t reduction_op)`

**Purpose**: Perform functional simulation of TMA transfer between global and shared memory.

**Parameters**:
- `tensormap`: Descriptor specifying tensor layout and tile dimensions
- `coords`: Tensor coordinates identifying which tile to transfer
- `shared_mem`, `global_mem`: Memory space interfaces
- `shared_addr`: Base address in shared memory
- `thread`: Thread context
- `pI`: Instruction pointer (for error reporting)
- `is_load`: True for load (global→shared), false for store (shared→global)
- `reduction_op`: Optional element-wise reduction for tensor stores

**Behavior**:
1. Calculate global base address from coordinates and tensormap
2. Determine tile size and dimensions from tensormap
3. Generate aligned memory requests covering tile (via `gen_aligned_req()`)
4. For each request:
   - Read from source memory space
   - Convert global offset to tile offset (if needed)
   - Write to destination memory space
5. Verify all bytes transferred

**Multi-Dimensional Support**: Handles 1D-5D tensors with proper coordinate-to-address mapping.

### Tensor Reduction Stores

`cp.reduce.async.bulk.tensor` uses the normal tensor-store address generation
and reverse-swizzle path, then atomically reduces each shared-memory element
into its corresponding global-memory element in functional simulation. The
supported operation/type pairs follow the PTX ISA:

- `add`: `u32`, `s32`, `u64`, `f16`, `bf16`, `f32`
- `min`, `max`: `u32`, `s32`, `u64`, `s64`, `f16`, `bf16`
- `inc`, `dec`: `u32`
- `and`, `or`, `xor`: 32-bit and 64-bit integer tensor-map types

`FLOAT32` preserves subnormal inputs and results, while `FLOAT32_FTZ` flushes
them to signed zero. This distinction is preserved by both the host tensor-map
encoder and the functional reduction path.

Only the `.tile` addressing mode is currently implemented. The functional
read-modify-write is indivisible in the simulator execution path and therefore
preserves the PTX element-wise atomic result.

The timing model is not calibrated for `UTMAREDG`. It currently reuses normal
TMA tensor-store requests and completion. In particular, it does not model the
atomic destination read, same-address serialization, or a dedicated reduction
backend resource. Timing results for tensor reduction stores must be treated as
uncalibrated until a hardware microbenchmark supplies those parameters.

---

## TMA AGU (Address Generation Unit)

### `tma_agu_unit_t`

**Purpose**: Generate sequence of memory addresses for TMA transfers, handling both linear and tensor modes.

#### `init_linear()`

**Signature**: `void init_linear(tma_agu_state_t &state, uint64_t base_addr, uint32_t size_bytes)`

**Purpose**: Initialize AGU for linear (non-tensor) bulk transfer.

**Parameters**:
- `state`: AGU state to initialize
- `base_addr`: Starting address
- `size_bytes`: Total bytes to transfer

---

#### `init_tensor()`

**Signature**: `void init_tensor(tma_agu_state_t &state, const tensormap_descriptor_t &tensormap, const int32_t coords[5])`

**Purpose**: Initialize AGU for tensor transfer using tensormap descriptor.

**Parameters**:
- `state`: AGU state to initialize
- `tensormap`: Descriptor specifying tensor layout
- `coords`: Tile coordinates

**Behavior**: Calculates base address from coordinates, extracts tile size from descriptor, initializes state.

---

#### `gen_next_req()`

**Signature**: `bool gen_next_req(tma_agu_state_t &state, uint64_t &out_addr, uint32_t &out_size)`

**Purpose**: Generate next memory request address and size from AGU state.

**Parameters**:
- `state`: Current AGU state (updated on each call)
- `out_addr`: Output parameter for next request address
- `out_size`: Output parameter for next request size

**Returns**: True if request generated, false if all bytes transferred.

**Algorithm**: Uses shadow stride accumulation to generate cache-line-aligned requests covering full transfer size.

---

## Data Structures

### `tma_transaction_t`

**Purpose**: Track state of a single TMA transaction from initiation to completion.

**Key Fields**:
- `m_thread`: Thread that initiated transaction
- `m_inst`: PTX instruction pointer
- `m_static_info`: Static information (TMA type, memory spaces)
- `m_dyn_info`: Dynamic information (addresses, size, coordinates, mbarrier)
- `m_bytes_completed`: Accumulated bytes transferred
- `agu_state`: AGU state for address generation
- `m_mf_issued_count`: Number of `mem_fetch` requests issued
- `m_mf_received_count`: Number of `mem_fetch` responses received (for debugging)

**Lifecycle**: Created when warp reaches TMA instruction, active during transfer, destroyed when all bytes complete and mbarrier notified.

---

## Known Limitations

1. **CP.ASYNC sector masking**: Corner cases with non-cacheline-aligned sizes not fully handled
2. **Tensormap options**: Some tensormap manipulation options not fully validated
3. **Tensor reduction timing**: Functional semantics are implemented, but the
   `UTMAREDG` atomic timing path is not calibrated

**Multi-dimensional testing**: Full test coverage for 1D and 3D-5D tensor
operations is implemented in
[`tma_multidim_test.cu`](../../../tests/src/tma/tma_multidim_test.cu)
(all tests passing as of issue #31).

---

## PTX ISA Reference

From PTX ISA 9.1, Section 9.7.9.25:
- `cp.async.bulk`: Asynchronous bulk data movement
- `cp.async.bulk.tensor`: Tensor memory operations using TMA
- `tensormap.replace`: Modify tensormap descriptor fields
- PTX ISA documentation: `docs/ptx_isa_9.1.pdf` section 9.7.9.25.4

---

## Integration

**Caller**: `ptx_thread_info::ptx_exec_inst()` dispatches to `handle_tma_inst()` when opcode is `CP_ASYNC_BULK_OP` or related.

**Dependencies**:
- `tensormap_descriptor_t`: Descriptor format (defined in tma.h)
- `barrier_set_t`: Mbarrier implementation for completion notification
- `mem_fetch_interface`: Interconnect interface for memory requests
- `memory_space`: Global and shared memory simulation
- L2 cache subsystem: Handles sector subdivision of large TMA requests

**Trace Output**: Uses `GPPRINTF_INST_EXEC(TMA, ...)` and `GPPRINTF_TMA(TMA, ...)` for debug logging.
