# Elect Instruction Implementation

Implements the PTX `elect.sync` instruction for deterministic leader election within warp subsets.

## External Interface

### `handle_elect_inst()`

**Signature**:
```cpp
void handle_elect_inst(const ptx_instruction *pI, ptx_thread_info *thread);
```

**Purpose**: Executes the `elect.sync` instruction, electing a leader thread from a subset of active threads.

**PTX Syntax**:
```ptx
elect.sync d|p, membermask;
```

**Parameters**:
- `pI`: PTX instruction containing destination operand (vector d|p) and membership mask operand
- `thread`: Current thread context for execution

**Behavior**:
- Elects the **lowest numbered lane** in the membership mask as the leader
- **Deterministic**: Same input mask always elects same lane
- All participating threads receive the same elected lane ID in destination register `d`
- Only the elected thread receives predicate value `True` (0) in `p`; all others receive `False` (1)

**Outputs**:
- `d` (32-bit register): Lane ID of elected thread (same value broadcast to all threads)
- `p` (predicate register): `True` for elected thread, `False` for all others

**Example**:
```ptx
// Threads 0-7 are active (membermask = 0xFF)
elect.sync d0|%p0, 0xFF;
// Result: d0 = 0 (lane 0 elected)
//         %p0 = True for thread 0, False for threads 1-7
```

**Error Conditions**:
- None (always succeeds with valid membership mask)

## Internal Helpers

### Election Algorithm

**Algorithm**: Linear scan to find lowest set bit in membership mask

**Steps**:
1. Extract thread's lane ID (hw_tid % 32)
2. Read membership mask from source operand
3. Scan bits 0-31, elect first set bit
4. Compare thread's lane ID with elected lane
5. Set destination vector (d|p) with elected lane ID and predicate

**Complexity**: O(32) worst case (always scans up to 32 lanes)

**Invariants**:
- Elected lane is always the lowest numbered lane in membership mask
- All threads in warp receive identical `d` value
- Exactly one thread receives `p = True` (unless mask is empty)

## PTX ISA Reference

From PTX ISA 9.1, Section 9.7.13.6:

> **elect.sync** - Elect a single active thread in warp
>
> Returns the laneid of the elected thread in the destination register.
> The elected thread is the active thread with the lowest laneid.
> A predicate register can be optionally specified as the second destination
> to indicate whether the current thread is the elected thread.

## Integration

**Caller**: `ptx_thread_info::ptx_exec_inst()` dispatches to `handle_elect_inst()` when opcode is `ELECT_SYNC_OP`

**Dependencies**:
- `ptx_instruction`: Provides operand access
- `ptx_thread_info`: Provides thread context and register access

**Trace Output**: Uses `GPPRINTF_INST_EXEC(WIP, ...)` for debug logging
