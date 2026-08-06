# `mbarrier` Implementation

This document describes FlashGPU-Sim's implementation of PTX memory barriers.
Refer to the
[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
for the normative instruction syntax and behavior.

## Scope

The implementation models CTA-scoped memory barriers used for asynchronous
producer-consumer synchronization, including coordination with Tensor Memory
Accelerator (TMA) transactions.

The functional instruction handler records each participating lane's barrier
operation. The timing model then applies the operation through
`barrier_set_t::warp_reaches_mbarrier()`, allowing waits and transaction
completion to affect warp scheduling.

## Barrier State

`mbarrier_manager_t` stores barriers by software CTA ID and shared-memory
address. Each barrier tracks:

- The expected and remaining arrival counts for the current phase.
- The number of outstanding asynchronous transaction bytes.
- A monotonically increasing phase whose low bit provides the parity value.
- The hardware warp IDs currently waiting for the phase to complete.

The manager also retains hardware CTA identity so that barriers can be removed
when a CTA completes and hardware CTA IDs are reused.

## Supported Operations

The modeled operations are:

- `mbarrier.init`
- `mbarrier.inval`
- `mbarrier.arrive`
- `mbarrier.expect_tx`
- `mbarrier.arrive.expect_tx`
- `mbarrier.complete_tx`
- `mbarrier.try_wait.parity`

The corresponding manager interfaces are declared in `mbarrier.h`:

```cpp
void init(gpgpu_sim *gpu, const thread_index_t &thread_index,
          uint64_t addr, int expected_count);
void inval(gpgpu_sim *gpu, const thread_index_t &thread_index,
           uint64_t addr);
std::set<int> arrive(gpgpu_sim *gpu,
                     const thread_index_t &thread_index,
                     uint64_t addr, int arrival_count);
void expect_tx(gpgpu_sim *gpu, const thread_index_t &thread_index,
               uint64_t addr, int expected_tx_count);
std::set<int> complete_tx(gpgpu_sim *gpu,
                          const thread_index_t &thread_index,
                          uint64_t addr, int completed_tx_count);
bool try_wait(gpgpu_sim *gpu, const thread_index_t &thread_index,
              uint64_t addr, int parity);
```

## Phase and Completion Semantics

Initialization sets the remaining arrival count to the configured expected
count, the outstanding transaction count to zero, and the phase to zero.

`arrive` reduces the remaining arrival count. `expect_tx` increases the
outstanding transaction count, and `complete_tx` reduces it. A phase advances
only when both counts reach zero. At that point the manager:

1. Releases all warps waiting on the barrier.
2. Restores the remaining arrival count for the next phase.
3. Increments the phase.

`try_wait.parity` succeeds immediately when the requested parity differs from
the current phase parity. Otherwise, the timing model records the warp as
waiting until the phase advances.

For `mbarrier.arrive.expect_tx`, the timing path registers the expected
transaction count before applying the arrival so that the arrival cannot
prematurely complete the phase.

## Current Limitations

- Barrier state is stored in simulator-side data structures rather than read
  from and written to the modeled shared-memory contents.
- Only CTA-scoped barriers are supported; cluster-scoped synchronization is not
  modeled.
- Waiting is represented at warp granularity. If a participating lane blocks,
  the entire warp blocks.
- Only the parity form of `mbarrier.try_wait` is supported. An optional timeout
  operand is accepted but its timeout behavior is not modeled.
- Unsupported `mbarrier` variants fail explicitly instead of being treated as
  no-ops.

## Configuration and Tests

The timing model exposes:

- `-gpgpu_mbarrier_arrive_latency`
- `-gpgpu_mbarrier_trywait_latency`

Integration coverage is located in:

- `tests/src/barrier/mbarrier_sanity_test.cu`
- `tests/src/barrier/mbarrier_test.cu`
- `tests/src/tma/tma_test.cu`
- `tests/src/tma/tma_multidim_test.cu`

Latency-focused tests and usage notes are under
`tests/src/microbench/mbarrier/`.
