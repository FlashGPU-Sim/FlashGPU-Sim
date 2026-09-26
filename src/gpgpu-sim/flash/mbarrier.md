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
`barrier_set_t::warp_reaches_mbarrier()`. A `try_wait` that cannot return
immediately creates one behavioral pending-wait record and blocks only its
warp; it does not create simulator instructions or opcode-level retry events.

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
bool test_wait(gpgpu_sim *gpu, const thread_index_t &thread_index,
               uint64_t addr, int parity) const;
void register_wait(const thread_index_t &thread_index, uint64_t addr);
void cancel_wait(int sw_cta_id, uint64_t addr, int hw_warp_id);
```

## Phase and Completion Semantics

Initialization sets the remaining arrival count to the configured expected
count, the outstanding transaction count to zero, and the phase to zero.

`arrive` reduces the remaining arrival count. `expect_tx` increases the
outstanding transaction count, and `complete_tx` reduces it. A phase advances
only when both counts reach zero. At that point the manager:

1. Restores the remaining arrival count for the next phase.
2. Increments the phase.
3. Notifies all registered pending waits that the phase may have changed.

`try_wait.parity` succeeds immediately when the requested parity differs from
the current phase parity. Otherwise, the timing model records the warp as
sleeping until either the requested phase completes or its deterministic
deadline is reached. A phase notification advances the pending wait's wake
cycle so the cycle hook performs an authoritative recheck; the notification
does not write the predicate directly.

At a scheduled recheck cycle, the state machine uses this fixed order:

1. Query each unresolved lane's authoritative phase as visible at the start of
   the recheck stage.
2. If complete, resolve that lane to `true`.
3. Otherwise, if that lane's inclusive deadline has been reached, resolve it
   to `false`.
4. Otherwise, keep that lane unresolved, re-register it for a later phase
   transition, and retain its original deadline.

The warp resumes only after every participating lane has a final result; all
active destination predicates are then committed together. Lanes may use
different CTA-shared barrier addresses, parities, or hints. This retains the
repository's thread-level mbarrier behavior without allowing individual lanes
to run independently of their warp.

The barrier cycle hook runs after TMA and TCGen05 completion processing and
before ordinary instruction issue. Consequently, completion already visible
to that hook wins over a same-cycle timeout. An arrive issued later in the
same simulator cycle does not retroactively alter an already committed result.
Predicate writeback and warp release have the pending record as their single
owner; the functional handler never writes a provisional predicate.

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
- Both the three-operand and four-operand parity forms are supported. The
  fourth PTX operand is `suspendTimeHint`, an unsigned nanosecond hint; it is
  not an architectural promise that hardware returns at an exact time.
- The simulator deliberately chooses a deterministic deadline policy for
  reproducibility. An explicit hint is converted with
  `ceil(hint_ns * core_frequency_hz / 1e9)`; an explicit zero returns false
  immediately when the initial query fails. This is modeled policy, not a
  claim about a particular hardware lowering.
- Unsupported `mbarrier` variants fail explicitly instead of being treated as
  no-ops.

## Configuration and Tests

The timing model exposes:

- `-gpgpu_mbarrier_arrive_latency`
- `-gpgpu_mbarrier_trywait_latency`: no-hint maximum suspension in core cycles;
  zero commits false immediately after a failed initial query.
- `-gpgpu_mbarrier_phase_wakeup_latency`: additional delay in core cycles after
  a phase notification triggers an authoritative recheck that resolves every
  active lane true. It applies only to a suspended `try_wait`; an initially
  ready query and a timeout-false result are unchanged.

An explicit `suspendTimeHint` replaces the no-hint bound after nanosecond to
core-cycle conversion. The pending wait performs no periodic polling. It is
rechecked only after a relevant phase notification or when its deadline is
due; these behavioral rechecks consume no scheduler issue slot and represent
no synthetic instruction.

The SM90, SM100, and SM120 architecture configs currently use a provisional
32-cycle no-hint bound. This keeps the three-operand operation on the bounded
suspension and final-query path while architecture-specific calibration is
refined.

The B200 single-call matrix measured the complete and incomplete three-operand
paths at the same 26 net cycles, and all 612 delayed-producer samples returned
false before arrival. The target SASS contained one `.TRYWAIT` operation with
no explicit `NANOSLEEP` or second phase check. These observations do not expose
the internal bounded-wait duration of `.TRYWAIT`, so they do not calibrate the
no-hint bound to zero. Hinted waits use the deterministic conversion policy
above. Native hinted timing depends on barrier completion and scheduler
activity, so the converted deadline remains a reproducible simulator policy.

The aggregate, warp-instruction-level counters printed under
`MBarrier Try-Wait Timing` are `logical_trywait`, `immediate_true`,
`suspended_waits`, `rechecks`, `true_after_suspend`, `timeout_false`, and
`sleep_cycles`. `phase_wakeups` and `phase_wakeup_cycles` report the number and
configured aggregate cost of notification-delayed successful wakeups.
`immediate_true` and `true_after_suspend` require every active
lane to return true; a logical wait with any timeout-false lane is classified
under `timeout_false`. `rechecks` counts event/deadline-driven warp-level
recheck stages (a stage may query multiple active lanes), and sleep cycles are
counted once per logical warp instruction, not once per lane. Set
`FLASHGPU_SIM_MBARRIER_TRACE=1` for
per-transition and per-lane result records. Trace is disabled by default.

Integration coverage is located in:

- `tests/src/barrier/mbarrier_sanity_test.cu`
- `tests/src/barrier/mbarrier_test.cu`
- `tests/src/unit/mbarrier_retry_timing_test.cc`
- `tests/src/tma/tma_test.cu`
- `tests/src/tma/tma_multidim_test.cu`
