# mbarrier Support

FlashGPU-Sim models CTA-scoped shared-memory barriers for asynchronous
producer/consumer synchronization, including TMA and TCGen05 completion.

## Operations and Phases

Supported operations are `mbarrier.init`, `inval`, `arrive`, `expect_tx`,
`arrive.expect_tx`, `complete_tx` and `try_wait.parity`.

A barrier tracks expected/remaining arrivals, outstanding transaction bytes
and a phase. Initialization starts at phase zero. Arrivals reduce the remaining
arrival count; `expect_tx` adds transaction bytes and `complete_tx` removes them.
The phase advances when both counts reach zero, and the arrival count is reset
for the next phase. `arrive.expect_tx` adds the expected bytes before applying
the arrival, preventing premature phase completion.

`try_wait.parity` succeeds when the requested parity differs from the current
phase parity. Both three-operand and four-operand forms are supported. The
fourth operand, `suspendTimeHint`, is an unsigned nanosecond hint.

## Waiting Model

A waiting lane suspends its entire warp. The warp resumes after every active
lane has a final result; destination predicates become visible together. Lanes
may specify different CTA-shared addresses, parities or hints.

The simulator uses deterministic deadlines for reproducibility. An explicit
hint is converted to core cycles with
`ceil(hint_ns * core_frequency_hz / 1e9)`. An explicit zero returns false
immediately after a failed initial query. This deadline policy is not a claim
that hardware returns at the exact converted time.

Pending waits are rechecked on relevant phase notifications or at their
deadline, without periodic polling or synthetic retry instructions. At a
recheck, a completed phase wins over a timeout visible in the same stage.
An arrival issued later in the simulator cycle cannot change a committed result.

A nonzero minimum predicate-ready latency latches a no-hint call's initial
query result and delays visibility. A phase completion during this delay does
not change the latched result. Notification wakeup latency separately delays
successful completion of an already suspended wait.

## Timing Controls

| Option | Meaning |
| --- | --- |
| `-gpgpu_mbarrier_arrive_latency` | Arrival latency |
| `-gpgpu_mbarrier_trywait_latency` | No-hint maximum suspension in core cycles; zero resolves a failed initial query immediately |
| `-gpgpu_mbarrier_predicate_latency` | Minimum no-hint issue-to-predicate-ready latency; zero permits an immediate query result |
| `-gpgpu_mbarrier_phase_wakeup_latency` | Additional latency when a notification-driven recheck resolves all active lanes true |

An explicit hint replaces the no-hint suspension bound. The supplied SM90 and
SM100 configurations use a provisional 32-cycle no-hint bound and zero minimum
predicate-ready latency. SM120 uses 34 cycles for both, reflecting its modeled
initial-result latching behavior. These values do not establish a universal
hardware wait duration.

See the [configuration guide](../../../configs/README.md#data-movement-and-synchronization)
for defaults and architecture settings.

## Limits and Diagnostics

- Only CTA-scoped barriers are modeled; cluster-scoped synchronization is outside
  the supported scope.
- Barrier state is tracked separately from modeled shared-memory contents.
- Suspension is warp-granular, including when only one active lane must wait.
- Unsupported instruction variants fail explicitly.

The `MBarrier Try-Wait Timing` report counts logical warp instructions:
`logical_trywait`, `immediate_true`, `suspended_waits`, `rechecks`,
`true_after_suspend`, `timeout_false` and `sleep_cycles`. A wait is classified
as successful only if all active lanes return true; any timeout-false lane
classifies the instruction as `timeout_false`. Rechecks count warp-level stages
and sleep cycles are counted once per logical instruction.
`phase_wakeups` and `phase_wakeup_cycles` report notification-delayed successful
wakeups and their aggregate configured cost.

Enable the `MBAR` component for transition and per-lane results; see
[runtime tracing](../../../docs/tracing.md).

## Validate

From the repository root:

```bash
./tests/run_tests.py run --arch sm120 --group barrier
./tests/run_tests.py run --arch sm100 --group barrier
```

The barrier and TMA integration groups exercise phase transitions and
asynchronous completion. The unit suite includes
[`mbarrier_retry_timing_test.cc`](../../../tests/src/unit/mbarrier_retry_timing_test.cc)
for deterministic wait timing.
