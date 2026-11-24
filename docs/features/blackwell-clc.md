# Blackwell CLC extraction (private draft)

This feature builds on the qualified TMEM/MMA/paired-CTA stack. Both PTX
`clusterlaunchcontrol.try_cancel` and SASS `UGETNEXTWORKID` submit to the same
service in `flash/clc.cc`. There is no second native launch queue.

## Service and adapter boundaries

Cancellation consumes the ordinary kernel admission cursor, in the serial GPU
work-distributor phase. Arbitration order is physical core, logical CTA, then
warp. A successful request removes an entire unstarted singleton or x-adjacent
CTA pair. Partial admission and exhaustion leave the cursor unchanged.

The shared service writes one opaque 16-byte response and completes 16 bytes
on the destination mbarrier. Broadcast delivers to the requesting cluster's
live peers. The native adapter records a request effect and later receives
barrier completion; it does not invent an immediate successful response.
Requests retain CTA generation and delay issuer retirement until delivery.

`-gpgpu_clc_latency` is an uncalibrated delay from serial arbitration to response
delivery. Zero still delivers on a later cycle. No production preset is changed.
The response bit layout follows the retained CUDA 13.3 SM100 lowering, not a
portable PTX ABI. Native barrier storage remains keyed by local offset; only
transport/import metadata carries the owner rank.

## Prerequisites found during extraction

The public-derived base recognizes legacy `.bb128`, but not scalar `.b128`.
CLC requires a scalar 128-bit response register, so this extraction adds the
lexer alias and preserves all four words across register assignment and shared
loads/stores. Both 4x32-bit and 2x64-bit mov packing are checked independently of
response decoding. The change is an added PTX data-type form, not a second CLC
response implementation.

The work-count validator lowers to native `REDG.E.ADD.STRONG.GPU`. Its discarded
result form reuses the existing 32-bit `ATOMG.ADD` RMW implementation, lock and
memory timing proxy. This instruction extension is independent of CLC. Its
unit case checks signed wrapping, multiple lanes and source-only dependencies;
no new atomic bandwidth calibration is implied.

## Acceptance matrix

`tests/src/clc/launch_queue_test.cu` exercises actual cancellation, verifies every
logical work item is processed exactly once, and compares scalar/vector query
coordinates. It includes:

- Singleton and paired broadcast, successful cancellation and exhausted queue.
- Shared-CTA and generic address forms, expect-tx before/after issue, repeated
  phases and repeated launches using reclaimed CTA slots.

The separate `tests/src/clc/b128_test.cu` checks scalar `.b128` memory roundtrip
and both vector register packing widths.

The `sm100-clc` local CI job selects queue/adapter components and executes the
same embedded-PTX/sm_100a binary through both frontends. It expects seven tests
and thirteen launches per frontend. Tests use an isolated 2-GPC x 4-SM config
with DSM; SASS remains opt-in. Validation results are recorded separately in
workspace `runs/CURRENT.json`.

The 2026-11-25 local `sm100-clc` job passed 10 component tests and all seven
numerical tests / thirteen launches per frontend. Identical binary/runtime
snapshots produced identical cycles with one and two host threads. Delays of
37 cycles passed the full suite; an explicit 512-cycle exhausted-queue probe
increased completion time on both frontends while retaining correct failure
responses. The existing four paired MMA tests / fourteen launches per frontend
also passed with cycles identical to the parent branch. These are simulator
acceptance results, not hardware latency measurements.

## Remaining qualification

General cluster shapes, SM120/Rubin native lowering, calibrated arbitration/
response costs and assembled DeepInfer acceptance remain outside this gate.
The current cursor supports the retained singleton and 2x1x1 cases; it does not
claim cancellation of arbitrary non-contiguous clusters. Existing real-world
private-stack results cannot substitute for replay on this extracted stack.

The NVIDIA specification requires an asynchronous response and mbarrier byte
completion; multicast targets each requesting CTA's corresponding shared
location. A CTA must not issue another request after observing failure, and
broadcast with an exited peer is undefined. Fixtures follow those rules.
See [PTX Cluster Launch Control](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-clusterlaunchcontrol-try-cancel).
