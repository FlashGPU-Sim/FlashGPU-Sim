# WS B zero-column mask and shift qualification

The [2026-11-25 B200 capture](blackwell-ws-native-validation.md) supersedes
the historical hardware-pending statements below. It also corrects the
out-of-span start-count oracle. Earlier simulator-only gates remain recorded
with their original scope and counts.

This follow-up originated on `feature/blackwell-timing-qualification` and is
now owned by `feature/blackwell-ws-collectors`. It extends Blackwell
M64, K16, single-CTA FP16/BF16 WS MMA. It does not enable Rubin collectors or
claim support for every WS datatype and layout. Public PRs remain unsubmitted.

## Common implementation

`tcgen05/ws_mask.h` decodes the 64-bit B descriptor and applies its column
mask and shift. Both PTX and SASS adapters use it after the common collector
operation and before the existing common arithmetic. Collector banks retain
the raw, unmasked N + maximum-shift columns. A different mask or shift on
use/lastuse therefore observes the original B values, even after a fill whose
mask zeroed every output column. The cached signature includes maximum shift.

Instruction-descriptor bits 30–31 select maximum shift 0/8/16/32 only for WS.
Ordinary MMA still rejects those bits. Reserved mask bits and out-of-window
shifts fail before a collector can be changed. Native mask dependencies cover
both uniform registers; a truncated register pair is rejected.

The [PTX zero-column descriptor and WS instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-ws)
describe per-segment start counts and alternating spans. The field table's
Skip Span / Use Span prose disagrees with its worked examples. This code uses
Skip Span = zeros and Use Span = original B, consistent with those examples
and [CUTLASS MaskAndShiftB and its generator](https://github.com/NVIDIA/cutlass/blob/56f0718a977454920a70b415343531e979ebf1ba/include/cute/arch/mma_sm100_desc.hpp).
Bit 39 enables zeroing when set, as specified by PTX and used by that generator;
CUTLASS's isolated `nzm_` field comment has the opposite polarity.

## Numerical coverage and regression

Evidence is `runs/tcgen-ws-mask-20261125` at the workspace root. The new
zero-mask fixture fails on the previous runtime in both frontends, then passes
with this implementation. All eleven WS launches pass PTX and native SASS:
the six existing collector cases plus five mask/shift cases. Each launch
uses two CTAs, two rounds, four banks, all four collector operations, and
mutates shared B after fill. The independent host oracle walks alternating
runs instead of importing simulator mask/layout code.

Additional cases cover all-zero and disabled masks, separate segment start
counts, counts reaching 255, 8/16/32-column shift windows, transposed B,
FP16/BF16, shared-A/TMEM-A, and N64/N128/N256. Unit tests also check the
manual's M32/M128 mask examples; these helper checks do not qualify complete
M32/M128 WS MMA launches. The host gate contains 108 passing tests.

`collect.py` in the evidence directory verifies frozen fixture and runtime
hashes, launch counts, before/after reproduction, and unchanged cycles for
ordinary MMA and the original collector/paired-CTA fixtures. The CI WS gate
now requires eleven tests and eleven launches per frontend. See `report.json`
for the completed gates and exact evidence boundaries.

## Timing and ownership

Masking does not reduce modeled MMA work or recurrence. Fresh fill/discard
gathers the raw N + maximum-shift window; use/lastuse omits shared B reads.
Both adapters account for this same requested operand extent. Its physical
fetch granularity and collector cost have not been newly measured on hardware;
this is a functional window model, not a calibrated mask/shift speedup claim.
No production config or timing constant changes.

These changes belong to the WS feature and shared TCGen05 backend. They do
not establish another independent public bug. The four previously prepared
public-fix bundles remain separate. This qualification is not a full DeepInfer
sweep, a GPU measurement, or hosted CI. The functional fixes have now been redistributed to their owner branches;
exact-head gates are in `/home/wzr/flashgpu-feature-closeout-20261126`.

## Native-qualified start-count boundary

The owner branch includes the subsequent hardware-qualified clamp: a segment
start count beyond the active window is clamped before alternating spans are
walked. Both frontends and the independent fixture oracle use the corrected
semantics. The older 2026-11-25 report above predates this fix; the current
eleven-launch gate and host collector checks qualify the corrected version.
Shift-window timing remains a deferred calibration question, not a mask
correctness failure or a reason to enable the experimental shared-write model.

## Prepared B200 hardware probe

`tests/native/ws_mask/` builds a separate native numerical and timing package
with CUDA 13.3 and pinned CUTLASS SW128 layouts. It has eleven numerical cases
and a 176-case timing matrix (48 selected timing cases with NCU full). Shared-A,
TMEM-A, shifted B windows, masks changed after fill, transposed B, four collector
slots and wide N are included in the numerical fixture.

The locally built package passed all eleven numerical tests through each of
PTX and SASS. Four timing-fixture cases per frontend also passed their output
checks, including zero burst, discard, masked reuse and TMEM fill/lastuse.
These are simulator launch qualifications, not fresh hardware measurements.
The deployed collector checks native numerical results before timing; it uses
NCU base clock control and does not shut down the server.

Evidence: `runs/tcgen-final-integration-20261125/native-package.json` in the
experiment workspace. The separately compiled fixture exposed a public dynamic
shared alignment defect; its repair is prepared in the independent runtime-safety
bundle. The original inline-PTX CI fixture remains distinct from this package.
