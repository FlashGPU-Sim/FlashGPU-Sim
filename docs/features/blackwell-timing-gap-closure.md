# Blackwell TCGen05: mask, TF32 and TMEM timing closure

Integration follow-up: [dense FP8 output-mask qualification](blackwell-fp8-mask-qualification.md).
The original gate counts below describe the earlier checkpoint.

This continues [timing qualification](blackwell-timing-qualification.md).
Workspace evidence is `runs/tcgen-gap-closure-20261125/`; original measurements,
failed attempts and previous frozen runtimes are retained. This is a private
integration branch, not a claim that every SM100 instruction form is supported.

## Functional boundaries

Both PTX and SASS now use the common TMEM accumulator writer for nonzero dense
F16/BF16 output-lane masks. Bits select physical TMEM lanes, including the M64
spaced layout, paired M128 folded layout and separate four-word mask groups for
the two CTAs. A disabled lane preserves its previous D value even when
`enable-input-d` is false. Masking does not discount compute timing.

Dense single-CTA TF32 now accepts shared A as well as TMEM A, using the same
shared descriptor/address gather, arithmetic and accumulator writer. Qualified
forms are M64/M128, N8..256 in steps of eight, K8, K-major operands, F32 output.
Nonzero output-lane masks and 32/64/128-byte shared swizzles have numerical
fixtures. Transposed TF32, paired TF32 and WS TF32 remain outside this subset.

WS's optional 64-bit **B zero-column descriptor** is distinct from the dense
output-lane mask. Its explicitly zero value is now accepted by both adapters;
nonzero descriptors remain rejected. The complete implicit register span is
included in SASS dependencies: four/eight words for dense output masks, two
words for the WS descriptor. A malformed span is rejected before reading it.
The implementation follows the PTX [MMA mask semantics and WS descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-instructions-mma).

New coverage checks sentinel-preserving masked writes, distinct masks on each
CTA, unused physical lanes, shared/TMEM A, accumulator enable/disable and
runtime mask registers. The native TF32 adapter is a separate short file;
arithmetic and physical layouts remain in the common backend.

Archived ordinary M128/N64 BF16 and shared-A TF32 probes now run successfully,
alongside the previously working mixed FP4/FP8 and compact FP4 probes. For each
family, 128/256 bursts differ by 6144 cycles (48 cycles/MMA), matching the native
recurrence; all 8192 outputs per run pass the independent oracle. Their host
shared-cudart relinks retain the archived device text/constants/metadata.

## Ordinary TMEM timing: fixed at the timing origin

The ordinary 32dp read table was fitted as **execution-pipeline** latency. The
extracted async engine instead applied it at scheduler issue, retiring the
physical instruction after a one-cycle execution and extending only its
native write barrier. That lost operand/execute/writeback overhead.

When the B200 ordinary-load model is enabled, both frontends now assign the
common table to the real execution pipeline. The async engine registers the
operation at scheduler issue, but cannot complete it or release its wait until
the matching dynamic instruction retires. A tagged completion preserves warp
isolation and wait cutoffs; it does not use a fixed five-cycle correction.
Pipeline capacity is sized after parsing the table and invalid II/latency pairs
are rejected. Calibrated TMEM loads bypass the optional classic-MMA issue queue.
Other transfer forms and configurations with this option disabled retain their
previous path.

For x1, the issue trace shows operand-read to writeback changing from 4 to 14
cycles. The dependent consumer previously waited on an artificial deadline at
+11; it now consumes at +16. The loop slope consequently changes from 39 to 44.

| Words per lane | Native dependent-loop slope | Before | After |
|---|---:|---:|---:|
| x1 | 44 | 39 | 44 |
| x8 | 47 | 42 | 47 |
| x32 | 59 | 54 | 59 |
| x64 | 75 | 70 | 75 |

The full retained ordinary-read matrix has 38 cases and 76 runs: all numerical
checks pass; 35 slopes match exactly. The three x4/depth-four independent cases
(one, two or four warps) retain the historical 48 versus 47 cycles (+2.13%).
Thus this restores the old calibration across concurrency as well as dependent
loads; it does not establish new STTM/CP/16dp or implicit MMA/TMEM calibration.

## MMA residual: throughput and burst completion are separate

The shared-contention QK probe is **BF16 WS TMEM-A**, M64/N128/K16. Its recurrence
lookup succeeds. The PV probe is **BF16 WS shared-A**, M64/N256/K16, with B in
MN-major layout; that exact layout is excluded from the K/K-major lookup.
The previously reported pure-MMA errors cannot both be called missing table
entries, and neither is established as a shared-read bandwidth error.

Archived WS probes retain separate issue and completion timestamps. These
native device functions are replayed unchanged; shared-cudart relinks preserve
all 187 device text/constant/metadata sections in each of the two families.
The long-burst native comparison uses the retained one-owner/one-chain data.
No new hardware measurements were collected.

| QK-like WS TS M64/N128/K16 | Native issue / complete | Sim issue / complete |
|---|---:|---:|
| Empty burst | 2 / 62 | 2 / 33 |
| 1 MMA | 29 / 260 | 29 / 109 |
| 32 MMA | 1136 / 1618 | 1219 / 1474 |
| 128 MMA | 5351 / 5835 | 5434 / 5689 |
| 256 MMA | 10983 / 11467 | 11066 / 11321 |

The 128-to-256 slope is 44 cycles/MMA on both sides. At 128, simulated issuing is
83 cycles **slower**, while the post-issue interval is 255 versus 484 cycles,
229 cycles too short. The net 146-cycle deficit is a burst-boundary/completion
problem, not a reason to slow every MMA. The total interval alone cannot
identify which physical accumulator/collector/barrier stage contributes it.
The empty-burst deficit also shows that not all of it belongs to MMA itself.

| PV-shaped WS SS K/K-major control | Native issue / complete | Sim issue / complete |
|---|---:|---:|
| 32 MMA | 2089 / 2717 | 2199 / 2634 |
| 128 MMA | 9739 / 10365 | 9866 / 10301 |
| 256 MMA | 19979 / 20605 | 20106 / 20541 |

Both long slopes are 80 cycles/MMA. This control is **K/K-major**, unlike the
original PV contention probe's transposed B. It must not be substituted for an
exact-layout timing qualification of that original probe.

No global MMA recurrence, barrier latency or completion-tail constant is fitted
from these residuals in this change. Additional control runs are diagnostic
configuration overrides only, recorded separately in `controls.json`.

## Diagnostic controls and acceptance

For QK-like 128-MMA bursts, individual overrides produce:

| Configuration | Issue | Post-issue wait | Complete |
|---|---:|---:|---:|
| Native | 5351 | 484 | 5835 |
| Current model | 5434 | 255 | 5689 |
| Only barrier 32 → 44 | 5434 | 267 | 5701 |
| Only owner queue 5 → 7 | 5346 | 343 | 5689 |
| Only completion tail 0 → 146 | 5580 | 255 | 5835 |

Queue depth 7 nearly matches issuing but cannot fix total completion. A 146-cycle tail matches
completion exactly while making issuing worse: the current admission count
includes operations waiting for their completion tail. These controls constrain
the next model change to separating issue/backpressure and completion/notification;
a total-runtime fit alone would hide the wrong internal timeline. None of these
overrides is committed as the B200 default.

Final frozen runtime `runtime-r4/libcudart.so` SHA256:
`5949baea80f8261a3ebf68351d2c19eebf86fdb2437159baafe2a6c00b27b18b`.
Later source edits are formatting/documentation and additional host tests.

- 151 focused host component tests pass.
- Each frontend: 55 numerical tests, 103 launches, all pass (MMA 29/62,
  pair 6/17, TMEM 14/18, WS 6/6).
- Both frontends’ 17 paired launches have identical cycles at one/four threads.
- The unchanged two-test legacy executable retains exact pre-change cycles in
  both frontends with the new timing options disabled.
- Ordinary-read matrix: 76 runs pass; archived WS probe matrix: 10 runs pass.
- The ordinary BF16/TF32/FP4 burst checks used frozen `runtime-r1`; the final
  full frontend suite and TMEM/WS matrices use `runtime-r4`. Report versions
  explicitly rather than presenting every probe as one executable/runtime.

## Remaining limits

- WS nonzero zero-column masks, ordinary FP8 output-mask variants, sparse and
  other excluded instruction forms are not newly claimed here.
- Implicit MMA accumulator/A traffic versus LDTM/STTM remains uncalibrated;
  the retained native PV/STTM slowdown (~8.7%) still needs a resource model.
- Shared write-side STS/TMA competition and physical placement across CTAs are
  not completed by the existing read-side half-region model.
- This is focused TCGen05 qualification, not another full DeepInfer sweep.
- Feature owner branches and public PRs still need integration after review.

Follow-up: [descriptor execution-boundary audit](blackwell-descriptor-boundary-audit.md)
fixes silent acceptance of two unimplemented layouts; the small timing residuals
above are deferred.
