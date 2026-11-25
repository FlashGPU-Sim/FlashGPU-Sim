# Blackwell WS B collectors

This private extension adds explicit `.collector::b0..b3::{fill,use,lastuse,discard}`
to the existing single-CTA F16/BF16 `tcgen05.mma.ws` path. Default WS continues
to mean buffer 0 discard. It does not enable Rubin ordinary-MMA B collectors.

## Common state and adapters

`tcgen05/collector.h` owns four logical B slots per issuing thread. The shared
TMEM manager places those banks in a CTA scope and clears them on CTA retirement
or reset. Each slot holds raw logical half/BF16 values, tagged with N, K, input
format, transpose layout and maximum column shift. Negation and accumulation remain in the existing
common MMA arithmetic, rather than being applied to cached B values.

Fill reads B and replaces the selected slot. Use reads the retained values;
lastuse reads them and invalidates the slot. Discard reads the current B and
invalidates the selected slot. Missing/incompatible reuse fails explicitly.
Other slots, issuing threads and CTA scopes remain independent.

PTX translates the numbered collector option into this common control object.
CUDA 13.3 SM100 SASS represents it with `B_KEEP`, `B_REUSE` and `BUFFER1..3` on
the B descriptor. The decoder accepts that spelling only on the B operand of
`UTCHMMA.WS`; ordinary MMA is not silently extended. Descriptor register-pair
dependencies retain the existing timing projection. The native adapter uses the
same backend bank operations as PTX and omits shared B reads on reuse.

## Numerical acceptance

`sm100-ws-collectors` runs the shared-state/decoder checks and eleven complete
numerical launches through both frontends. The CUDA fixtures cover FP16/BF16,
shared/TMEM A, N64/N128/N256, all four slots and all four control operations.
Each launch uses two CTAs and repeats the sequence twice. TMEM-A fixtures issue
MMA from warp 1, whereas shared-A fixtures use warp 0.

The fixture fills all four slots with different B matrices, waits for completion,
then changes every B value in shared memory before use/lastuse/discard. Every
output is compared to an independent host calculation. Reloading B on reuse,
aliasing slots, or retaining stale data across refills produces different
results. Inputs and results are exactly representable in the tested arithmetic;
this is not a claim about GPU bit-exact reduction order in general.

Component tests cover missing fills, mismatched shape/type/layout, invalid
controls, invalidation, owner isolation and CTA retirement/reset. The launch
runner freezes its executable before running either frontend so recompilation
cannot replace cubins partway through a multi-kernel test.

## Boundaries

This extension qualifies existing M64, K16 F16/BF16 WS shapes. Other WS types,
M32/M128 layouts, sparse MMA and cross-thread collector
handoff are not qualified here. Invalid extra forms remain rejected.

The subsequent [WS mask qualification](blackwell-ws-mask-qualification.md)
adds nonzero B zero-column masks and 8/16/32-column maximum shifts to the
same M64 subset. Its five additional launches change masks and shifts between
fill/use/lastuse/discard; masking never modifies the saved collector bank.

The common timing backend continues to charge the configured MMA work/recurrence.
Native functional shared-read metadata reflects reuse, but this extraction does
not claim calibrated collector latency, capacity/bandwidth contention, shared
memory contention, or a measured speedup from collector reuse. Production presets
are unchanged. Local simulator acceptance is separate from native-GPU testing
and full DeepInfer acceptance.

Reference: [NVIDIA PTX tcgen05.mma.ws](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-ws).
