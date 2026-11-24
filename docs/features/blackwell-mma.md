# Blackwell MMA extraction (private draft)

`feature/blackwell-mma` extends the independently qualified TMEM feature. It
moves math, operand layouts and scale/accumulator storage together with PTX and
SASS adapters and complete CUDA launch fixtures. It is not a public PR yet.

## Ownership

- `tcgen05/descriptor.*`, `mma.*`, `operands.h`, and `tmem.*` own descriptor
  interpretation, packed operand traversal, numerical reductions, scale reads
  and physical accumulator layouts. PTX and SASS use these implementations.
- `tcgen05/ptx_mma.cc` adapts PTX operands. `ptx_internal.h` shares the existing
  PTX operand access helpers with the other TCGen05 operations.
- `sass/functional/tcgen_mma_dense.cc` and `tcgen_mma_scaled.cc` adapt native
  dense and scaled operands. Native address wrappers are shared with TMEM in
  `tmem_internal.h`, not copied into each adapter.
- Both frontends publish work and completion into the existing TCGen05 timing
  engine. Native descriptor pairs and the implicit SFA/SFB register pair are
  all dependency inputs. No timing constants or production configs change.

The common arithmetic helpers retain generic matrix shapes needed by the next
paired-CTA/Rubin layers, but those native instruction entry points, collector
state and peer scheduling are not enabled here. Shared-read records are kept
as functional metadata; this extraction does not claim new bandwidth or
recurrence calibration.

## Current launch evidence

The same executable retains PTX and sm_100a cubins. Host references check every
output after launch, synchronize and copy-back. The first expanded run passed
20 tests / 26 launches on each frontend: the inherited 12 TMEM tests / 14
launches, plus these 8 MMA tests / 12 launches:

| Cases | What is checked |
|---|---|
| Shared FP16 | MMA, commit, barrier wait, all four warp TMEM partitions |
| Block-scaled FP4 x FP8 | Padded operand containers, scale reads, exact output |
| Compact FP4 block32 | Both scale selectors, unequal scales, two accumulated MMAs |
| TF32 TMEM-A | M64/M128 physical partitions, two accumulated MMAs |
| Dense FP16 and BF16 | M64/M128, two CTAs, nonuniform A/B, negation, accumulation |
| Dense E4M3 and E5M2 | Nonuniform operands, two CTAs, negation, accumulation |

Component coverage includes all ordered mixed-format arithmetic pairs, narrow
conversions, swizzled shared layouts, scale-byte selection and accumulator
mapping. This is broader than the complete-launch matrix; it is not a claim
that every ISA modifier/shape is qualified end to end.

The WS follow-up adds four numerical tests / eight launches per frontend:
FP16 M64 N64/N128/N256 with either shared-A or TMEM-A, and BF16 with both A
storage kinds. Two successive MMAs test accumulation. The TMEM-A fixtures
intentionally populate different values in the upper/lower physical halves,
then check every output against an independent host reference. CUDA 13.3
emits `UTCHMMA.WS` for these fixtures. The expanded MMA CI gate contains
12 numerical tests / 20 launches per frontend and 64 component tests.

WS uses the default collector-discard form. Explicit collector fill/use,
zero-column masks, WS TF32/FP8 and integer WS are not qualified by this gate.
PTX and SASS share descriptor validation, folded accumulator addressing and
the physical TMEM-A arithmetic helper. No duplicate WS math implementation
was introduced.

The dense-forms follow-up adds six tests / sixteen launches: ordinary
FP16/BF16 TMEM-A at M64/M128, both M64 datapath origins, F16 output in shared-A,
TMEM-A and WS forms, and A/B transpose combinations with negation. Layout F
uses sixteen datapaths per warp partition; TMEM A and D must use the same
origin (0 or 16). Both adapters use the common alignment gate and physical-A
mapper. F16-output fixtures load the low half of each 32-bit TMEM column and
check two accumulated MMAs against the selected per-instruction RN model.
This does not claim GPU bit-exact reduction order.

The expanded local gate comprises 65 component tests and 18 numerical tests /
36 launches per frontend. Unsupported `scale-input-d` operands are rejected
rather than silently ignored. Extra scale/mask/collector variants remain
separate qualification work.

A component fixture was updated from K-by-N B storage to the common backend's
non-transposed N-by-K representation. Its expected results were not relaxed.
The complete-launch nonuniform F16/BF16 cases independently check that contract.

`sm100-mma` runs the component gate followed by the same binary through both
frontends. `sm100-tmem` retains its separate filter and launch count; both reuse
`tests/ci/tcgen05_dual.py`. Scratch configs disable PTX guided reordering for
inline-PTX fixtures and explicitly select each frontend. Shipped configs are
unchanged. Local CI evidence must not be described as hosted CI success.

## Remaining before public readiness

- Audit and qualify the remaining native/PTX asymmetries, particularly non-default WS,
  shared-memory swizzle variants and scaled forms beyond the recorded matrix. Native registration is not numerical acceptance.
- Complete the malformed descriptor/modifier audit. The native MMA adapter
  rejects swizzle mode 1 while generic PTX shared decoding accepts it; unify
  that MMA gate and qualify swizzled launches before public readiness.
- The RN conversion correction belongs with the added F16-output path: at public base `b3378cb9`, the helper is called only by
  unit fixtures and the runtime supports F32 output only. Other conversion
  implementations were not changed. This is not a public runtime regression.
- Qualify paired CTA in its own functional unit, then CLC and Rubin, and rerun
  the retained DeepInfer acceptance cases on the assembled stack.
- No new GPU timing measurements, full DeepInfer acceptance, private research
  configs, placement changes or public PR are implied by this draft.

Reference: [NVIDIA PTX ISA TCGen05](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions).
