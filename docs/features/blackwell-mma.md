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

A component fixture was updated from K-by-N B storage to the common backend's
non-transposed N-by-K representation. Its expected results were not relaxed.
The complete-launch nonuniform F16/BF16 cases independently check that contract.

`sm100-mma` runs the component gate followed by the same binary through both
frontends. `sm100-tmem` retains its separate filter and launch count; both reuse
`tests/ci/tcgen05_dual.py`. Scratch configs disable PTX guided reordering for
inline-PTX fixtures and explicitly select each frontend. Shipped configs are
unchanged. Local CI evidence must not be described as hosted CI success.

## Remaining before public readiness

- Audit and qualify the remaining native/PTX asymmetries, particularly WS,
  TMEM-A half forms, transpose/negation combinations and scaled variants beyond
  the recorded matrix. Native registration is not numerical acceptance.
- Audit malformed descriptor/modifier gates and add rejection cases.
- Separate any generic fixes that should not depend on this feature. The old
  private stack includes an F32-to-F16 rounding correction that needs this
  review before publication.
- Qualify paired CTA in its own functional unit, then CLC and Rubin, and rerun
  the retained DeepInfer acceptance cases on the assembled stack.
- No new GPU timing measurements, full DeepInfer acceptance, private research
  configs, placement changes or public PR are implied by this draft.

Reference: [NVIDIA PTX ISA TCGen05](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions).
