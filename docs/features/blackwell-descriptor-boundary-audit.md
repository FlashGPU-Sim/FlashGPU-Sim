# Blackwell descriptor execution-boundary audit

This follows [timing gap closure](blackwell-timing-gap-closure.md). The remaining
small MMA/TMEM timing residuals are deferred at the user's request. This audit
changes functional rejection boundaries only, not timing constants or presets.

## Correctness fixes

1. **Nonzero shared swizzle pattern-base offsets:** the common MMA descriptor
   gate accepted bits 49–51, while the address helpers never consumed them.
   Dense FP8 SASS happened to reject these separately; the other adapters did
   not. The shared gate now rejects nonzero offsets for both frontends, before
   any accumulator write. This does not reject a shifted matrix start address
   with a zero pattern-base field, which existing numerical fixtures exercise.
   Implementing a nonzero pattern base remains separate work.
2. **Block-scaled eight-bit transpose in PTX:** the ISA decoder correctly
   recognizes a transposed FP8 operand, but the PTX mixed operand gather always
   reads padded K-major storage. The arithmetic then interprets that buffer as
   MN-major. SASS already rejected the descriptor. A common Blackwell mixed
   execution-subset predicate now gates both adapters before gathering. The
   general decoder/arithmetic retains its broader layouts for future support.
   This fix rejects unimplemented transpose; it does not implement transpose.

The distinction follows the [PTX shared descriptor and instruction descriptor
specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-matrix-descriptors):
a pattern-base field affects swizzle addressing, and a valid ISA descriptor alone
does not establish that this simulator's operand path implements its layout.

## Reproduction and validation

Evidence directory: `runs/tcgen-boundary-audit-20261125/`.

- The new pattern-base host regression fails against the previous frozen
  runtime, which accepts all 35 tested mode/base combinations. It passes with
  the fix, including decoder rejection.
- `transpose_probe.cu` is a standalone sm_100a PTX/cubin fixture: A is one,
  B has alternating one/two columns stored in MN-major order, UE8M0 scales are
  one, and the checked first output column is 32. The previous PTX runtime
  computes 16 instead. SASS rejects it. With the fix, both reject before MMA.
  These are expected-negative runs, not passing numerical launches or new
  native-hardware measurements; their raw runner reports remain failed.
- 42 host checks pass, including all existing arithmetic/descriptor checks in
  the selected files, the full supported mixed shape/format cross-product,
  transpose/reserved-bit exclusions, and four native MMA families rejecting
  the pattern base without modifying PC, accumulator or timing effects.
- The affected complete-launch gates run through PTX and SASS using one frozen
  fixture and runtime: MMA, paired CTA and WS collectors. `report.json` records
  their final counts and cycle comparisons against the previous qualified run.

No public PR is opened. These fixes belong with the Blackwell MMA feature,
not an independent public-base fix. The feature-owner branch refresh remains
separate from this private integration branch's qualification. Nonzero WS B
zero-column masks, other unqualified datatype/layout variants and implicit
MMA/TMEM contention are not implemented by this audit.
