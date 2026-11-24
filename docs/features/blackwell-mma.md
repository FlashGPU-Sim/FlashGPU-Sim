# Blackwell MMA extraction (private draft)

`feature/blackwell-mma` extends the independently qualified TMEM feature. It
moves arithmetic, operand layouts and scale/accumulator storage together with
PTX/SASS adapters and complete CUDA launch fixtures. No public PR is open.

## Ownership

- `tcgen05/descriptor.*`, `mma.*`, `operands.h`, and `tmem.*` own descriptor
  interpretation, packed operand traversal, reductions, scale reads and
  physical accumulator layouts. Both frontends use these implementations.
- `tcgen05/ptx_mma.cc` adapts PTX operands; `ptx_internal.h` shares operand access
  helpers with the other TCGen05 operations.
- Native `tcgen_mma_dense.cc` and `tcgen_mma_scaled.cc` adapt native operands.
  `tmem_internal.h` shares address wrappers with the TMEM adapter.
- Both frontends publish work and completion into the existing TCGen05 timing
  engine. Native descriptor pairs and implicit SFA/SFB registers are dependency
  inputs. No timing constants or production configs change.

The common arithmetic retains shapes needed by the subsequent paired-CTA and
Rubin layers. Their native entry points, collector state and peer scheduling
are not enabled by this branch. Functional shared-read metadata does not imply
new shared bandwidth or recurrence calibration.

## Complete-launch gate

The same executable embeds PTX and sm_100a cubins. Each launch synchronizes,
copies results back, and checks every output against a host reference. The
`sm100-mma` local gate runs 67 component tests and 23 numerical tests / 49
launches through each frontend:

| Family | Tests / launches per frontend | Checks |
|---|---:|---|
| Basic dense and scaled MMA | 8 / 12 | Shared FP16; padded FP4 x FP8; compact FP4 block32 with unequal scales and both selectors; TF32 TMEM-A M64/M128; nonuniform FP16/BF16 and E4M3/E5M2, negation and accumulation |
| Default-discard WS | 4 / 8 | FP16 M64 N64/N128/N256, shared-A and TMEM-A; BF16 with both A storage kinds; two accumulated MMAs |
| Dense layouts and output forms | 6 / 16 | Ordinary FP16/BF16 TMEM-A M64/M128, both M64 datapath origins; F16 output in shared-A/TMEM-A/WS; A/B transpose combinations and negation |
| Shared swizzles | 5 / 13 | K-major/MN-major 32/64/128B; starts offset within the swizzle period; shared-A/TMEM-A/WS; MN-major 128B with 32-byte atoms |

TMEM-A fixtures populate different values in the physical halves, exposing
incorrect continuous-row or folded mappings. Layout F uses sixteen datapaths
per warp partition, with matching A/D origins at lane 0 or 16. F16 D occupies
the low half of each 32-bit TMEM column; its fixtures check the chosen
per-instruction RN model, not GPU bit-exact reduction order.

`mma_test_layout.h` constructs test input images independently of simulator
helpers; expected products use logical matrices. MN-major 128B/32B-atom layout
uses four K rows and `Swizzle<2,5,2>`. That encoding is invalid for K-major.
Both frontends use one raw MMA shared-descriptor gate, including Blackwell
reserved bits, fixed fields and layout restrictions. Generic TMEM copy
decoding remains unchanged.

Component tests additionally cover mixed-format arithmetic pairs, narrow
conversions, scale-byte selection, malformed descriptors and physical layouts.
They do not imply complete-launch acceptance for every ISA modifier.

The inherited `sm100-tmem` gate retains its separate 12 tests / 14 launches.
Both gates use `tests/ci/tcgen05_dual.py`; MMA permits 600 seconds per frontend.
Scratch configs disable PTX guided reordering for inline-PTX fixtures and select
an explicit frontend. Shipped configs are unchanged. Local CI success is not
hosted CI or real-GPU qualification.

## Remaining qualification

- Non-default WS collectors/masks, other WS types and scaled forms beyond the
  recorded matrix; unqualified extra `scale-input-d` operands are rejected.
- Remaining modifier gates, nonzero pattern-base-offset fields and absolute
  leading-address forms. The shifted-start tests above use base-offset field 0.
- Paired CTA as a functional unit, followed by CLC and Rubin; then retained
  DeepInfer numerical acceptance on the assembled branches. Historical private
  stack results do not qualify this new extraction.

The F32-to-F16 RN conversion correction belongs with the added F16-output path:
at public base `b3378cb9`, its only callers are unit fixtures and runtime MMA
supports F32 output only. Other conversion implementations were not changed.

References: [NVIDIA PTX TCGen05](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions)
and [CUTLASS UMMA descriptors](https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm100_desc.hpp).
