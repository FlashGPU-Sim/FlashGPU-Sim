# Experimental Blackwell gather4

Private branch: `feature/tma-gather4`, directly based on public `01bdef86`.
It does **not** depend on `feature/sass-scalar-coverage`. SASS stays experimental
and opt-in; default configurations are unchanged. No public PR is created.

## Instruction and descriptor coverage

| Dimension | Implemented scope |
| --- | --- |
| PTX load | `cp.async.bulk.tensor.2d.tile::gather4`, global to shared CTA/cluster, byte-count mbarrier completion |
| CTA group | Default/explicit `cta_group::1`, `cta_group::2`; data destination and completion owner are separate |
| Multicast | 16-bit cluster mask, including selective destinations and paired multicast; repeated completion owners receive one credit per copy |
| Cache hint | Optional `L2::cache_hint`, also combined with multicast and CTA group 2 |
| Prefetch | `cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4`; accepted as a nonbinding hint, without modeled cache warming |
| Coordinates | `{x,y0,y1,y2,y3}`, reordered/repeated rows, signed coordinates and out-of-bounds fill |
| Tensor shape | Rank 2, `boxDim[1]=1`, no interleave; traversal stride 1..8 in dimension 1 (one selected row per slot) |
| Raw formats | U8/U16/U32/S32/U64/S64, F16/F32/F64/BF16, F32_FTZ, TF32, TF32_FTZ |
| Packed formats | `16U4_ALIGN8B`, `16U4_ALIGN16B`, `16U6_ALIGN16B`; physical global bytes and expanded shared layout are distinct |
| Swizzle | None, 32B, 64B, 128B, 128B atom-32, atom-32 flip-8, atom-64, subject to the format-specific legal combinations |
| OOB fill | Zero; floating NaN fill; TF32 conversion and FTZ are handled in the shared payload helper |
| Native SASS | SM100/SM103 `UTMALDG.2D.GATHER4`, `.2CTA`, `.MULTICAST`, `.MULTICAST.2CTA`, and `UTMAPF.L2.2D.GATHER4` |

The same payload/conversion/swizzle and completion-target helpers serve PTX
and SASS. Both use the existing common TMA AGU, queues, memory service, and
mbarrier timing. Native adapters decode ptxas' packed multicast mask and
CTA-rank address fields; they do not introduce a second timing engine.
Cluster launch, rendezvous, local shared address normalization, and retained
CTA state are included only as the foundation needed for these forms.
Timing completion locates resident CTAs independently of PTX shared-memory
pointers, so native cluster copies reach the common barrier manager.

## Boundaries

- No Rubin/SM107 overrides, validity-report modifiers, or wider multicast masks.
- Scatter4 is a separate global-store instruction, not a gather4 variant.
- Illegal descriptor combinations fail before payload writes. Packed-expanded
  loads require a 128-element inner box and the permitted swizzles/alignment.
- The simulator's explicit host tensor-map ABI and registered descriptors are
  supported. This does not claim decoding every opaque NVIDIA driver map.
- Newly added packed/extended descriptor layouts are qualified here for
  gather4; ordinary tensor load/store paths reject them rather than silently
  applying an incompatible layout. Other TMA features remain separate work.
- Cache hints are nonbinding; acceptance does not imply a calibrated cache
  eviction policy or prefetch performance model. Multicast performance retains
  the existing backend model, not a new hardware-calibrated network model.
- Validation below is compiler-generated simulation replay and host-reference
  comparison, **not a new hardware measurement**.

## Validation

The SM100 manifest includes four CUDA source files:

- `tma_gather4_test.cu`: repeated/permuted rows and column/row OOB cases.
- `tma_gather4_formats_test.cu`: 91 legal raw-format/swizzle combinations plus
  13 packed-format/swizzle combinations, nonzero shared swizzle phase,
  traversal stride, NaN/zero fill, prefetch and cache hints.
- `tma_gather4_cluster_test.cu`: six cluster launches covering multicast,
  paired multicast, remote unicast to rank zero, paired remote completion,
  local rank-one copies and selective multicast, with cache hints.
- `tma_ordinary_tensor_test.cu`: control regression ensuring ordinary tensor
  loads are not accidentally decoded as gather4.

The combined binary contains five GoogleTests and 114 kernel launches per
frontend, each checking output against a host reference. Component tests also
cover invalid descriptors, packed-byte AGU accounting, negative coordinates,
packet sizes, pair membership and duplicate completion credits. Build with
`make -C tests build-sm100-tma`; select `TmaGather4*.*:TmaOrdinaryTensorRegression.*`
when running the integration subset. Use the experimental SASS timing runner
with `FLASHGPU_SASS_ARCH=sm100`, or PTX replay of the embedded compute_100a code.

Workspace evidence: `runs/feature-expansion-20261121`. Final qualification is
recorded separately from intermediate debugging runs; see `qualification.json`.
The local CUDA 13.3 component run excludes only the pre-existing
`trace_validation_test.cc` legacy `cuCtxCreate` signature issue.

Reference: [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/),
`cp.async.bulk.tensor`, tiled gather4/scatter4, and tensor swizzling; also CUDA
13.3 `cuTensorMapEncodeTiled` format/alignment contracts. PTX specifies the
operation, while the native operand conventions are checked against CUDA 13.3
ptxas/nvdisasm output retained with the test evidence.
