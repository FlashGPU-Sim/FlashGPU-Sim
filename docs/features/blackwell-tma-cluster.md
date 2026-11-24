# Blackwell clustered tensor loads

This private feature extends ordinary tensor loads to the same CTA-target and
completion-mask machinery used by gather4. It is layered on the captured dense
tensor-map, CLC and paired-TCGen05 feature branches. No public PR is submitted.

## Semantics and implementation

PTX `cp.async.bulk.tensor` `.cta_group::2` may notify a barrier in the destination
CTA or its paired CTA. With multicast, each selected destination contributes a
completion to the parity selected by the barrier address. Two destinations in
one pair can therefore credit one barrier twice. `tma_cluster_targets.h` defines
this relationship once; both frontends and the existing asynchronous TMA backend
use explicit destination/completion masks. Shared offsets remain CTA-local.

Native CUDA 13.3 SM100/103 `UTMALDG.{1D..5D}` supports `.2CTA`, `.MULTICAST`, and
`.MULTICAST.2CTA`. Ordinary native cluster loads also resolve absolute rank bits
for the unmodified opcode: rank zero is a real destination, not an alias for the
issuing CTA. Non-cluster ordinary loads retain their previous path.

PTX also permits a peer completion barrier with a `shared::cta` destination and
`.cta_group::2`. Only the barrier resolver allows that peer; the destination
retains its CTA-local restriction.

Reference: [NVIDIA PTX tensor asynchronous copy semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor).

## Acceptance and boundaries

`tests/src/driver_tma/cluster_test.cu` runs twenty complete numerical launches
through each frontend: 1D–5D, paired and unpaired multicast, explicit remote
unicast, same/peer completion, two separate clusters and two barrier phases.
Every destination word and untouched peer word is checked. Higher-rank fixtures
use the retained Driver formats and 128B swizzle; they do not claim new support
for the compatibility host encoder's 3D/5D layout. The rank-1 fixture uses F32.
The existing twelve dense-map launches remain in the same CI gate.

The existing gather4 tests retain their separate acceptance. These are semantic
fixtures, not a calibration of multicast traffic or paired transfer latency.
Production timing presets are unchanged. General cluster-shape coverage,
additional host descriptor formats, packed ordinary TMA payloads and end-to-end
DeepInfer acceptance remain separate work.

Fresh retained `projection-index-q` now passes its ordinary paired TMA and reaches
an unextracted UTCCP descriptor form. It has not yet passed numerical acceptance
on this extracted stack. Older combined-private-stack passes remain historical.
