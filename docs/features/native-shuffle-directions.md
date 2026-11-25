# Native shuffle directions

Native `SHFL.UP` and `SHFL.DOWN` use the same lane-selection helper as PTX
UP/DOWN/BFLY/IDX. The helper decodes the five-bit offset and packed segment/clamp;
UP applies the lower bound and the other modes the upper bound. Native BFLY no
longer adds an extra lower-bound rejection absent from the PTX semantics.
Register access and participation stay in each frontend's adapter, while source
snapshots prevent aliased destinations from affecting subsequent lanes.

The prior independent PTX source-snapshot fix is a prerequisite. The direction
feature does not change timing constants, warp scheduling or PTX cycle counts.
Values sourced from nonparticipating lanes are undefined by the ISA; the
frontends' existing deterministic fallbacks are not hardware guarantees.

`sm100-warp-shuffle` runs three native component tests and five numerical
launches per frontend from one executable. Coverage includes all four modes,
sub-warp segments, offsets beyond their segment, result predicates, aliased
data/guard registers, and repeated predicated execution. Complete retained
indexer/MoE acceptance remains a separate gate.

Reference: [NVIDIA PTX ISA: shfl.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-shfl-sync).
