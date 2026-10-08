# FlashGPU-Sim GPU Features

FlashGPU-Sim extends execution-driven PTX simulation with data movement,
synchronization and matrix operations for the supplied Hopper and Blackwell
configurations. Support and timing limits differ by instruction family.

| Feature | Support and model documentation |
| --- | --- |
| TMA bulk/tensor transfers and tensor-map operations | [TMA](tma.md) |
| CTA memory barriers | [mbarrier](mbarrier.md) |
| Warp leader election | [elect.sync](elect.md) |
| Shared-memory matrix transfers | [ldmatrix/stmatrix](ld_st_matrix.md) |
| Warp-level `mma.sync` | [MMA](mma/README.md) |
| Blackwell TMEM and `tcgen05` | [TCGen05](tcgen05/README.md) |
| Hopper warp-group `wgmma` | [WGMMA validation](../../../tests/src/wgmma/README.md) and [SM90 configuration](../../../configs/README.md#matrix-execution) |

`wmma`, `mma.sync`, `wgmma` and `tcgen05` are distinct instruction families;
one family's supported data types and shapes do not establish support in
another. See each feature's validated variants and model limits when selecting
workloads.

Use the [configuration guide](../../../configs/README.md) for timing and
resource controls, the [test guide](../../../tests/README.md) for validation,
and [runtime tracing](../../../docs/tracing.md) to inspect simulated events.
