# SM120_RTX5090_REDUCED_CLUSTER2x1 Configuration

Lightweight GPU configuration for fast testing of multi-SM physical packing
and TMA `.shared::cluster` multicast.

## Naming: `CLUSTERmxn`

| Token | Meaning | Knob |
|-------|---------|------|
| **m=2** | Physical cores (SMs) per `simt_core_cluster` | `-gpgpu_n_cores_per_cluster 2` |
| **n=1** | Number of physical clusters | `-gpgpu_n_clusters 1` |

**This is not the CUDA Thread Block Cluster size.** TB cluster size comes from
the launch (`cudaLaunchKernelEx` / `__cluster_dims__` / required func attrs).
Physical packing must satisfy `ctas_per_cluster ≤ m`.

## Key Parameters

- `-gpgpu_n_clusters 1`
- `-gpgpu_n_cores_per_cluster 2`
- `-gpgpu_n_mem 16`
- `-network_mode 2` (local interconnect; BookSim `config_ampere_islip.icnt` unused)

## Usage

```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 test "ClusterBasic"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 test "TMACluster*"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x1 test "*ClusterLaunch*"
```

Preferred config for validating physical packing m=2 and functional TMA
multicast (including one-producer peer mbarrier complete).

## Limitations

- Cluster TMA timing is idealized (free multicast after a single L2/TMA stream).
- mbarrier knobs (200/120) are end-to-end TMA+mbarrier calibration, not pure
  hardware barrier cost (same values as full SM120).
