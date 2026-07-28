# SM120_RTX5090_REDUCED_CLUSTER2 Configuration

Lightweight GPU configuration for fast testing of 2-SM-per-cluster scheduling
and TMA `.shared::cluster` multicast.

This reduced configuration provides a minimal GPGPU-Sim setup with one cluster
containing two SMs. It maintains fast simulation time while exercising the
cluster round-robin CTA issuance path and issue-order `cluster_group` peer
matching.

## Key Parameters

- `-gpgpu_n_clusters 1`
- `-gpgpu_n_cores_per_cluster 2`
- `-gpgpu_n_mem 16`
- `-network_mode 2` (local interconnect; BookSim `config_ampere_islip.icnt` unused)

## Usage

```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2 test "ClusterBasic"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2 test "TMACluster*"
```

Preferred config for validating cluster/CTA=2 topology and functional TMA
multicast (including one-producer peer mbarrier complete).

## Limitations (read before interpreting cycles)

- **Not CUDA Thread Block Clusters.** Tests use plain `<<<N, threads>>>` launches.
  The simulator co-schedules CTAs onto multi-SM clusters via round-robin issue
  and issue-order `cluster_group`; this is **not** `cudaLaunchKernelEx` /
  `__cluster_dims__` cooperative cluster launch.
- **Cluster TMA timing is idealized.** Functional multicast is free after a
  single L2/TMA stream (no DSM hop cost / no peer mem_fetches).
- **mbarrier knobs (200/120)** are end-to-end TMA+mbarrier calibration, not pure
  hardware barrier cost (same values as full SM120).
