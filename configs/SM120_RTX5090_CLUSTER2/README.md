# SM120_RTX5090_CLUSTER2 Configuration

Full-scale RTX 5090 configuration with 2 SMs per cluster.

This variant matches the total SM count of the standard RTX 5090 config
(170 SMs) but groups them into 85 clusters with 2 SMs each, enabling
cluster-level features such as distributed shared memory style TMA multicast
under GPGPU-Sim’s multi-core-per-cluster topology.

## Key Parameters

- `-gpgpu_n_clusters 85`
- `-gpgpu_n_cores_per_cluster 2`
- `-network_mode 2` (local interconnect; BookSim `config_ampere_islip.icnt` unused)

## Usage

```bash
./test/run_tests.sh -c SM120_RTX5090_CLUSTER2 test ClusterBasic
```

Use this configuration for full-scale cluster/CTA=2 scheduling and functional
cluster TMA. Prefer `SM120_RTX5090_REDUCED_CLUSTER2` for fast functional tests.

## Limitations

- Topology / issue-order `cluster_group` model — not full CUDA cluster launch APIs.
- Cluster TMA multicast hop is free in the timing model (one issuer memory path).
- BookSim icnt `k` may not match cluster=2 node counts; regenerate before
  switching off `network_mode 2`.
