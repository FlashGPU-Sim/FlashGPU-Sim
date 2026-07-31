# SM120_RTX5090_CLUSTER2x85 Configuration

Full-scale GPU configuration with multi-SM physical packing for cluster-aware
workloads (TMA multicast, multi-SM scheduling).

## Naming: `CLUSTERmxn`

| Token | Meaning | Knob |
|-------|---------|------|
| **m=2** | Physical cores (SMs) per `simt_core_cluster` | `-gpgpu_n_cores_per_cluster 2` |
| **n=85** | Number of physical clusters | `-gpgpu_n_clusters 85` |

Total SMs = 170 (matches base SM120_RTX5090 SM count). **TB cluster size is
not implied by the name** — it comes from `cudaLaunchKernelEx` /
`__cluster_dims__` / required func attributes. Capacity rule:
`ctas_per_cluster ≤ m`.

## Key Parameters

- `-gpgpu_n_clusters 85`
- `-gpgpu_n_cores_per_cluster 2`
- Same TMA/mbarrier knobs as full SM120

## Usage

```bash
./test/run_tests.sh -c SM120_RTX5090_CLUSTER2x85 test ClusterBasic
```

Prefer `SM120_RTX5090_REDUCED_CLUSTER2x1` for fast functional tests.
