# SM120_RTX5090_CLUSTER2x85 Configuration

Full-scale GPU configuration with **TPC packing** (2 SMs per physical cluster)
for cluster-aware workloads. Exact product SM count (170).

For **GPC-aligned** packing (16 SMs per cluster, TB size up to 16), prefer
`SM120_RTX5090_CLUSTER16x11`. For reduced m>2 functional tests, prefer
`SM120_RTX5090_REDUCED_CLUSTER4x4`.

## Naming: `CLUSTERmxn`

| Token | Meaning | Knob |
|-------|---------|------|
| **m=2** | Physical cores (SMs) per `simt_core_cluster` | `-gpgpu_n_cores_per_cluster 2` |
| **n=85** | Number of physical clusters | `-gpgpu_n_clusters 85` |

Total SMs = 170 (matches base SM120_RTX5090 SM count; models RTX 5090 TPC
count of 85 × 2 SMs). **TB cluster size is not implied by the name** — it comes
from `cudaLaunchKernelEx` / `__cluster_dims__` / required func attributes.
Capacity rule: `ctas_per_cluster ≤ m` (so TB size ≤ 2).

## Key Parameters

- `-gpgpu_n_clusters 85`
- `-gpgpu_n_cores_per_cluster 2`
- Same TMA/mbarrier knobs as full SM120

## Usage

```bash
./test/run_tests.sh -c SM120_RTX5090_CLUSTER2x85 run test --target sm120 --group integration ClusterBasic
```

Prefer `SM120_RTX5090_REDUCED_CLUSTER2x1` / `4x4` for fast functional tests.
