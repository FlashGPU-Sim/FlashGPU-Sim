# SM120_RTX5090_REDUCED_CLUSTER2x2

Multi-cluster reduced config for development iteration with cluster-aware features.

## Naming: `CLUSTERmxn`

| Token | Meaning | Knob |
|-------|---------|------|
| **m=2** | Physical cores (SMs) per `simt_core_cluster` | `-gpgpu_n_cores_per_cluster 2` |
| **n=2** | Number of physical clusters | `-gpgpu_n_clusters 2` |

Total SMs = m×n = 4. **TB cluster size is not “2×2”** — it comes from the
launch API. This topology is for multi-cluster isolation tests (ordinary
`<<<>>>`) and for co-residency checks of real TB-cluster launches on a
multi-cluster GPU.

- Identical packing **m** to `SM120_RTX5090_REDUCED_CLUSTER2x1`, but **n=2**
- `-network_mode 2` (local interconnect; BookSim icnt unused / not sized for 4 SMs)

## Usage

```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x2 test "*MultiCluster*"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x2 test "TMAClusterOneProducer*"
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER2x2 test "*ClusterLaunch*"
```

## Limitations

- Ordinary grid launches: no co-residency guarantee (validates isolation).
- Cluster launches (`cudaLaunchKernelEx` / required cluster dims): CTAs of one
  TB cluster are co-scheduled on one physical cluster.
- Cluster TMA timing idealized (free multicast after single L2/TMA path).
