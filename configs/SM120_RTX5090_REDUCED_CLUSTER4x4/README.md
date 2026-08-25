# SM120_RTX5090_REDUCED_CLUSTER4x4

Cluster / DSM design: `docs/cluster_noc/README.md`. Checklist: `docs/cluster_noc/todos.md`.

Reduced multi-cluster config with **m>2** physical packing for development
and functional validation of Thread Block Clusters larger than 2.

## Naming: `CLUSTERmxn`

| Token | Meaning | Knob |
|-------|---------|------|
| **m=4** | Physical cores (SMs) per `simt_core_cluster` | `-gpgpu_n_cores_per_cluster 4` |
| **n=4** | Number of physical clusters | `-gpgpu_n_clusters 4` |

Total SMs = m×n = **16**. **TB cluster size is not “4×4”** — it comes from the
launch API. Capacity rule: `product(clusterDim) ≤ m` (so TB cluster size ≤ 4).

## Intended use

- Multi-cluster isolation tests (ordinary `<<<>>>`)
- Co-residency of TB-cluster launches with clusterDim up to 4
- Primary **functional** regression target for m>2 (prefer over full 16×11)

- `-network_mode 2` (local interconnect; BookSim icnt unused / not sized for 16 SMs)

## Usage

```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER4x4 test \
  "*Cluster*:*TMACluster*:*MultiCluster*"

./test/run_tests.sh -c SM120_RTX5090_REDUCED_CLUSTER4x4 test \
  "*ClusterLaunch*"
```

## Limitations

- Ordinary grid launches: no co-residency guarantee (validates isolation).
- Cluster launches (`cudaLaunchKernelEx` / required cluster dims): CTAs of one
  TB cluster are co-scheduled on one physical cluster.
- Functional correctness focus; not a cycle-accurate GPC model.
- Cluster TMA timing idealized (free multicast after single L2/TMA path).
