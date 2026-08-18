# SM90_H200_REDUCED_CLUSTER4x4

Reduced multi-cluster **H200 (Hopper + HBM3e)** config with **m>2** physical
packing for development and functional validation of Thread Block Clusters
larger than 2.

## Naming: `CLUSTERmxn`

| Token | Meaning | Knob |
|-------|---------|------|
| **m=4** | Physical cores (SMs) per `simt_core_cluster` | `-gpgpu_n_cores_per_cluster 4` |
| **n=4** | Number of physical clusters | `-gpgpu_n_clusters 4` |

Total SMs = m×n = **16**. **TB cluster size is not “4×4”** — it comes from the
launch API. Capacity rule: `product(clusterDim) ≤ m` (so TB cluster size ≤ 4).

## Hardware baseline

Same GH100 compute as full `SM90_H200` (cc 9.0, Hopper SM, TMA, WGMMA, 50 MiB
L2 model, 80 HBM channels). Clocks and latency knobs match `SM90_H200`
(H200 NVL job **2034797**). Only SM packing is reduced.

## Intended use

- Multi-cluster isolation tests (ordinary `<<<>>>`)
- Co-residency of TB-cluster launches with clusterDim up to 4
- Primary **functional** regression target for m>2 on H200 (prefer over full 132 SM)
- Fast Hopper TMA / mbarrier / cluster iteration

- `-network_mode 2` (local interconnect; BookSim icnt unused / not sized for 16 SMs)
- `-gpgpu_tma_idealized_memory 1` (functional TMA path)
- `-gpgpu_dsm_store_immediate 0` (peer DSM `st` visible only after NoC deliver)

## Usage

```bash
./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER4x4 test \
  "*Cluster*:*TMACluster*:*MultiCluster*"

./test/run_tests.sh -c SM90_H200_REDUCED_CLUSTER4x4 test \
  "*ClusterLaunch*"
```

## Latency

Synced with `SM90_H200` / job 2034797. Absolute performance still differs from
full 132-SM packing; use full `SM90_H200` for scale studies.

## Limitations

- Ordinary grid launches: no co-residency guarantee (validates isolation).
- Cluster launches (`cudaLaunchKernelEx` / required cluster dims): CTAs of one
  TB cluster are co-scheduled on one physical cluster.
- Functional correctness focus; not a cycle-accurate GPC model.
- Cluster TMA timing idealized (free multicast after single L2/TMA path).
- Not a substitute for full `SM90_H200` when measuring absolute performance.
