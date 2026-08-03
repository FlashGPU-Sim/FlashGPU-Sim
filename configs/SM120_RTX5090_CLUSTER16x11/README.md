# SM120_RTX5090_CLUSTER16x11

Full-scale **GPC-aligned** GPU configuration for cluster-aware workloads
(TMA multicast, multi-SM TB-cluster co-scheduling).

## Naming: `CLUSTERmxn`

| Token | Meaning | Knob |
|-------|---------|------|
| **m=16** | Physical cores (SMs) per `simt_core_cluster` | `-gpgpu_n_cores_per_cluster 16` |
| **n=11** | Number of physical clusters | `-gpgpu_n_clusters 11` |

Total SMs = **176**.

**TB cluster size is not “16×11”** — it comes from `cudaLaunchKernelEx` /
`__cluster_dims__` / required func attributes. Capacity rule:
`product(clusterDim) ≤ m`.

## Hardware rationale (RTX 5090 / GB202)

From the NVIDIA RTX Blackwell architecture whitepaper:

| Level | Full GB202 | GeForce RTX 5090 |
|-------|------------|------------------|
| GPCs | 12 | 11 |
| TPCs | 96 | 85 |
| SMs | 192 | 170 |
| SMs per full GPC | 16 (8 TPCs × 2 SMs) | ~15.45 avg (floorsweep) |

CUDA Thread Block Clusters co-schedule within a **GPC**. This config models
**ideal GPC packing** (16 SMs × 11 GPCs = 176 SMs). Product RTX 5090 is
floorswept to **170 SMs**; we intentionally keep uniform m=16 rather than
uneven per-GPC counts.

Consumer Blackwell (SM 12.0) max **portable** TB cluster size is typically **8**;
this packing allows TB clusters up to **16** (capacity = m).

Compare with `SM120_RTX5090_CLUSTER2x85` (TPC packing, m=2, exact 170 SMs) —
that profile only supports TB cluster size ≤ 2.

## Key Parameters

- `-gpgpu_n_clusters 11`
- `-gpgpu_n_cores_per_cluster 16`
- Same TMA/mbarrier knobs as other SM120 cluster configs

## Usage

```bash
# Functional smoke (full 176-SM suite is heavy)
./test/run_tests.sh -c SM120_RTX5090_CLUSTER16x11 test \
  "ClusterLaunchApiTest.*:TMAClusterOneProducer*"
```

Prefer `SM120_RTX5090_REDUCED_CLUSTER4x4` for fast multi-cluster + m>2
functional iteration.

## Limitations

- Ideal GPC grid (176 SMs), not product floorsweep (170).
- Functional cluster model (co-residency + peer TMA); cycle-accurate GPC
  interconnect timing is not the goal of this profile.
- Cluster TMA timing idealized (free multicast after single L2/TMA path).
