# SM90_H200_CLUSTER16x8

Full-chip GPC-packed **H200** config for cycle-accurate cluster calibration.
Design: `docs/cluster_noc/README.md`. Report:
`docs/cluster_noc/calibration.md`.

## Naming: `CLUSTERmxn`

| Token | Meaning | Knob |
|-------|---------|------|
| **m=16** | Enabled SMs per GPC | `-gpgpu_n_cores_per_cluster` / `-gpgpu_num_sms_per_gpc` |
| **n=8** | Number of GPCs | `-gpgpu_n_clusters` / `-gpgpu_num_gpcs` |

Total SMs = 16 × 8 = **128**. Product H200 is 132 SMs (~16.5 per GPC). This
preset uses the same uniform 16-SM GPC as `SM90_H200_REDUCED_CLUSTER16x2`.
TB-cluster size is a launch attribute: `product(clusterDim) ≤ 16`.

## Intended use

- Published mbarrier / TMA / DSM / GEMM **cycle** comparisons vs H200
- Intra-GPC fabric on (`-gpgpu_dsm_enable 1`)
- `OMP_NUM_THREADS=4` when running (shared machine)

**Do not** use this config for day-to-day functional filters. Prefer
`SM90_H200_REDUCED_CLUSTER16x2` (32 SMs) for `DsmTest.*` / `MbarrierClusterTest.*`.

**Do not** use shipped `SM90_H200` (132 × 1 SM/GPC) for DSM fabric numbers.

## Relationship to other configs

| Config | Packing | Role |
|--------|---------|------|
| `SM90_H200` | 132 × 1 | Product clocks / L2 / HBM; NoC idle |
| `SM90_H200_REDUCED_CLUSTER16x2` | 16 × 2 = 32 | Functional cluster tests |
| **`SM90_H200_CLUSTER16x8`** | 16 × 8 = 128 | Cycle calibration |

Clocks, caches, HBM, and SM pipeline knobs are copied from `SM90_H200`
(job 2034797). Fabric knobs: `docs/cluster_noc/knobs.md` §3.

## Usage

```bash
export OMP_NUM_THREADS=4
export FLASHGPU_ALLOW_CC_MISMATCH=1
bash scripts/run_calibration_sim.sh -- --suite smoke
# or any binary that reads gpgpusim.config from cwd
```
