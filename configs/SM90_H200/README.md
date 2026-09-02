# SM90_H200

Full **NVIDIA H200 SXM / NVL** configuration (Hopper GH100 compute + HBM3e memory).

## Hardware reference

| Item | H200 product | Notes |
|------|--------------|-------|
| Architecture | Hopper (GH100) | Same die as H100 SXM5 |
| SMs | **132** | 8 GPCs, 66 TPCs, 2 SMs/TPC |
| Full die | 144 SMs | 8 GPCs × 9 TPCs × 2 SMs; product floorswept |
| L2 | **50 MiB** model | Driver may report ~60 MiB on NVL |
| Memory | **~141–143 GiB HBM3e** | H100: 80 GB HBM3 |
| Mem BW | **~4.8 TB/s** class | Measured STREAM copy ~2.9 TB/s (suite) |
| Compute capability | **9.0** | sm_90 / sm_90a |

## Relationship to other configs

| Config | Role |
|--------|------|
| `SM90_H100` | Same compute silicon; older HBM3 / clock defaults |
| **`SM90_H200`** | Full 132-SM H200 latency profile |
| `SM90_H200_REDUCED_CLUSTER16x2` | 16×2 = 32 SMs (two GPCs × 16 enabled) for cluster/TMA/DSM |
| `SM120_RTX5090` | Blackwell consumer reference (different arch) |

Flat packing: `-gpgpu_n_clusters 132` / `-gpgpu_n_cores_per_cluster 1`.
Does **not** model GPC/TPC co-scheduling.

## Latency calibration (H200 NVL, job 2034797)

Primary source: `H200_profiling/output-2034797-H200Profiling.txt`  
(suite `all`, samples=48, complete CSV including `tma_mc` / MMA / WGMMA / DSM).

Prior calibration: job 2032656 (superseded by 2034797 for most knobs).

| Knob area | Job 2032656 | Job 2034797 (current) |
|-----------|-------------|------------------------|
| Core / DRAM clocks | 1785 / 3201 | **1785 / 3201** (unchanged) |
| Kernel launch | 8700 | **8270** |
| mbarrier arrive / trywait | 6 / 43 | **6 / 43** |
| INT DIV | 58 | **64** |
| FP DIV | 466 (bad) | **37** (pure `div.rn`) |
| DP DIV | 111 | **110** |
| SFU | 41 | **45** (rcp) |
| MMA tensor[0/1/2] | 28 / 32 / 19 | **29 / 29 / 29** |
| WGMMA SS issue / wait | 11 / 51 | **11 / 51** |
| WGMMA RS issue / wait | 19 / 41 | **19 / 41** |
| TMA issue | 68 | **68** |
| cp.async lat / init | 39 / 39 | **112 / 39** (pure gap / burst) |
| L1 / smem | 39 / 30 | **39 / 30** |
| L2 ROP | 280 | **286** (`multi_sm_l2_p50`) |
| far-L2 extra | 150 (H100 inherit) | **23** (measured spread) |
| DRAM residual | 360 | **360** (kept; cold RTT not isolated) |
| HBM STREAM copy | n/a | **~2909 GB/s** (validate peak) |
| DSM remote (e2e / fabric) | 193 / +156 | See job **2046238** below |
| TMA multicast | n/a | See job **2046238** below |

### DSM / NoC calibration (job **2046238**, supersedes 2034797 for NoC)

Source: `H200_profiling/output-2046238-H200Profiling.txt` (suite includes `dsm` + `tma_mc`).

| Metric | Value | Sim |
|--------|------:|-----|
| DSM local mean | ~37.05 cyc | `gpgpu_dsm_local_latency=37` |
| DSM remote e2e mean | ~193.41 cyc | load ≈ local + 2×hop |
| One-way hop | ~78 cyc | matrix / `gpgpu_dsm_remote_latency=78` |
| Stride ratio | **1.001** | **flat** all-to-all (not multi-hop tree) |
| TMA mcast−unicast e2e | ~**135** cyc | `gpgpu_tma_mcast_hop_latency=135` |
| Matrix file | 16×16 one-way | `dsm_latency_matrix_16.csv` |

See `docs/cluster_noc/knobs.md`, `docs/cluster_noc/todos.md`, and `docs/cluster_noc/calibration.md` for remaining BW/latency/GEMM calibration. This flat 132×1 packing **cannot** exercise the DSM fabric; cycle-accurate cluster numbers use `SM90_H200_CLUSTER132`.

### Still open

- HBM **bank** timings (CCD/RRD/…) — not invertible from BW alone  
- Cold DRAM RTT under exclusive GPU (this job had co-resident python)  
- int-WGMMA completion; RF-pressure byte/cycle budgets  
- Cluster mbarrier / TMA / DSM / GEMM cycle calibration (`docs/cluster_noc/calibration.md`). Do **not** fit `-gpgpu_dsm_bytes_per_cycle` as the fabric model.

## Usage

```bash
./test/run_tests.sh list-configs
./test/run_tests.sh -c SM90_H200 test "*WGMMA*"
./test/run_tests.sh -c SM90_H200 test "*TMA*"
```

Prefer `SM90_H200_REDUCED_CLUSTER16x2` for day-to-day multi-GPC functional tests.
