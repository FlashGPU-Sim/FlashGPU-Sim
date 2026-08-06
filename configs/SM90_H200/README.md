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
| `SM90_H200_REDUCED_CLUSTER4x4` | 4×4 = 16 SMs for fast functional cluster/TMA work |
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
| DSM remote (future) | 193 / +156 | **196 / +159** |
| TMA multicast (future) | n/a | mask extras **≈ 0** (free in sim) |

### Still open

- HBM **bank** timings (CCD/RRD/…) — not invertible from BW alone  
- Cold DRAM RTT under exclusive GPU (this job had co-resident python)  
- int-WGMMA completion; RF-pressure byte/cycle budgets  
- Cycle-accurate TMA multicast / DSM NoC knobs (documented only)

## Usage

```bash
./test/run_tests.sh list-configs
./test/run_tests.sh -c SM90_H200 test "*WGMMA*"
./test/run_tests.sh -c SM90_H200 test "*TMA*"
```

Prefer `SM90_H200_REDUCED_CLUSTER4x4` for day-to-day multi-cluster functional tests.
