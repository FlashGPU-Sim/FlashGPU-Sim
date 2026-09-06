# SM90_H200_CLUSTER132

Product-scale **H200 NVL** packing for published cycle calibration.

**This is the default full-chip GPC map** for published H200 calibration.

## Packing (`inferred`, not a measured SMID layout)

| GPC | Enabled SMs | PG'd CPC slots (of 18) |
|-----|------------:|-----------------------:|
| 0–5 | 16 | 2 |
| 6–7 | 18 | 0 |

Total SMs = **132**. Knob: `-gpgpu_gpc_sms 16,16,16,16,16,16,18,18`.

TB-cluster `product(clusterDim) ≤ 16` (min enabled SMs in a GPC).

## Occupancy

CUDA CC 9.0: 2048 threads / 64 warps / 32 blocks per SM (`-gpgpu_shader_core_pipeline 2048:32`, `-gpgpu_shader_cta 32`).

## HBM and L2 geometry

`-gpgpu_n_mem 94` is derived from the
[NVIDIA H200 NVL product brief](https://dam-cdn.nvd.orangelogic.com/AssetLink/7n7vya4684sdccfyy6kv37ek5lw702h7.pdf)'s
6016-bit bus: 94 simulated channels × 64 bits. At the specified 3201 MHz
memory clock and data-command ratio 2, the modeled peak is 4.814 TB/s versus
the brief's 4.813 TB/s. Two subpartitions per channel produce 188 L2 slices;
the existing 320 KiB slice geometry therefore models 58.75 MiB total L2.

Selected latency and throughput knobs use validation-clean rows from H200 job
2119329. The job is only partially accepted because its 4096^3 GEMM case
failed. Suspicious fields not measured by that job use explicitly documented
H100 same-Hopper baselines; see `docs/cluster_noc/todos.md` under B6h.

## Relationship

| Config | Packing | Role |
|--------|---------|------|
| `SM90_H200` | 132 × 1 | Product clocks; **cannot** exercise DSM fabric |
| `SM90_H200_REDUCED_CLUSTER16x2` | 16 × 2 = 32 | Functional CI |
| **`SM90_H200_CLUSTER132`** | 6×16 + 2×18 | **Default published H200 calibration** |

## Usage

```bash
export OMP_NUM_THREADS=4
export FLASHGPU_ALLOW_CC_MISMATCH=1
# list-configs includes this directory name
```
