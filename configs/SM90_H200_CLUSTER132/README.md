# SM90_H200_CLUSTER132

Product-scale **H200 NVL** packing for published cycle calibration.

**This is the default full-chip GPC map** for published H200 calibration.

## Packing (`inferred`, not a measured SMID layout)

| GPC | Enabled SMs | PG'd CPC slots (of 18) |
|-----|------------:|-----------------------:|
| 0–3 | 17 | 1 |
| 4–7 | 16 | 2 |

Total SMs = **132**. Knob: `-gpgpu_gpc_sms 17,17,17,17,16,16,16,16`.

TB-cluster `product(clusterDim) ≤ 16` (min enabled SMs in a GPC).

## Occupancy

CUDA CC 9.0: 2048 threads / 64 warps / 32 blocks per SM (`-gpgpu_shader_core_pipeline 2048:32`, `-gpgpu_shader_cta 32`).

## Relationship

| Config | Packing | Role |
|--------|---------|------|
| `SM90_H200` | 132 × 1 | Product clocks; **cannot** exercise DSM fabric |
| `SM90_H200_REDUCED_CLUSTER16x2` | 16 × 2 = 32 | Functional CI |
| **`SM90_H200_CLUSTER132`** | 4×17 + 4×16 | **Default published H200 calibration** |

## Usage

```bash
export OMP_NUM_THREADS=4
export FLASHGPU_ALLOW_CC_MISMATCH=1
# list-configs includes this directory name
```
