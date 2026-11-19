# RTX 4060 FP16 GEMM full-set cycle comparison

## Scope

The retained set contains 40 paper GEMM shapes, all run as FP16 with the fixed `64×64×32` tile, `GROUP_SIZE_M=8`, `num_warps=4`, and `num_stages=3`. Each case compares the median of five NCU `sm__cycles_elapsed.avg` samples against simulator `gpu_tot_sim_cycle`. Numerical output validation passed for all retained rows.

## Results

| Simulator configuration | Cases | FP16 full-set cycle MAPE |
|---|---:|---:|
| Baseline | 40/40 | 14.817% |
| E118 with 1 KiB/CTA shared-memory reserve | 40/40 | 14.676% |

These rows are filtered from the completed combined experiment; this commit does not claim new profiling or simulation runs. Per-case retained values are in `baseline_40case_fp16_results.csv` and `e118_reserved_40case_fp16_results.csv`.

## Interpretation

The E118 shared-memory reserve improves this full-set MAPE by about 0.141 percentage points. The remaining error is shape-dependent, so the result does not establish general GEMM accuracy or accuracy for other tiles, dtypes, or kernels. Further calibration should use the fixed FP16 case manifest and preserve the current result files as the prior measurement record.
