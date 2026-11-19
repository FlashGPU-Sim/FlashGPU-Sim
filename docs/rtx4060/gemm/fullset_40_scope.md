# RTX 4060 GEMM FP16 full-set scope

This selected run contains 40 experiments: the 40 paper GEMM shapes, each run once with FP16. All use `BLOCK_M=64`, `BLOCK_N=64`, `BLOCK_K=32`, `GROUP_SIZE_M=8`, `num_warps=4`, and `num_stages=3`.

The runnable case list is [`paper_gemm_cases_4060_fullset.csv`](../../../tests/triton_trace/calibration/paper_gemm_cases_4060_fullset.csv). No BF16 GEMM support or accuracy claim is included in this change.

Results are retained in `baseline_40case_fp16_results.csv` and `e118_reserved_40case_fp16_results.csv`; they are filtered from the previously completed run, not new measurements.
