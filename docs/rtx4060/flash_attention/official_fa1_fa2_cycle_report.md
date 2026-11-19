# FA1/FA2 RTX 4060 Fullset Cycle Results

All entries compare NCU `sm__cycles_elapsed.avg` with simulator `gpu_tot_sim_cycle`.
MAPE is the mean of per-case absolute cycle percentage errors.

| Implementation | Cases | Cycle MAPE | Mean signed bias (sim/hw-1) | Min error | Max error |
|---|---:|---:|---:|---:|---:|
| FA1 | 8 | 22.904% | +22.904% | 15.825% | 29.969% |
| FA2 | 8 | 9.859% | +9.859% | 2.756% | 16.799% |
| FA1 + FA2 | 16 | 16.382% | +16.382% | 2.756% | 29.969% |

See [`official_fa1_fa2_cycle_results.csv`](official_fa1_fa2_cycle_results.csv) for per-case values and [`official_fa1_fa2_manifest.csv`](official_fa1_fa2_manifest.csv) for the fixed workload definitions. Detailed NCU reports, native outputs, and simulator logs remain in the local backup experiment tree.

## Provenance and scope

- FA1 used upstream FlashAttention v1.0.9, commit `6d48e14a6c2f551db96f0badc658a6279a929df3`, with CUTLASS commit `319a389f42b776fae5701afcb943fc03be5b5c25`.
- FA2 used the version pinned by the repository's existing [`tests/third_party/flash-attention/prepare.sh`](../../../tests/third_party/flash-attention/prepare.sh) workflow.
- Both used FP16 input/output, FP32 accumulation, no dropout, and the cases in the manifest. Batch was fixed at 8; sequence lengths were 128 and 256; `(H,D)` was `(32,64)` or `(16,128)`; both full and causal masks were included.
- The FA1 standalone harness and downloaded upstream source are preserved in the local full-evidence backup, but are not added as another vendored dependency or CI workflow in this PR. The repository-integrated FA2 path remains `tests/src/fa2/`.
- These results are an experimental calibration subset. The 16 rows passed output validation; they do not establish general FA1/FA2 support for every shape or a fully calibrated SM89 timing model.

## Initial analysis

- Every case has positive signed bias: the simulator overpredicts the NCU SM cycle metric across all 16 implementation/case rows.
- FA2 is closer than FA1 on this set: 9.859% versus 22.904% MAPE. Combined MAPE is 16.382%. The two implementations select different kernels and tiles, so the distinction is important when interpreting the aggregate.
- Mean absolute error is 17.593% at S=128 and 15.170% at S=256; 17.935% for D=64 and 14.828% for D=128; 15.877% full mask and 16.886% causal. These are descriptive patterns, not isolated causal effects because shape, mask, and kernel specialization vary together.
- Correctness replay passed, so the remaining discrepancy is in timing prediction rather than an observed output mismatch. The current data do not identify a single responsible hardware-model parameter; follow-up diagnosis should compare resource use and instruction/memory scheduling for the specific FA1/FA2 kernel variants.

The complete per-case NCU and simulator cycle values are in [`official_fa1_fa2_cycle_results.csv`](official_fa1_fa2_cycle_results.csv). Raw reports and logs are preserved in the local backup experiment tree.
