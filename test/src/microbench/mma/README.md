# MMA Microbenchmark Suite

This directory contains MMA-focused microbenchmarks for Tensor Core issue gap,
peak throughput, and instruction-latency calibration.

## Supported Tests

Run these tests from `test/` with `./run_tests.sh bench "<pattern>"`.

| Test | Source | Description | Output |
|------|--------|-------------|--------|
| `MMAIssueTest.ILPMinimal` | `mma_issue_bench.cc` | Typed test that sweeps ILP `{1,2,4,8}` for every supported MMA variant. | `MMAIssueTest.ILPMinimal.<variant>.txt` |
| `MMAIssueTest.MultiWarpMinimal` | `mma_issue_bench.cc` | Typed test that sweeps warp count `{1,2,4,8,16,32}` for every supported MMA variant. | `MMAIssueTest.MultiWarpMinimal.<variant>.txt` |
| `MMAIssueSummary.AllVariants` | `mma_issue_bench.cc` | Cross-variant summary of cycles per MMA from the issue-gap microbench. | `MMAIssueTest.Summary.txt` |
| `MMAPeak.AllVariants` | `mma_issue_bench.cc` | Full-device peak throughput summary across all MMA variants. | `MMAPeak.Summary.txt` |
| `InstLatencyTest.FullCalibrationSuite` | `inst_latency_bench.cc` | Broad scalar/SFU/MMA instruction-latency calibration sweep used for simulator tuning. | Timestamped `test/run/logs/inst_latency_*.log` |

## Variant Coverage

`mma_issue_bench.cc` uses `TYPED_TEST_SUITE`, so the issue-gap benchmarks run
for every supported MMA op in one build. There is no longer a manual
`CurrentMmaOp` switch in the source.

- `MmaOp_F16_M16N8K16`
- `MmaOp_F16_M16N8K8`
- `MmaOp_BF16_M16N8K8`
- `MmaOp_TF32_M16N8K8`
- `MmaOp_TF32_M16N8K4`
- `MmaOp_S8_M16N8K32`
- `MmaOp_S8_M16N8K16`

## Example Commands

```bash
./run_tests.sh bench "MMAIssueTest.ILPMinimal"
./run_tests.sh bench "MMAIssueSummary.AllVariants"
./run_tests.sh bench "MMAPeak.AllVariants"
./run_tests.sh bench "InstLatencyTest.FullCalibrationSuite"
```

For `MMAPeak.AllVariants`, `BestILP` and `Blocks/SM` are chosen by sweeping
candidate launch settings and keeping the highest measured throughput point.
`Meas GHz`, `Theo Now`, and `% Peak` are all derived from the measured runtime
clock instead of a fixed spec clock.

## Running Calibration

To compare native hardware against GPGPU-Sim, use the provided Python script:

```bash
# From repository root
python3 test/src/microbench/mma/run_calibration.py

# Or from test/ directory
python3 src/microbench/mma/run_calibration.py
```

This script:
1. Runs `MMAIssueTest.ILPMinimal` and `MMAIssueTest.MultiWarpMinimal` on native hardware.
2. Sources `setup.sh` and `setup_environment`, then reruns the same cases on GPGPU-Sim.
3. Captures the per-variant outputs and generates comparison plots/CSVs in `test/calibration_results/`.

**Output:**
- `calibration_results/ilp_issue_gap_calibration.png`
- `calibration_results/ilp_issue_gap_calibration.csv`
- `calibration_results/multiwarp_throughput_calibration.png`
- `calibration_results/multiwarp_throughput_calibration.csv`
