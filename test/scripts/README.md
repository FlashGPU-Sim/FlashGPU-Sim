# Supported Test Utilities

This directory contains the maintained test-side command-line tools.
Architecture and source membership live in `test/arch/*.toml`; build helpers
consume the same manifests instead of carrying a second compatibility table.

## Simulator utilities

### Extract statistics

`extract_sim_stats.py` reads one or more simulator logs and prints the last
statistics report from each log:

```bash
python3 test/scripts/extract_sim_stats.py test/run/example.log
python3 test/scripts/extract_sim_stats.py --all test/run/*.log
python3 test/scripts/extract_sim_stats.py --csv stats.csv test/run/*.log
```

### Run a simulator queue

`run_sim_queue.py` executes independent simulator GTest jobs with a bounded
worker pool. It requires a dedicated output directory and a tab-separated job
manifest:

```bash
python3 test/scripts/run_sim_queue.py --print-example-jobs > jobs.tsv

CUDA_INSTALL_PATH=/usr/local/cuda-12.8 \
python3 test/scripts/run_sim_queue.py \
  --jobs jobs.tsv \
  --run-root test/run/my-queue \
  --max-parallel 4
```

The required TSV columns are:

| Column | Meaning |
| --- | --- |
| `job_id` | Unique stable identifier used for status and resume files. |
| `stage` | User-defined phase label. |
| `case` | User-defined case label. |
| `binary` | Absolute path or path relative to the repository root. |
| `gtest_filter` | Exact GTest filter passed to the binary. |

Optional columns are `args` (shell-split arguments), `config` (a directory
under `configs/`, default `SM90_H100`), and `skip` (`1`, `true`, or `yes`).
Use `--config` for the queue-wide simulator config and `--env KEY=VALUE` for
extra environment variables. `--cuda-path` defaults to
`CUDA_INSTALL_PATH`. `--cpu-sets` is optional; without it, jobs are not pinned
with `taskset`.

The output contains per-job logs and work directories plus
`status/summary.tsv`. `--resume` skips jobs whose existing status is `done`.
Use a run directory dedicated to this command: a rerun recreates the work
directory of every job that it executes.

## FA2 hardware profiling

The FA2 flow has two entry points:

1. `prepare_fa2_prebuilt.sh` builds a device-specific package.
2. `run_fa2_ncu.sh` runs selected packaged cases and collects native logs and
   Nsight Compute reports.

Build a package for an architecture whose manifest includes the FA2 test
group:

```bash
./test/scripts/prepare_fa2_prebuilt.sh \
  --device h100 \
  --cuda-root /usr/local/cuda-12.8 \
  --out-dir test/run/fa2-h100 \
  --group full \
  --group breakdown \
  --group scaling

./test/scripts/prepare_fa2_prebuilt.sh \
  --device rtx5090 \
  --cuda-root /usr/local/cuda-12.8 \
  --out-dir test/run/fa2-rtx5090 \
  --group full
```

`h100` resolves `sm90.toml` and its `sm_90a` target; `rtx5090` resolves
`sm120.toml` and its `sm_120a` target. Both manifests include FA2, so neither
path needs a hidden compiler-target override. The runner rejects a package
created for another device.

Selectors are repeatable. `full` expands to `smoke`, `small`, `medium`, and
`large`; `all` additionally includes every breakdown, scaling, and
concurrency mode. A mode-less `breakdown`, `scaling`, or `concurrency`
selector includes all modes. Use `GROUP:MODE`, for example
`breakdown:only_mma`, to package or run one mode.

On the destination machine:

```bash
test/run/fa2-h100/run_fa2_ncu.sh \
  --device h100 \
  --group full \
  --group breakdown:only_mma

test/run/fa2-h100/run_fa2_ncu.sh \
  --device h100 \
  --group scaling \
  --case H1D128
```

`--case` is a substring filter and may be repeated. `--print-cases` and
`--dry-run` need no GPU. `--no-native` and `--no-ncu` select collection
phases. Both hardware runners enable `--resume` by default: reuse the same
`--out-dir` to skip cases with successful `.done` markers. Use `--force` to
rerun them, or `--no-resume` to ignore the markers. The package contains its
binary/case manifest, `libcudart`, hashes, build logs, and provenance. The
packager also writes a sibling `.tar.gz` archive for transfer.

The build host needs Bash 4+, a CUDA Toolkit with the requested architecture,
and the FA2/FA3 source preparation dependencies. The run host needs a matching
NVIDIA driver and GPU; NCU collection additionally needs Nsight Compute.

## FA3 hardware profiling

`run_fa3_ncu.sh` builds and runs manifest-owned FA3 profile modes on H100:

```bash
./test/scripts/run_fa3_ncu.sh \
  --cuda-root /usr/local/cuda-12.8 \
  --group breakdown:qk_pv_only_no_tma \
  --group breakdown:qk_pv_only_no_tma_reg_timeline

./test/scripts/run_fa3_ncu.sh \
  --group scaling:baseline \
  --group concurrency:sync_only_no_tma
```

Each selector must name `breakdown`, `scaling`, or `concurrency` and an
explicit manifest mode. Repeating `--group` supports comparisons across
experiment groups or modes. The manifest supplies the binary, GTest filter,
and default case list. Repeated `--case` arguments replace those case lists.
Use `--print-cases` to inspect the resolved plan, `--dry-run` to inspect
commands, and `--dump-sass` when SASS is part of the experiment. Resume and
force behavior is identical to `run_fa2_ncu.sh`.

This flow requires H100-compatible hardware, an `sm_90a`-capable CUDA Toolkit,
and Nsight Compute unless `--no-ncu` is used. It intentionally does not
support the former ad-hoc FA3 prebuilt layout or the mixed FA3/WGMMA
experiment.

## Migration from removed scripts

| Removed or moved entry | Replacement or status |
| --- | --- |
| `test_gtest_discovery_output.py` | Moved to `test/ci/test_gtest_discovery_output.py`; CI runs it once in the SM120/core job. |
| `test_ptx_scheduler_probe_operands.py` | Removed with the Python-only PTX/SASS probes; no production scheduler test was lost. |
| `ptx_sass_guided_scheduler_probe.py` | Removed; no supported replacement. |
| `ptx_window_scheduler_probe.py` | Removed; no supported replacement. |
| `manual_sass_replay_ptx.py` | Removed; no supported replacement. |
| `prepare_fa2_breakdown_prebuilt.sh` | `prepare_fa2_prebuilt.sh --group breakdown` |
| `prepare_fa2_scaling_prebuilt.sh` | `prepare_fa2_prebuilt.sh --group scaling` |
| `prepare_fa2_full_ncu_prebuilt.sh` | `prepare_fa2_prebuilt.sh --group full` |
| `run_fa2_breakdown_h100.sh` | `run_fa2_ncu.sh --device h100 --group breakdown` |
| `run_fa2_scaling_h100.sh` | `run_fa2_ncu.sh --device h100 --group scaling` |
| `run_fa2_full_ncu_h100.sh` | `run_fa2_ncu.sh --device h100 --group full` |
| `run_fa2_full_ncu_5090.sh` | Build an `rtx5090` package, then use `run_fa2_ncu.sh --device rtx5090 --group full`. GPU clock locking is no longer managed by the test script. |
| `run_fa3_breakdown_h100.sh` | Repeat the old modes with `run_fa3_ncu.sh --group breakdown:MODE`. |
| `run_fa3_sync_noprofile_regtimeline_h100.sh` | Repeat `--group breakdown:qk_pv_only_no_tma_noprofile`, `--group breakdown:sync_only_no_tma_noprofile`, and `--group breakdown:qk_pv_only_no_tma_reg_timeline`. |
| `run_fa3_extended_n176_h100.sh` | Use the corresponding `qk_pv_only_no_tma_extended` modes and `--dump-sass`. |
| `run_fa3_prebuilt_noprofile_regtimeline_h100.sh` | Removed; its unowned prebuilt package format is unsupported. |
| `run_fa3_extended_wgmma_h100.sh` | Removed; the mixed FA3/WGMMA experiment is unsupported. |
| `plot_fa3_timeline_compare.py` | Removed; no supported replacement. |
| `generate_l2_bucket_pair_plan.py` | Removed with the narrow L2 calibration flow. |
| `run_l2_bucket_oracle_5090.sh` | Removed with the narrow L2 calibration flow. |
| `analyze_l2_bucket_pair_ncu.py` | Removed with the narrow L2 calibration flow. |
| `run_l2_hbm_interleave_h100.sh` | Removed with the narrow L2 calibration flow. |
| `summarize_l2_hbm_interleave_ncu.py` | Removed with the narrow L2 calibration flow. |
| `analyze_l2_partition_latency.py` | Removed; no supported replacement. |
| `run_mma_saturation_h100.sh` | Removed; maintained MMA benchmarks remain available through the test manifest. |
