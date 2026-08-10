# Supported Test Utilities

This directory contains the maintained test-side command-line tools.

## Architecture manifest generation

`arch_manifests_to_make.py` validates architecture manifests and generates the
Make variables used by the test build. `tests/Makefile` invokes it
automatically when `tests/arch/*.toml` changes. Validate the manifests without
building tests with:

```bash
python3 tests/scripts/arch_manifests_to_make.py validate
```

## CI log validation

`check_cp_async_size.py` validates the per-case transaction and byte
counters emitted by the SM90 `CpAsyncSrcSizeTest` CI run:

```bash
python3 tests/scripts/check_cp_async_size.py path/to/integration.log
```

## Simulator utilities

### Extract statistics

`extract_sim_stats.py` reads one or more simulator logs and prints the last
statistics report from each log:

```bash
python3 tests/scripts/extract_sim_stats.py tests/run/example.log
python3 tests/scripts/extract_sim_stats.py --all tests/run/*.log
python3 tests/scripts/extract_sim_stats.py --csv stats.csv tests/run/*.log
```

### Run a simulator queue

`run_sim_queue.py` executes independent simulator GTest jobs with a bounded
worker pool. It requires a dedicated output directory and a tab-separated job
manifest:

```bash
python3 tests/scripts/run_sim_queue.py --print-example-jobs > jobs.tsv

CUDA_INSTALL_PATH=/usr/local/cuda-12.8 \
python3 tests/scripts/run_sim_queue.py \
  --jobs jobs.tsv \
  --run-root tests/run/my-queue \
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
