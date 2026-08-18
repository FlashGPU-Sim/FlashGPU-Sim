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

`run_sim_queue.py` executes independent commands with a bounded worker pool.
It reads CSV or TSV from a file (format is auto-detected) or from stdin with
`--jobs -`:

```bash
python3 tests/scripts/run_sim_queue.py --print-example-jobs > jobs.csv

python3 tests/scripts/run_sim_queue.py \
  --jobs jobs.csv \
  --run-root tests/run/my-queue \
  --max-parallel 4
```

The minimal manifest fields are:

| Column | Meaning |
| --- | --- |
| `job_id` | Unique stable identifier used for status and resume files. |
| `executable` | Absolute path or path relative to `--root`. |

Generic optional fields are `args` (shell-split), `cwd`, `env` (JSON object or
shell-split `KEY=VALUE` entries), `timeout`, and `enabled`. Every other column
is opaque metadata copied into per-job status and summaries; it never changes
execution. The old `binary`/`gtest_filter`/`config`/`skip` simulator manifest
continues to work as a compatibility format.

By default, the queue selects four idle physical cores per worker and runs at
most four workers. Change these independently with `--cpus-per-job` and
`--max-parallel`; `--threads-per-job` controls OMP/BLAS thread limits. Use
`--cpu-sets 0-3 4-7` for manual worker affinity, or `--no-pin` to let every
worker use all CPUs allowed to the queue process. `--cpu-sets` and `--no-pin`
are mutually exclusive. GPGPU-Sim setup is opt-in for new manifests through
`--config`; generic jobs do not receive an implicit simulator config.

`--run-root` owns `queue.json`, per-job logs/work/status, and both CSV and TSV
summaries. An existing queue root requires `--resume` (skip completed jobs) or
`--overwrite` (replace queue-managed output). `Ctrl+C` stops dispatch,
terminates active job process groups, records unfinished jobs as interrupted,
and returns 130; a second `Ctrl+C` escalates to `SIGKILL`.
