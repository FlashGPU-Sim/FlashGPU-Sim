# Test utilities

Existing build helpers and the general queue remain unchanged. SASS adds only
the FA-3 medium/large forward regression workflow below; hardware calibration
sweeps and instruction-stream audits are not part of the public source tree.

## FA-3 forward regression

Use CUDA 13.3. Build a fresh, isolated package (no GPU required):

```sh
tests/scripts/package_fa3_regression_h100.sh /tmp/fa3-regression
```

Transfer the package's `bin/`, `lib/`, `collect.sh`,
`summarize_fa3_standard_ncu.py`, `checksums.sha256`, `source-commit.txt`,
`source-changes.patch` and `compiler.txt` to H100; build/dependency directories
need not be transferred. Collect native execution and full NCU at 1500 MHz:

```sh
./collect.sh --profile medium --profile large --out-dir /tmp/fa3-native
```

Collection needs non-interactive sudo for clock locking and resets that lock
on exit (it does not restore an earlier user lock). Use a native CUDA
environment, not the simulator's LD_LIBRARY_PATH. Keep the exact package,
NCU reports, and summary together; rebuilding can change SASS and timing.

Back on the simulator host, source `setup_environment release`, then:

```sh
python3 tests/scripts/run_fa3_large_forward_queue.py \
  --binary /tmp/fa3-regression/bin/standard_tests \
  --hardware /tmp/fa3-native/case_cycles_summary.csv \
  --profile large --config configs/SM90_H100_SASS_FRONTEND \
  --out-dir /tmp/fa3-large-sim
```

Use `--profile medium` and a different output directory for its four cases.
Large runs twenty cases in ascending S order: S=512/1024/2048/4096/8192,
B=64/32/16/8/4 respectively, H32/D64 or H16/D128, full/causal.
Medium uses B16/S512 with the same H/D/full/causal combinations.

Four workers each use four distinct physical cores (no SMT overlap), taking
the next task as soon as one finishes. Library, config and decoder are frozen;
resume rejects changed binary/hardware inputs. Each case JSON reports
simulation cycles, time and error against NCU duration. Compare time:
`sim_us = cycles / core_clock_MHz`, not unrelated NCU active-cycle counters.

Medium additionally checks all O/LSE elements against a CPU reference.
Large checks successful execution and timing, not full numerical output.
The immutable [baseline](../baselines/fa3-sass-ccd2-20260914/README.md)
and `tests/ci/check_sass_medium.py` gate medium simulator drift at 0.5%;
this threshold is not hardware accuracy.

## Local checks

```sh
python3 tests/scripts/arch_manifests_to_make.py validate
python3 tests/ci/planner.py validate
PYTHONPATH=tests python3 -m unittest discover -s tests/runner/tests -v
CI_JOB=sm90-sass-medium-d64f bash tests/ci/run_ci_tests.sh
```

Repeat CI for d64c, d128f and d128c. For the pre-existing general queue,
see `run_sim_queue.py --help`.
