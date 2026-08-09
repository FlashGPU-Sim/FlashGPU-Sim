# FlashGPU-Sim Tests

The test system keeps hardware architecture selection separate from test
organization. Test sources describe workloads; architecture manifests select
the sources supported by each GPU target.

## Quick start

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda
source setup_environment

./tests/run_tests.py list
./tests/run_tests.py list --arch sm90 --group fa3 --profile breakdown
./tests/run_tests.py list-cases --arch sm120 --group integration
./tests/run_tests.py list-cases --arch sm120 --group integration \
  --gtest-filter '*VectorAdd*'
./tests/run_tests.py build --arch sm120 --group integration
./tests/run_tests.py run --arch sm120 --group integration
./tests/run_tests.py run --arch sm90 --group wgmma \
  --gtest-filter 'WgmmaF16*'
```

The runner resolves repository paths internally, so it works from either the
repository root or the `tests/` directory.

`run_tests.py` is the only public entry point. Its implementation is split by
responsibility under `tests/runner/`: selection resolution, Make orchestration,
GoogleTest discovery/filtering, and runtime executors. CI invokes the same
Python entry point directly; there is no shell compatibility wrapper.

## Public hierarchy

```text
architecture manifest
└── selected test sources
    └── test group / binary
        └── GoogleTest suite
            └── GoogleTest case
```

The public selectors are:

- `--arch`: an architecture manifest, currently `sm90` or `sm120`;
- `--group`: the first directory component below `tests/src/`;
- `--profile`: an optional build/run profile for a complex test group;
- `--mode`: an optional compile-time variant inside a profile; and
- `--gtest-filter`: an exact GoogleTest filter expression.

A positional filter remains available as a convenient substring search.

`list` uses progressive discovery: without `--profile` it stops at the profile
level; selecting one profile expands its compile-time modes.

```bash
./run_tests.py run --arch sm120 --group unit
./run_tests.py run --arch sm120 --group tma CudaTMATest
./run_tests.py run --arch sm90 --group fa2 --profile smoke
./run_tests.py run --arch sm120 --group fa2 --profile smoke
./run_tests.py run --arch sm90 --group fa3 \
  --profile breakdown --mode baseline
./run_tests.py build --arch sm90 --group microbench --profile tma
./run_tests.py run --arch sm120 --group trace \
  --profile gpt2 flash_attn
```

Mode `all` and standalone calibration microbenchmarks are build-only.

Build every selection supported by one architecture:

```bash
./tests/run_tests.py build --arch sm120 --group all
./tests/run_tests.py build --arch sm90 --group all
```

`all` is a runner-level aggregate, not a test group stored in the TOML
manifest. The runner discovers the architecture's groups dynamically, builds
every profile, selects `mode=all` for profiles that provide an aggregate mode,
and deduplicates selections that resolve to the same Make target. It builds but
does not run the resulting tests. Because this includes FA2/FA3 analysis modes,
microbenchmarks, and trace programs, it can be substantially heavier than a
focused test-group build.

## Source layout

The common path form is:

```text
tests/src/<test_group>/<test>.cu
```

Pure host-side C++ tests use `.cc` instead. A test group is a source and
default binary boundary; it may contain many independent workloads and many
GoogleTest cases.

```text
src/
├── include/       shared test-only headers (not a test group)
├── unit/          host-side simulator component tests
├── integration/   cross-architecture standalone CUDA tests
├── barrier/       named barrier and mbarrier tests
├── tma/           TMA and tensor-map tests
├── mma/           mma.sync tests
├── wgmma/         WGMMA tests
├── fa2/           FlashAttention 2 tests and build variants
├── fa3/           FlashAttention 3 tests and build variants
├── microbench/    existing microbenchmark layout
└── trace/         existing trace-driven GPT-2 tests
```

Executable sources follow `tests/src/<test_group>/<test>.cu`; shared headers
under `tests/src/include/` are included by those sources but are not workloads,
test groups, or architecture-manifest entries.

`integration/` admits only standalone sources that compile for every supported
architecture without test-specific compiler flags or link dependencies.

## Architecture manifests

Architecture support is declared in readable TOML files:

```text
arch/
├── sm90.toml
└── sm120.toml
```

Each file contains only one NVCC target, one simulator configuration, and an
explicit source list:

```toml
[arch]
nvcc_target = "sm_90a"
config = "SM90_H100"

[workloads]
sources = [
  "integration/cp_async_src_size_test.cu",
  "barrier/named_barrier_test.cu",
  "wgmma/wgmma_b1_test.cu",
]
```

There is no per-test target override. The manifest reader derives compute
capability from `nvcc_target`, verifies the simulator configuration, checks
every source path, and requires every architecture to include the complete
portable `integration/` source set.

The first path component is the test group. Ordinary groups are built with
shared rules into architecture-isolated artifacts:

```text
build/obj/<arch>/<test_group>/...
build/bin/<arch>/<test_group>_tests
```

FA2, FA3, microbenchmarks, and trace tests retain specialized recipes under
`make/workloads/` when they require repeated compilation, profiles, multiple
binaries, or a non-GTest executor. Those recipes do not own architecture
membership and cannot override the manifest's NVCC target.

Useful machine-readable queries include:

```bash
make -s list-architectures
make -s list-test-groups ARCH=sm90
make -s print-architecture-metadata ARCH=sm90
make -s print-test-group-sources ARCH=sm90 TEST_GROUP=wgmma
make -s list-test-group-profiles ARCH=sm90 TEST_GROUP=fa3
make -s list-test-group-modes \
  ARCH=sm90 TEST_GROUP=fa3 PROFILE=scaling
```

## GoogleTest and the repository runner

GoogleTest owns fixtures, assertions, parameterized cases (`TEST_P`), filters,
discovery, and XML output inside each binary. It does not coordinate multiple
binaries or hardware platforms. The repository runner reads the architecture
selection, builds the required binary set, prepares the simulator
configuration, and launches the binaries.

List every fully qualified GoogleTest case registered by one runner selection:

```bash
./tests/run_tests.py list-cases \
  --arch sm120 \
  --group integration

./tests/run_tests.py list-cases \
  --arch sm120 \
  --group integration \
  --gtest-filter '*VectorAdd*'

./tests/run_tests.py list-cases \
  --arch sm120 \
  --group fa2 \
  --profile smoke
```

`list-cases` builds the selected binary set when necessary, merges discovery
output from all of its binaries, intersects `--gtest-filter` with the profile's
default filter, and prints one `Suite.Case` name per line. Parameterized names
retain their full GTest prefix and parameter suffix. Trace selections and
build-only selections without a GTest binary manifest do not have a case list;
a mode such as `all` can still be listed when its metadata identifies the
generated GTest binaries. Case names are written to stdout; configuration and
build progress are written to stderr, so the output can be redirected or piped
without extra filtering.

Use one of those names as an exact runtime selection:

```bash
./tests/run_tests.py run \
  --arch sm120 \
  --group integration \
  --gtest-filter 'CudaVectorAddTest.BasicVectorAddition'
```

## CI

PR CI jobs and their runner selections live in `tests/ci/jobs.toml`. Each job
selects one architecture manifest and an explicit list of test-group,
profile, mode, and optional GoogleTest-filter combinations:

```toml
[jobs.sm120-core]
arch = "sm120"
tests = [
  { name = "unit", group = "unit" },
  { name = "integration", group = "integration" },
  { name = "trace", group = "trace", profile = "gpt2" },
]
```

The architecture TOML remains the only source-membership list. The CI manifest
references its groups rather than repeating source paths, and the planner
rejects a job whose group/profile/mode is unsupported by that architecture.
GitHub Actions obtains its dynamic job matrix from the same file.

Run one CI job locally, or run the complete manifest:

```bash
CI_JOB=sm120-core ./tests/ci/run_ci_tests.sh
CI_JOB=sm90-core ./tests/ci/run_ci_tests.sh
CI_JOB=sm90-fa2 ./tests/ci/run_ci_tests.sh
CI_JOB=sm90-fa3 ./tests/ci/run_ci_tests.sh
CI_JOB=all ./tests/ci/run_ci_tests.sh
```

Logs and GoogleTest XML results are written below `tests/logs/ci/`. GitHub
Actions uploads them only when a `run-tests` job fails. Failed jobs also parse
the available XML into a compact job summary that lists failed cases only. If
a test process aborts before writing XML, the summary recovers the active
GoogleTest case from its log and reports it as `Aborted`. Successful jobs
produce neither an artifact nor a test summary.

The planner can be inspected without building or running tests:

```bash
./tests/ci/planner.py list-jobs
./tests/ci/planner.py plan --job sm90-fa3
CI_JOB=all CI_LIST_JOBS=1 ./tests/ci/run_ci_tests.sh
```

Cycle-validation jobs are defined separately in `ci/perf/cases.toml`. Each
top-level table names one simulator configuration and therefore one GitHub
Actions job; its cases run sequentially in that job. A case provides a stable
ID, a summary label, one or more repository-relative scripts, and its NCU cycle
reference:

```toml
[SM120_RTX5090.cases.cuda-vector-add-2m]
label = "Tutorial - CUDA Vector Add"
scripts = ["tutorials/vectorAdd/run.sh"]
ncu_cycles = 29642.67
```

Inspect the performance plan without building or simulating:

```bash
python3 tests/ci/perf/perf.py validate
python3 tests/ci/perf/perf.py matrix
```

GitHub Actions builds the simulator once per configured performance job, runs
its cases in manifest order, and compares `gpu_tot_sim_cycle` with the stored
NCU value. The config jobs are informational and do not feed `verify-status`.
Their small JSON results are combined by `cycle-validation-report`; complete
logs are uploaded only for failed config jobs.
