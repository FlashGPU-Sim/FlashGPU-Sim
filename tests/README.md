# FlashGPU-Sim Tests

Use `run_tests.py` to select an architecture, build its workloads and run
functional or timing checks. Commands below run from the repository root;
from `tests/`, use `./run_tests.py` instead of `./tests/run_tests.py`.

## Quick Start

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda
source setup_environment

./tests/run_tests.py list
./tests/run_tests.py list-cases --arch sm100 --group unit
./tests/run_tests.py build --arch sm120 --group integration
./tests/run_tests.py run --arch sm120 --group integration
./tests/run_tests.py run --arch sm90 --group wgmma --gtest-filter 'WgmmaF16*'
```

Use a Toolkit compatible with the selected target. Some cases require a newer
Toolkit or GPU feature and report a GoogleTest skip when unavailable.

## Select Workloads

| Selector | Meaning |
| --- | --- |
| `--arch` | `sm90`, `sm100` or `sm120` |
| `--group` | Test family, such as `unit`, `integration`, `barrier`, `tma`, `mma`, `wgmma`, `fa2`, `fa3`, `microbench` or `trace`; availability depends on architecture |
| `--profile` | Build/run profile within a test family |
| `--mode` | Compile-time variant within a profile |
| `--gtest-filter` | Exact GoogleTest filter expression |
| `-c`, `--config` | Override the architecture's default simulator configuration |
| `-t`, `--timeout` | Per-run timeout in seconds |

A positional filter provides a substring search. `list` shows supported
selections; choosing a profile expands its available modes.

```bash
./tests/run_tests.py list --arch sm90 --group fa3 --profile breakdown
./tests/run_tests.py run --arch sm100 --group unit
./tests/run_tests.py run --arch sm120 --group tma TmaProducerConsumerTest
./tests/run_tests.py run --arch sm90 --group fa2 --profile smoke
./tests/run_tests.py run --arch sm90 --group fa3 --profile breakdown --mode baseline
./tests/run_tests.py run --arch sm120 --group trace --profile gpt2 flash_attn
```

`list-cases` builds the selected GoogleTest binaries when needed and prints
fully qualified `Suite.Case` names. Case names go to stdout; build progress goes
to stderr. Trace programs and standalone build-only selections have no
GoogleTest case list.

```bash
./tests/run_tests.py list-cases --arch sm120 --group integration \
  --gtest-filter '*VectorAdd*'
./tests/run_tests.py run --arch sm120 --group integration \
  --gtest-filter 'CudaVectorAddTest.BasicVectorAddition'
```

## Build-Only Selections

`--mode all` aggregates compilation variants without running them.
`build --group all` builds every supported selection for an architecture and
can be substantially heavier than a focused build.

```bash
./tests/run_tests.py build --arch sm90 --group all
./tests/run_tests.py build --arch sm100 --group all
./tests/run_tests.py build --arch sm120 --group all
```

Standalone calibration profiles such as SM90 `cp-async`/`mma`/`tma`, SM100
`tma`, and SM120 `memory` are build-only; run their binaries with the
benchmark-specific arguments described in their guides. GoogleTest
microbenchmarks in SM90 `wgmma` and SM120 `mma` can run through the runner.

```bash
./tests/run_tests.py build --arch sm100 --group microbench --profile tma
./tests/run_tests.py run --arch sm90 --group microbench --profile wgmma WgmmaN16Chain
./tests/run_tests.py run --arch sm120 --group microbench --profile mma MMAIssueSummary
```

## Test Families

| Family | Coverage and usage |
| --- | --- |
| `unit` | Simulator component checks |
| [integration](src/integration/README.md) | PTX execution, addressing and CUDA runtime behavior |
| `barrier`, `tma` | Asynchronous synchronization and data movement |
| [mma](src/mma/README.md), [wgmma](src/wgmma/README.md) | Matrix instructions checked against CPU references |
| [fa2](src/fa2/README.md), [fa3](src/fa3/README.md) | FlashAttention workloads and analysis profiles |
| `microbench` | [MMA](src/microbench/mma/README.md), [WGMMA](src/microbench/wgmma/README.md), [cp.async](src/microbench/cp_async/README.md) and [memory](src/microbench/memory/README.md) probes |
| `trace` | Standalone GPT-2 kernel replay |

Builds fetch GoogleTest on demand. FA2/FA3 builds prepare the pinned
[FlashAttention dependency](third_party/flash-attention/README.md) automatically.
DSL capture/replay workflows have separate guides:
[TritonTrace](dsl/triton/README.md) and [CutedslTrace](dsl/cutedsl/README.md).

For native GPU checks, use a clean shell without `setup_environment` sourced
and a GPU matching the compiled target. Simulator runs use the environment
configured by `setup_environment`; avoid mixing native and simulator libraries.

## CI

The [CI selections](ci/jobs.toml) and [PR workflow](../.github/workflows/pr-tests.yml)
currently define five functional jobs:

| Job | Coverage |
| --- | --- |
| `sm100-core` | Unit, integration, barrier and TMA |
| `sm120-core` | Unit, integration, barrier, TMA, MMA and GPT-2 trace |
| `sm90-core` | Unit, integration, barrier and WGMMA |
| `sm90-fa2` | Numerically checked forward smoke cases |
| `sm90-fa3` | Forward/backward smoke, fixed forward and PackGQA |

Run a CI job locally:

```bash
CI_JOB=sm100-core ./tests/ci/run_ci_tests.sh
CI_JOB=sm120-core ./tests/ci/run_ci_tests.sh
CI_JOB=sm90-core ./tests/ci/run_ci_tests.sh
CI_JOB=sm90-fa2 ./tests/ci/run_ci_tests.sh
CI_JOB=sm90-fa3 ./tests/ci/run_ci_tests.sh
CI_JOB=all ./tests/ci/run_ci_tests.sh
```

Logs and GoogleTest XML go under `tests/logs/ci/`. GitHub Actions uploads these
when a functional job fails and reports failed or aborted cases.

Inspect the plan without building or simulating:

```bash
./tests/ci/planner.py list-jobs
./tests/ci/planner.py plan --job sm90-fa3
```

Separate informational cycle-validation jobs replay the two tutorials, TMA
GEMM and Llama3 kernels against stored NCU cycle references. They do not feed
`verify-status`. Their scope is defined in [cases.toml](ci/perf/cases.toml).

```bash
python3 tests/ci/perf/perf.py validate
python3 tests/ci/perf/perf.py matrix
```

For broader Triton FlashAttention validation:

```bash
bash tutorials/triton-flash-attention/capture.sh
bash tutorials/triton-flash-attention/run.sh
```
