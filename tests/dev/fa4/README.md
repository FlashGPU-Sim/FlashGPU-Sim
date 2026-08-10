# FA4 B200 Development Workloads

## Usage

Run commands from the repository root. Point `FA4_PYTHON` at an environment
that contains PyTorch, CuTe DSL, and FA4:

```bash
export FA4_PYTHON=/path/to/fa4-venv/bin/python
```

Run one case through export, runner build, simulation, and result checking:

```bash
tests/dev/fa4/run_fa4_b200_cases.sh run \
  --config SM100_B200_REDUCED \
  --case H32D64CausalB2S128
```

Reuse its exported artifacts and run the simulation again:

```bash
tests/dev/fa4/run_fa4_b200_cases.sh sim \
  --config SM100_B200_REDUCED \
  --case H32D64CausalB2S128
```

Export artifacts without running the simulator:

```bash
tests/dev/fa4/run_fa4_b200_cases.sh export \
  --config SM100_B200_REDUCED \
  --case H32D64CausalB2S128
```

Run every specialization in the smoke suite:

```bash
tests/dev/fa4/run_fa4_b200_suite_matrix.sh \
  --config SM100_B200_REDUCED \
  --suite smoke
```

The two FA4 scripts remain the user-facing entry points: use
`run_fa4_b200_cases.sh` for one case and `run_fa4_b200_suite_matrix.sh` for a
suite. The suite script builds the task list and calls the generic queue
internally.

By default, each workload is pinned to four idle physical cores and a suite
runs at most four workloads at once. Use `--max-parallel N`,
`--cpus-per-job N`, and `--threads-per-job N` to tune the suite. Use
`--cpu-sets LIST...` for manual worker slots or `--no-pin` to let each workload
use all CPUs allowed to the suite process. A direct case accepts the same core
and thread counts, singular `--cpu-set LIST`, and `--no-pin`.
Press `Ctrl+C` once to stop the queue and every active workload; press it again
to force termination if a workload does not respond to `SIGTERM`.

Run the complete small suite. Simulations have no timeout by default:

```bash
tests/dev/fa4/run_fa4_b200_suite_matrix.sh \
  --config SM100_B200_REDUCED \
  --suite small
```

Pass `--timeout SECONDS` when a finite per-case limit is needed.

Add `--rebuild-launcher` to `run` or `export` only when the FA4/CuTe DSL
export must be regenerated. Add `--verbose` to stream full tool output.

Single-case results are stored under
`temp/fa4-b200-runs/<case>/<config>/`. Suite case results and queue summaries
are stored under `temp/fa4-b200-suite-matrix/`. Each case directory contains
logs and a latest `run-manifest.json`; `latest-queue.txt` identifies the latest
suite queue.

Print the simulation cycles from the latest smoke run:

```bash
jq -r '
  .cases[]
  | select(.result.status == "pass")
  | "\(.name)\t\(.result.statistics.cycles)"
' temp/fa4-b200-suite-matrix/fwd_smoke_*/*/run-manifest.json
```

## B200 Hardware Profiling

Build the upload bundle locally. Portable mode uses a Debian 11 build container
so the AOT runners do not inherit the development host's newer glibc:

```bash
tests/dev/calibration/prepare_b200_profiling/prepare_fa4_b200_ncu_bundle.sh \
  --output temp/fa4-b200-ncu-bundle
```

Upload the printed `.tar.gz` and its `.sha256` to an x86_64 host with a full,
idle B200, a compatible driver, `nvidia-smi`, and `ncu` (preferably 2026.1.1).
No Python, CUDA toolkit, compiler, or headers are required remotely. Unpack it
and run all 32 cases serially:

```bash
sha256sum -c fa4-b200-ncu-*.tar.gz.sha256
mkdir -p ~/fa4-b200-profile
tar -xzf fa4-b200-ncu-*.tar.gz \
  -C ~/fa4-b200-profile --strip-components=1
cd ~/fa4-b200-profile
./run_remote.sh all --device 0
```

Copy the uncompressed result `.tar` and `.sha256` paths printed by the remote
script back to this machine, then summarize cycles and the measured base clock:

```bash
tests/dev/calibration/prepare_b200_profiling/analyze_fa4_b200_ncu.py \
  --archive temp/fa4-b200-hardware-archive/fa4-b200-profile-*.tar \
  --output temp/fa4-b200-hardware-archive/derived
```

The remote sweep has no timeout by default and is resumable. Each case retains
its `.ncu-rep`, raw/session CSV, detailed text export, CUDA-event timing, output
check, GPU snapshots, and checksums.

## Call Flow

```text
run_fa4_b200_suite_matrix.sh
  -> run_sim_queue.py: default four CPU-pinned case workers
  -> run_fa4_b200_cases.sh for each selected CSV row

run_fa4_b200_cases.sh
  -> export_fa4_b200_launcher.py
       -> CuTe DSL export_to_c
       -> launcher .h, host .o, PTX, cubin, and metadata
  -> g++: fa4_b200_cute_launcher_runner.cc + launcher .o
  -> fa4_b200_cute_launcher_runner
       -> allocate and initialize Q/K/V/O
       -> call the generated CuTe DSL wrapper
       -> construct TensorMaps, pack kernel arguments, and launch
  -> GPGPU-Sim libcudart
       -> load the sidecar PTX instead of the embedded cubin
       -> simulate using the selected gpgpusim.config
  -> write_fa4_b200_run_manifest.py
       -> record artifacts, CPU affinity, logs, checks, and simulator statistics
```

`fa4_b200_cases.csv` is the workload table used by both the single-case driver
and the suite matrix.
