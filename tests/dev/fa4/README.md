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

Add `--rebuild-launcher` to `run` or `export` only when the FA4/CuTe DSL
export must be regenerated. Add `--verbose` to stream full tool output.

Single-case results are stored under
`temp/fa4-b200-runs/<case>/<config>/`. Suite results are stored under
`temp/fa4-b200-suite-matrix/`. Each run directory contains logs and a latest
`run-manifest.json`.

Print the simulation cycles from the latest smoke run:

```bash
jq -r '
  .cases[]
  | select(.result.status == "pass")
  | "\(.name)\t\(.result.statistics.cycles)"
' temp/fa4-b200-suite-matrix/fwd_smoke_*/run-manifest.json
```

## Call Flow

```text
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
       -> record artifacts, logs, checks, and simulator statistics
```

`fa4_b200_cases.csv` is the workload table used by both the single-case driver
and the suite matrix.
