# CutedslTrace examples and validation

This directory parallels `tests/dsl/triton/`. The generic exporter lives in
[`tools/CutedslTrace`](../../../tools/CutedslTrace/README.md); workloads and their
configuration live here. These standalone tests are not registered in `run_tests.py`.

| Directory | Contents |
| --- | --- |
| `examples/` | Minimal vector-add export and exact output check |
| `validation/` | Normal FA4 Python invocation captured with `CutedslTrace.Tracker.enable()` / `export()` |
| `validation/configs/` | FA4 forward smoke/small/medium/large case definitions |
| `exports/` | Generated harnesses, Makefiles, modules and data; ignored by Git |

From the repository root, install with `python -m pip install -e tools` into a
Python environment with PyTorch, CuTe DSL and TVM
FFI. FA4 additionally requires `flash-attn-4` (tested revision
`d80a77103021c4e980f8cbbf85774f6a19e6474a`, CuTe DSL 4.5.2).
Run the following from this directory:

```bash
python examples/example_vector_add.py
make -C exports/vector_add
# In a shell without the simulator environment, this runs on a matching GPU:
exports/vector_add/replay

python validation/fa4_forward.py --suite smoke --list
python validation/fa4_forward.py --suite smoke
make -C exports/fa4/smoke/H16D128CausalB2S128
```

Export is offline and needs no GPU. The vector example defaults to `sm_120a` and 170 SMs for RTX 5090. FA4 currently targets `sm_100a`, with
`--sm-count 148` for B200. Its native replay needs a matching GPU.
FA4 uses Q=K=0 and V=1 and checks O=1; this is a simple functional check,
not random-input numerical validation.

| FA4 suite | Batch × sequence length | Cases |
| --- | --- | ---: |
| smoke | 2 × 128 | 4 |
| small | 32 × 256 | 4 |
| medium | 16 × 512 | 4 |
| large | 64 × 512, 32 × 1024, 16 × 2048, 8 × 4096, 4 × 8192 | 20 |

Each shape covers H32/D64 and H16/D128, causal and non-causal, in FP16.
`--suite all` selects all cases; `--case NAME` selects one. Each case is exported
in a fresh process into `exports/fa4/<suite>/<case-name>/` by default, resolved
relative to this directory regardless of the current working directory.
With no selection arguments, all four suites (32 cases) are exported.
Explicit manual shape flags instead export a single case to
`exports/fa4/<shape-name>/`.
`--output-dir` overrides the destination (preset selection appends the case name).
Manual shape flags remain available without preset selection.

For simulation, prepare a config in the execution directory and source the
simulator environment, with `CUDA_INSTALL_PATH` set to your CUDA Toolkit.
From this directory, for example:

```bash
repo_root="$(cd ../../.. && pwd)"
case_dir="$PWD/exports/fa4/smoke/H16D128CausalB2S128"
cp "$repo_root"/configs/SM100_B200/{gpgpusim.config,config_ampere_islip.icnt,sass_primary_hints.rules} "$case_dir/"
source "$repo_root/setup_environment"
cd "$case_dir"
./replay
```

Use the same built executable for native and simulated replay. Native profiling
can wrap `./replay` with NCU in a shell without the simulator environment.
The exporter does not create simulator configs or automatically run simulations.

If `ptxas` rejects the exported PTX version, set `PTXAS_CUDA_INSTALL_PATH`
to a Toolkit that supports that version before running the replay.

## FA4 stage controller

From this directory, use the current Python environment:

```bash
python validation/run_fa4.py track --suite smoke
python validation/run_fa4.py compile --suite smoke
python validation/run_fa4.py replay --case H16D128CausalB2S128
python validation/run_fa4.py all --shape 2,16,128,128,true
python validation/run_fa4.py compile --suite all --list
```

Stages are explicit: `compile` requires an existing export; `replay` requires a
compiled executable. `all` performs all three stages. Selection defaults to all
32 presets. Repeat `--suite`, `--case`, or `--shape B,H,S,D,true|false` to select
multiple cases. A shape matching a preset reuses its existing directory; other
shapes use `custom/<case-name>`.

Exports remain in `exports/fa4/<suite>/<case>/`. Each controller invocation creates
`runs/fa4/<timestamp>/<suite>/<case>/` for stage logs, replay configuration and
output data, plus a run-level `summary.json`. Replay copies the executable and
links input artifacts, leaving generated result files inside the run directory.
`--export-root` and `--run-root` override these roots.

Replay defaults to simulation: it copies files from `configs/SM100_B200` (override
with `--config-dir`) and sources this repository's simulator environment in a
child shell. Set `CUDA_INSTALL_PATH` and, if required, `PTXAS_CUDA_INSTALL_PATH`
in the caller's environment. Simulations use up to four workers (`--jobs N`), limited by the available
physical cores. Whole-machine memory is checked every second: new replays wait
at the configured limit (default 50 GiB; override with `--memory-limit-gib N`), and above that limit the newest active replay is stopped. A stage
failure stops the run and its remaining child processes.
For hardware replay, use `--native` from a shell without the simulator environment
on a matching B200 GPU. The controller stops on the first failed stage and records
the failure in `summary.json`.

Simulated replay uses `tests/scripts/cpu_affinity.py` to assign each worker four physical
cores based on topology and current load. Worker CPU sets do not overlap. `taskset` applies the
CPU set and `OMP_NUM_THREADS` matches its size. Override with `--cpus-per-job N`
or an explicit `--cpus 0,2,4-5` within the caller's allowed CPUs (an explicit CPU set selects one worker). Selected CPUs
are recorded in `summary.json`. Track, compile and native replay retain the
caller's affinity.

Track and compile remain sequential; `all` finishes these stages before parallel
simulation. Native replay is sequential to avoid GPU contention.
