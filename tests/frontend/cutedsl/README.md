# CutedslTrace examples and validation

This directory parallels `tests/frontend/triton/`. The generic exporter lives in
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
