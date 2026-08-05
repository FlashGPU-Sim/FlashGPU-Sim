# TritonTrace

TritonTrace records Triton CUDA kernel compilation and launch state and generates standalone CUDA C++ harnesses. The generated harnesses can replay a captured launch on a compatible physical GPU or with FlashGPU-Sim. The public API is [`TritonTrace.Tracker`](TritonTrace/tracker.py).

## Package Structure

- `tracker.py` provides the public `Tracker` facade, selects the Online or Offline backend, controls capture with `enable()` and `disable()`, and saves the session summary.
- `online.py` observes normal Triton compilation and GPU execution through runtime hooks.
- `offline.py` intercepts Triton launches and compiles them for an explicit CUDA target without accessing a GPU driver.
- `session.py` owns kernel and launch records, serializes arguments, captures Online reference outputs, and writes reports.
- `harness.py` generates standalone launchers, Makefiles, and FlashGPU-Sim resource sidecars.
- `templates/` contains the text used by the harness generator. Files ending in `.tpl` are complete generated-file templates; files ending in `.inc` are C++ fragments inserted into those templates.

## Installation

Install TritonTrace from the repository root into the active Python environment:

```bash
python -m pip install -e tools
```

The Python environment must provide PyTorch, Triton, and NumPy. Building a generated harness also requires a CUDA Toolkit compatible with the PTX version and target emitted by Triton.

## Tracker API

```python
tracker = TritonTrace.Tracker(
    output_dir,
    save_binaries=True,
    capture_args=True,
    enabled=True,
    mode="online",
    target=None,
)
```

| Argument | Meaning |
| --- | --- |
| `output_dir` | Directory that receives captured artifacts and reports. |
| `save_binaries` | Copies compiler artifacts and generates launcher files. Keep this enabled when capturing arguments. |
| `capture_args` | Serializes tensor and scalar launch arguments and enables launch-specific harness generation. |
| `enabled` | Sets the initial launch-capture state. Its effect on a disabled launch depends on the selected mode. |
| `mode` | Selects `"online"` or `"offline"`; Online is the default. |
| `target` | Supplies the CUDA architecture for Offline mode, such as `"sm90"` or `"sm120"`. Online mode rejects this argument. |

`enable()` and `disable()` change the launch-capture state, `is_enabled()` returns it, and `save_summary()` writes `tracking_summary.json` and `tracking_report.txt`.

## Capture Modes

| Behavior | Online | Offline |
| --- | --- | --- |
| GPU required during tracking | Yes | No |
| CUDA target | Detected by Triton | Supplied through `target` |
| Kernel execution | Normal Triton execution | Compilation stops before execution |
| Autotuning | Triton can benchmark normally | One fixed `triton.Config` only |
| Reference outputs | Captured from modified tensors | Unavailable |
| Disabled kernel call | Executes without recording the launch | Is skipped |

### Online Capture

Online mode lets Triton compile and execute normally. TritonTrace registers kernel-load, launch-enter, and launch-exit hooks, and observes the Python arguments and grid passed through `JITFunction.run`. Triton evaluates callable grids with its fully bound arguments, and TritonTrace records the resulting numeric grid.

```python
from pathlib import Path

import TritonTrace

tracker = TritonTrace.Tracker(
    output_dir=Path("run/tracking"),
    enabled=False,
)

run_warmup_and_autotuning()

tracker.enable()
run_kernel_once()
tracker.save_summary()
```

Disabling Online tracking suppresses launch and argument capture while the original kernel call continues to execute. Kernel-load events remain active, so compiled kernels created during warmup or autotuning are still registered and their artifacts are copied when `save_binaries=True`. Enabling the tracker for a selected launch records its grid, block dimensions, shared-memory and scratch requirements, and arguments.

Before an enabled launch, TritonTrace serializes tensor inputs and keeps tensor snapshots. After the launch, it synchronizes the device and compares each tensor with its snapshot. Modified tensors are saved as reference outputs and included in the launch-specific validation harness.

### Offline Capture

Offline mode intercepts the same `kernel[grid](*args)` call and compiles it for an explicit CUDA architecture before Triton queries an active device:

```python
from pathlib import Path

import TritonTrace

tracker = TritonTrace.Tracker(
    output_dir=Path("run/tracking"),
    mode="offline",
    target="sm120",
)

run_kernel_once_with_cpu_tensors()
tracker.save_summary()
```

The Offline backend binds the Python arguments, derives the signature, constexpr values, compiler options, and numeric grid, and calls `triton.compile` with the selected target. It records the compiled PTX, CUBIN, compiler metadata, launch dimensions, scratch requirements, CPU tensor data, and scalar values without initializing the CUDA driver.

A plain `@triton.jit` kernel works directly. An `@triton.autotune` kernel must contain exactly one `triton.Config`, because Offline mode cannot benchmark candidate configurations. Offline capture does not execute the kernel, run runtime pre-hooks, or create reference outputs. Its launch-specific harness restores the recorded inputs and reports whether the replay completes; it has no result validation.

Offline mode can generate a launch-specific harness for PyTorch tensor arguments and `int`, `float`, or `bool` scalars. Tensor contents are saved in `.bin` files, and scalar values are written into the generated C++ code. If a call contains another Python object or pointer-like value, TritonTrace cannot recreate that argument after Python exits and skips the launch-specific harness.

## Generated Artifacts

With binary and argument capture enabled, an output directory typically contains:

```text
tracking/
├── binaries/
│   └── <kernel>_<hash>/
│       ├── <kernel>.cubin
│       ├── <kernel>.ptx
│       └── <kernel>_metadata.json
├── data/
│   ├── <kernel>_launch<N>_arg<M>.bin
│   └── <kernel>_launch<N>_arg<M>_output.bin
├── launchers/
│   ├── <kernel>_<hash>_template.cu
│   ├── <kernel>_launch<N>_harness.cu
│   ├── <kernel>_launch<N>_kernel.cubin
│   ├── <kernel>_launch<N>_kernel.ptx
│   ├── <kernel>_launch<N>_kernel.ptxinfo
│   └── <kernel>_launch<N>_Makefile
├── tracking_report.txt
└── tracking_summary.json
```

The exact set depends on the compiler artifacts and capture mode. The hash-specific `template.cu` is created as soon as a compiled kernel is registered and contains no launch arguments. The launch-specific harness is the replay entry point. An `_output.bin` file is created only by Online capture for a tensor modified by the selected launch. The `.ptxinfo` sidecar records resource usage for FlashGPU-Sim; TritonTrace uses `cuobjdump` when available and falls back to a PTX register estimate.

Triton may remove unused Python arguments from PTX. TritonTrace cannot tell which arguments were removed, so launch-specific harness generation stops with `Argument count mismatch` when the Python and PTX argument counts differ.

`tracking_summary.json` is the machine-readable record of kernels, launches, metadata, and arguments. `tracking_report.txt` presents the same session in a compact human-readable form. Both files are written when `save_summary()` is called.

## Build and Replay

Build a launch-specific harness from the generated `launchers/` directory:

```bash
make -f kernel_name_launch1_Makefile
./kernel_name_launch1
```

Replace `kernel_name` with the captured kernel name. The Makefile packages the captured PTX and CUBIN into `kernel_name_launch1_kernel.fatbin` and builds the `kernel_name_launch1` executable. These two files are build products and are not present immediately after tracking.

At runtime, the harness resolves its own directory through `/proc/self/exe`, loads the fatbin, restores tensor arguments from `data/`, allocates Triton scratch buffers, and launches the kernel with the captured grid, block, and shared-memory settings. An Online harness also compares modified tensors with the captured reference files. An Offline harness reports execution completion without checking kernel results.

To replay through FlashGPU-Sim, place a matching simulator configuration in the generated `launchers/` directory and select the simulator CUDA runtime before running the executable:

```bash
cp -a /path/to/flashgpu-sim/configs/SM90/. .
source /path/to/flashgpu-sim/setup_environment
./kernel_name_launch1
```

FlashGPU-Sim consumes the PTX and `.ptxinfo` files next to the captured module. A simulator run should report `gpu_tot_sim_cycle` in its log.

## Requirements

- Linux
- Python 3.10 or newer
- PyTorch, Triton, and NumPy
- CUDA Toolkit tools used to build a harness, including `nvcc` and `fatbinary`; `cuobjdump` improves the generated resource sidecar when available
- A compatible NVIDIA GPU and driver for Online capture or physical-GPU replay

TritonTrace is developed and tested with Python 3.12.3, PyTorch 2.9.0, Triton 3.5.0, and NumPy 2.4.0.
