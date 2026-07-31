# TritonTrace

TritonTrace captures selected Triton kernel launches and generates standalone
CUDA C++ harnesses that can be replayed on a physical GPU or with
FlashGPU-Sim. The implementation is provided by
[`tritontrace.py`](tritontrace.py).

For bundled examples and validation sweeps, see
[`test/triton_trace/`](../test/triton_trace/README.md).

## Capabilities

- Capture PTX and CUBIN images from Triton's kernel cache.
- Record launch parameters, including grid and block dimensions and dynamic
  shared-memory size.
- Serialize tensor and scalar arguments.
- Detect modified tensor arguments and save their post-launch values as
  reference outputs.
- Generate standalone CUDA C++ harnesses that replay launches and validate
  captured outputs.

## Installation

From the repository root, install TritonTrace into the active Python
environment:

```bash
python -m pip install -e tools
```

The capture environment must also provide PyTorch, Triton, and NumPy. Select a
CUDA Toolkit compatible with the PTX version and target emitted by the
installed Triton release.

## Capture a Kernel Launch

Create a tracker around the Triton launch to capture:

```python
from pathlib import Path

import tritontrace

tracker = tritontrace.Tracker(
    output_dir=Path("run/tracking"),
    save_binaries=True,
    capture_args=True,
)

tracker.disable()
run_warmup_and_autotuning()

tracker.enable()
run_kernel_once()
tracker.save_summary()
```

Keep tracking disabled during compilation, warmup, and autotuning. Enable it
only for the launch that will be replayed; otherwise the tracker may capture
multiple candidate kernels or autotuning launches.

## How Capture Works

TritonTrace registers three hooks with Triton's runtime:

```python
triton.knobs.runtime.kernel_load_end_hook.add(self._on_kernel_load)
triton.knobs.runtime.launch_enter_hook.add(self._on_launch_enter)
triton.knobs.runtime.launch_exit_hook.add(self._on_launch_exit)
```

At kernel load, TritonTrace records the compiled kernel images. Immediately
before launch, it saves the launch parameters and argument values and snapshots
tensor arguments. After launch, it compares each tensor with its snapshot.
Modified tensors are recorded as reference outputs and included in generated
validation code.

## Generated Artifacts

For each captured launch, TritonTrace generates:

- `{kernel}_launch{N}_harness.cu`: standalone CUDA C++ harness with validation.
- `{kernel}_launch{N}_kernel.ptx`: PTX captured from Triton.
- `{kernel}_launch{N}_kernel.cubin`: CUBIN used for native-GPU replay.
- `{kernel}_launch{N}_arg{M}.bin`: serialized argument data.
- `{kernel}_launch{N}_arg{M}_output.bin`: reference data for a modified tensor.
- `{kernel}_launch{N}_Makefile`: harness build rules.

A typical output directory has this structure:

```text
tracking/
├── binaries/
│   └── kernel_*/
│       ├── kernel.cubin
│       ├── kernel.ptx
│       └── kernel_metadata.json
├── data/
│   ├── kernel_launch1_arg0.bin
│   └── kernel_launch1_arg1_output.bin
├── launchers/
│   ├── kernel_launch1_harness.cu
│   ├── kernel_launch1_kernel.cubin
│   ├── kernel_launch1_kernel.ptx
│   └── kernel_launch1_Makefile
├── tracking_report.txt
└── tracking_summary.json
```

## Build and Replay a Harness

Build the generated harness from its `launchers/` directory:

```bash
make -f kernel_launch1_Makefile
./kernel_launch1
```

The harness:

- loads the captured CUBIN with `cuModuleLoad`;
- restores tensor arguments from the generated binary files;
- launches the kernel with the captured parameters; and
- compares modified outputs against the captured reference values.

The generated harness should first be run on a compatible physical GPU to
confirm that the captured launch can be reconstructed correctly.

## Replay with FlashGPU-Sim

To replay with FlashGPU-Sim, copy a matching simulator configuration into the
generated `launchers/` directory and select the simulator CUDA runtime:

```bash
cp -a /path/to/flashgpu-sim/configs/<config>/. .
source /path/to/flashgpu-sim/setup_environment
./kernel_launch1
```

The harness loads the captured module through `cuModuleLoad`; during simulation,
FlashGPU-Sim uses the captured PTX file next to the CUBIN. Confirm that the run
log contains `gpu_tot_sim_cycle`. A validation result without simulator cycle
counters indicates that the harness used the physical GPU runtime instead.

## Requirements

- Linux
- Python, PyTorch, Triton, and NumPy
- CUDA Toolkit tools, including `nvcc` and `cuobjdump`

TritonTrace is developed and tested with Python 3.12.3, PyTorch 2.9.0,
Triton 3.5.0, and NumPy 2.4.0.

## Limitations

- CUDA only; ROCm is not supported.
- Generated harnesses use `/proc/self/exe` for path resolution.
- `constexpr` arguments are not fully captured.
- Grid evaluation may fail for complex lambda expressions.
