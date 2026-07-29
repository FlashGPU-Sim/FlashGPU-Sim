# Triton Kernel Tracker

Track Triton kernel compilation and invocation, extract binaries, capture inputs/outputs, and generate standalone validation harnesses for GPGPU-Sim simulation.

## Features

- **Binary Extraction**: Capture PTX/CUBIN from Triton's cache
- **Argument Capture**: Serialize tensor inputs and scalars to binary files
- **Output Detection**: Automatically identify modified tensors (outputs) via snapshot comparison
- **Validation Harnesses**: Generate C++ programs that replay kernels and validate results
- **GPGPU-Sim Ready**: Compatible with cuobjdump and simulator workflows

## Files

- `track_triton_kernels.py` - Main tracking tool with output validation
- `examples/` - Single-workload examples that demonstrate tracker usage
- `validation/` - Systematic sweep tests, reference CSVs, and comparison tools
- `triton_kernel_tracking/` - Generated artifacts for both examples and validation sweeps
- `README.md` - This file

## Overview

This tool hooks into Triton's runtime to:
- Capture compiled kernel binaries (PTX/CUBIN)
- Record launch parameters (grid size, block size, shared memory)
- Save kernel arguments (inputs and outputs) as binary files
- **Detect outputs automatically** by comparing tensor states before/after execution
- Generate standalone C++ harnesses with **automatic validation**

### Key Innovation: Output Detection

Since Triton doesn't annotate which arguments are outputs, the tracker **empirically detects them**:

1. **Before launch**: Clone all tensor arguments (create snapshots)
2. **After launch**: Compare tensors with snapshots
3. **If changed** → Save as output for validation
4. **If unchanged** → Skip (input-only, no validation needed)

This eliminates false positives and ensures validation only checks actual outputs.

## Quick Start

When this repository is checked out through `flashgpu-gem5-top`, prefer the
top-level Docker image for Triton tracing and simulator/gem5 runs:

```bash
cd /flashgpu-gem5-top
make docker-build
make docker-shell
cd flashgpu-sim/test/triton_trace/validation
```

That container provides the CUDA and Python tooling used by this workflow,
including a Python 3.12 venv with `uv`, PyTorch, Triton, NumPy, and SCons. The
manual host-side venv setup below is only for working outside Docker.

Create the local Python environment first. Use Python 3.12 for the current
Torch/Triton wheels; the host default Python may be newer than supported wheels.
Install `uv` inside the venv, then use `uv pip` for the large packages:

```bash
cd test/triton_trace
python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip uv
.venv/bin/uv pip install --python .venv/bin/python torch triton numpy
```

Select a CUDA Toolkit compatible with the PTX version and target emitted by
the installed Triton release. Generated artifacts normally include a captured
CUBIN and a PTX sidecar. CUDA tool packages installed through pip are optional
fallbacks when equivalent system Toolkit tools are unavailable.

```bash
# Run the example
source .venv/bin/activate
python examples/example_vector_add.py

# Build and run the generated harness
cd triton_kernel_tracking/example_vector_add/launchers
make -f add_kernel_launch1_Makefile
./add_kernel_launch1
```

Expected output from harness:
```
Validating outputs...
  Validation PASSED for arg[2]: all 1024 elements match within tolerance 1.00e-05
```

## Example

See `examples/example_vector_add.py` for a complete working example that:
- Defines a simple vector addition kernel
- Initializes the tracker
- Runs the kernel with captured arguments
- Generates standalone harnesses

Run it:
```bash
python examples/example_vector_add.py
```

Additional examples live under `examples/`:

- `example_vector_add.py`
- `example_tensor_add.py`
- `example_tma_gemm.py`
- `example_gemm.py`
- `example_flash_attn.py`
- `example_gpt2_triton.py`

All examples write generated artifacts to the root `triton_kernel_tracking/`
directory, not to `examples/triton_kernel_tracking/`.

## Validation Sweeps

The `validation/` directory contains systematic sweep tests and reference data:

- `validation/test_tma_gemm.py` - TMA GEMM trace generator for sweep shapes
- `validation/test_flash_attn.py` - Flash Attention trace generator for sweep shapes
- `validation/sweep_tests.sh` - Trace, simulate, and profile shape sweeps
- `validation/compare_cycles.py` - Compare NCU and GPGPU-Sim cycle summaries
- `validation/extract_metrics.py` - Extract sim vs NCU memory/cache metrics
- `validation/plot_cta_lifecycle.py` - Plot CTA lifecycle timelines from sim logs
- `validation/kernel-validation.csv` - Reference validation results
- `validation/configs/` - Shape CSVs used by validation sweeps

Validation outputs are also kept under the root `triton_kernel_tracking/`
directory:

```text
triton_kernel_tracking/test_tma_gemm/
triton_kernel_tracking/test_flash_attn/
```

Typical commands:

```bash
# Run TMA GEMM inference sweep in GPGPU-Sim
./validation/sweep_tests.sh tma_gemm run --csv configs/gemm_shapes_inference_server.csv

# Compare current GEMM inference cycles against NCU summaries
python3 validation/compare_cycles.py test_tma_gemm --csv configs/gemm_shapes_inference_server.csv

# Sync simulator config into existing validation launcher directories
./validation/sync_config.sh test_tma_gemm
```

## How It Works

### Triton Runtime Hooks

The tracker uses three hooks from Triton's runtime:

```python
# 1. Capture binaries when loaded
triton.knobs.runtime.kernel_load_end_hook.add(self._on_kernel_load)

# 2. Capture inputs BEFORE launch
triton.knobs.runtime.launch_enter_hook.add(self._on_launch_enter)

# 3. Capture outputs AFTER launch
triton.knobs.runtime.launch_exit_hook.add(self._on_launch_exit)
```

### Output Detection Algorithm

```
Before Launch (launch_enter_hook):
  1. Clone all tensor args → snapshots
  2. Save input data to .bin files
  3. Generate initial harness

Kernel Executes

After Launch (launch_exit_hook):
  1. Synchronize GPU
  2. Compare each tensor with snapshot
  3. If changed → save as output
  4. Regenerate harness with validation code
```

### Generated Files

For each kernel launch, generates:
- `{kernel}_launch{N}_harness.cu` - Standalone C++ launcher **with validation**
- `{kernel}_launch{N}_kernel.ptx` - PTX source captured from Triton
- `{kernel}_launch{N}_kernel.cubin` - Compiled cubin for real-GPU replay
- `{kernel}_launch{N}_arg{M}.bin` - Input tensor data
- `{kernel}_launch{N}_arg{M}_output.bin` - Output tensor data (only if modified)
- `{kernel}_launch{N}_Makefile` - Build script

## Usage in Your Code

```python
from pathlib import Path
from track_triton_kernels import TritonKernelTracker

# Initialize tracker
tracker = TritonKernelTracker(
    output_dir=Path("./triton_tracking"),
    save_binaries=True,
    capture_args=True
)

# Run your Triton kernels normally
my_kernel[grid](x, y, output, ...)

# Save tracking data
tracker.save_summary()
```

See `examples/example_vector_add.py` for a complete example.

## Building and Running Standalone Harnesses

```bash
cd triton_kernel_tracking/example_vector_add/launchers

# Build
make -f add_kernel_launch1_Makefile

# Run
./add_kernel_launch1
```

The harness:
- Loads the cubin from file (using cuModuleLoad)
- Resolves the module image path relative to executable location (portable)
- Loads tensor arguments from .bin files
- Launches the kernel with captured parameters

## GPGPU-Sim Workflow

The generated launchers are first validated on a real GPU. For GPGPU-Sim,
confirm that `setup_environment` has selected the simulator `libcudart` and that
the run log contains `gpu_tot_sim_cycle`. If validation passes but the log has no
simulator cycle counters, the executable ran through the real CUDA runtime.

Generated Triton launchers load a CUBIN through `cuModuleLoad`; for simulation,
the runtime uses the captured PTX sidecar next to that CUBIN. Build
FlashGPU-Sim with a CUDA Toolkit compatible with the captured artifacts.

```bash
export CUDA_INSTALL_PATH=/path/to/cuda
source setup_environment
cd test/triton_trace/triton_kernel_tracking/<test>/.../launchers
./<kernel>_launch1
```

## Output Structure

```
triton_kernel_tracking/
├── example_vector_add/
│   ├── binaries/
│   │   └── add_kernel_*/
│   │       ├── add_kernel.cubin
│   │       ├── add_kernel.ptx
│   │       └── add_kernel_metadata.json
│   ├── launchers/
│   │   ├── add_kernel_launch1_harness.cu      # Harness with validation
│   │   ├── add_kernel_launch1_kernel.ptx
│   │   ├── add_kernel_launch1_kernel.cubin
│   │   └── add_kernel_launch1_Makefile
│   ├── data/
│   │   ├── add_kernel_launch1_arg0.bin        # Input tensors
│   │   ├── add_kernel_launch1_arg1.bin
│   │   ├── add_kernel_launch1_arg2.bin
│   │   └── add_kernel_launch1_arg2_output.bin # Output tensors (only modified)
│   ├── tracking_summary.json
│   └── tracking_report.txt
├── test_tma_gemm/
│   ├── m512_n3000_k1536/
│   └── results/
└── test_flash_attn/
    ├── b32_h32_seq512_d64/
    └── results/
```

## Important Notes

- **Output detection**: Automatically identifies modified tensors via snapshot comparison
- **Validation**: Only tensors that changed are validated (eliminates false positives)
- Grid size evaluated from lambda functions (not in Triton's metadata)
- PTX architecture variants (sm_120a) fixed to base version (sm_120) for nvcc
- Fatbin loaded from file at runtime (portable executables)
- Uses readlink(/proc/self/exe) for path resolution

## Requirements

- CUDA Toolkit (nvcc, cuobjdump)
- PyTorch (for tensor serialization)
- Triton

## Limitations

- Linux only (uses /proc/self/exe)
- CUDA only (no ROCm support)
- Constexpr arguments not fully captured
- Grid evaluation may fail for complex lambda functions
