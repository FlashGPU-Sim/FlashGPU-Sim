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
- `example_vector_add.py` - Complete working example
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

```bash
# Run the example
python example_vector_add.py

# Build and run the generated harness
cd triton_kernel_tracking/launchers
make -f add_kernel_launch1_Makefile
./add_kernel_launch1
```

Expected output from harness:
```
Validating outputs...
  Validation PASSED for arg[2]: all 1024 elements match within tolerance 1.00e-05
```

## Example

See `example_vector_add.py` for a complete working example that:
- Defines a simple vector addition kernel
- Initializes the tracker
- Runs the kernel with captured arguments
- Generates standalone harnesses

Run it:
```bash
python example_vector_add.py
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
- `{kernel}_launch{N}_kernel.ptx` - PTX source (sm_XXXa → sm_XXX fixed)
- `{kernel}_launch{N}_kernel.fatbin` - Compiled fatbin
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

See `example_vector_add.py` for a complete example.

## Building and Running Standalone Harnesses

```bash
cd triton_kernel_tracking/launchers

# Build
make -f add_kernel_launch1_Makefile

# Run
./add_kernel_launch1
```

The harness:
- Loads the fatbin from file (using cuModuleLoad)
- Resolves fatbin path relative to executable location (portable)
- Loads tensor arguments from .bin files
- Launches the kernel with captured parameters

## GPGPU-Sim Workflow

The generated fatbin files are compatible with cuobjdump:

```bash
# Extract PTX for GPGPU-Sim
cuobjdump --dump-ptx add_kernel_launch1_kernel.fatbin > kernel.ptx

# Use in GPGPU-Sim
# The standalone harness can be run under GPGPU-Sim's LD_PRELOAD interception
```

## Output Structure

```
triton_kernel_tracking/
├── binaries/
│   └── add_kernel_*/
│       ├── add_kernel.cubin
│       ├── add_kernel.ptx
│       └── add_kernel_metadata.json
├── launchers/
│   ├── add_kernel_launch1_harness.cu      # Harness with validation
│   ├── add_kernel_launch1_kernel.ptx
│   ├── add_kernel_launch1_kernel.fatbin
│   └── add_kernel_launch1_Makefile
├── data/
│   ├── add_kernel_launch1_arg0.bin        # Input tensors
│   ├── add_kernel_launch1_arg1.bin
│   ├── add_kernel_launch1_arg2.bin
│   └── add_kernel_launch1_arg2_output.bin # Output tensors (only modified)
├── tracking_summary.json
└── tracking_report.txt
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