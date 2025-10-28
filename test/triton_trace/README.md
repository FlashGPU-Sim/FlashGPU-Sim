# Triton Kernel Tracker

Track Triton kernel compilation and invocation, extract binaries and generate standalone harnesses for GPGPU-Sim simulation.

## Files

- `track_triton_kernels.py` - Main tracking tool
- `example_vector_add.py` - Complete working example
- `README.md` - This file

## Overview

This tool hooks into Triton's runtime to:
- Capture compiled kernel binaries (PTX/CUBIN)
- Record launch parameters (grid size, block size, shared memory)
- Save kernel arguments (tensors and scalars)
- Generate standalone C++ harnesses that can run the kernel independently

## Quick Start

```bash
# Run the example
python example_vector_add.py

# Or use the tracker directly
python track_triton_kernels.py --output-dir ./my_output
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

The tracker patches Triton's JITFunction.run to capture:
1. Kernel arguments (before launch)
2. Grid size (evaluated from lambda functions)
3. Kernel binaries from Triton's cache

For each kernel launch, it generates:
- {kernel}_launch{N}_harness.cu - Standalone C++ launcher
- {kernel}_launch{N}_kernel.ptx - PTX source (with sm_XXXa -> sm_XXX fix)
- {kernel}_launch{N}_kernel.fatbin - Compiled fatbin
- {kernel}_launch{N}_arg{M}.bin - Serialized tensor data
- {kernel}_launch{N}_Makefile - Build script

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
│   ├── add_kernel_launch1_harness.cu
│   ├── add_kernel_launch1_kernel.ptx
│   ├── add_kernel_launch1_kernel.fatbin
│   └── add_kernel_launch1_Makefile
├── data/
│   ├── add_kernel_launch1_arg0.bin
│   ├── add_kernel_launch1_arg1.bin
│   └── add_kernel_launch1_arg2.bin
├── tracking_summary.json
└── tracking_report.txt
```

## Important Notes

- Grid size is NOT in Triton's launch metadata by default - we evaluate it from the grid lambda
- PTX architecture variants (sm_120a) are fixed to base version (sm_120) for nvcc compatibility
- Fatbin is loaded from file at runtime, not embedded in executable
- Executable uses readlink(/proc/self/exe) to find fatbin, so it's portable
- Only CUDA tensors are captured (CPU tensors are not supported yet)

## Requirements

- CUDA Toolkit (nvcc, cuobjdump)
- PyTorch (for tensor serialization)
- Triton (obviously)

## Limitations

- Linux only (uses /proc/self/exe for path resolution)
- CUDA only (no ROCm support yet)
- Constexpr arguments are not fully captured
- Grid evaluation may fail for complex lambda functions

├── ptx/                        # PTX files for GPGPU-Sim
├── harness/                    # Standalone C++ launchers
│   ├── kernel_harness.cu
│   ├── kernel_Makefile
│   └── kernel_README.md
├── gpgpusim_summary.json      # All metadata
└── README.md                   # Overview
```

## 🔗 Integration with Triton

These tools use Triton's hook system to non-invasively track kernel compilation and execution:

- `triton.knobs.runtime.kernel_load_end_hook` - Captures binaries
- `triton.knobs.runtime.launch_enter_hook` - Captures launch parameters
- `triton.knobs.runtime.jit_post_compile_hook` - Captures compilation

---

**Version**: 1.0  
**Date**: October 2025  
**Triton Version**: 3.5.0+
