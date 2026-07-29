# Build Instructions for GPGPU-Sim

This document provides instructions for building the GPGPU-Sim simulator.

## Prerequisites

Before building, ensure the environment is properly configured.

## Build Steps

### 1. Configure the Environment

```bash
export CUDA_INSTALL_PATH=/path/to/cuda
source setup_environment
```

`CUDA_INSTALL_PATH` must name the CUDA Toolkit root containing `bin/nvcc`.

### 2. Build the Project

```bash
make FLASH=1
```

The `FLASH=1` flag enables FLASH-specific features.

### 3. Clean Build (if needed)

```bash
make clean
make FLASH=1
```

## Build Artifacts

After a successful build, the following artifacts will be generated:
- `lib/gcc-*/cuda_runtime_api.so` - CUDA runtime library
- Various object files in `build/` directories

## Common Build Issues

### Missing Dependencies

If you encounter missing dependency errors, ensure all prerequisites are installed:
- GCC/G++ compiler
- CUDA Toolkit (for PTX parsing)
- Flex and Bison (for lexer/parser generation)
- Python (for build scripts)

### PTX Parser Generation

The PTX parser (`src/cuda-sim/ptx.tab.c`, `ptx.tab.h`) is generated from:
- `src/cuda-sim/ptx.l` (lexer)
- `src/cuda-sim/ptx.y` (parser grammar)

If you modify these files, the parser will be regenerated during build.

### Opcode Definition Changes

If you modify `src/cuda-sim/opcodes.def`, the build system will regenerate:
- Opcode switch statements in various files
- Instruction dispatcher logic

This is done automatically during the build process.

## Testing

After building, run tests:

```bash
./test/run_tests.sh test                    # Run all verification tests
./test/run_tests.sh test "*TensorMMA*"      # Run specific tests by pattern
```

## Incremental Builds

For faster development cycles, the build system supports incremental builds:
- Only modified files and their dependencies are recompiled
- Clean builds are only needed when:
  - Switching branches with significant changes
  - Modifying build configuration
  - Encountering unexplained build errors

## Build Time

Typical build times (on modern hardware):
- Clean build: 5-15 minutes
- Incremental build: 30 seconds - 2 minutes

## Notes

- **DO NOT** use `cd` commands! Always use relative paths from project root
- The build process uses `make` targets defined in multiple Makefiles
- Build artifacts are placed in architecture-specific directories
