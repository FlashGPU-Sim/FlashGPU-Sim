# Code Formatting Workflow

This document describes the code formatting standards and tooling for the Flash module.

## Overview

The Flash module (`src/gpgpu-sim/flash/`) uses LLVM-style formatting enforced via:
- A local `.clang-format` file in the Flash directory
- Pre-commit hook validation for staged files
- One-time formatting script for initial/bulk formatting

## Requirements

- `clang-format` must be installed and available in PATH
- Version 10+ recommended (supports `--dry-run` and `--Werror` flags)

## Pre-commit Hook

The pre-commit hook is automatically installed when you run `source setup_environment` (required for building). It checks staged Flash C/C++ files:

1. Verifies `clang-format` is available
2. Identifies staged files in `src/gpgpu-sim/flash/` with extensions: `.cc`, `.cpp`, `.h`, `.hh`, `.hpp`
3. Runs `clang-format --dry-run --Werror` to check formatting
4. Fails commit if any file is incorrectly formatted

### Bypassing the Hook

For milestone commits or emergencies, use `--no-verify`:
```bash
git commit --no-verify -m "message"
```

## Formatting Scripts

### One-time Formatter

Format all Flash module files:
```bash
./scripts/format-flash.sh
```

This script:
- Finds all C/C++ files in `src/gpgpu-sim/flash/`
- Applies LLVM-style formatting in-place
- Reports success/failure

### Manual Formatting

Format specific files:
```bash
clang-format -i --style=file src/gpgpu-sim/flash/myfile.cc
```

## Style Configuration

The Flash module uses LLVM style with:
- **IndentWidth**: 2 spaces
- **ColumnLimit**: 80 characters

See `src/gpgpu-sim/flash/.clang-format` for the full configuration.

## Workflow

1. Make changes to Flash files
2. Stage changes: `git add src/gpgpu-sim/flash/...`
3. Commit: `git commit`
4. If formatting check fails:
   - Run `./scripts/format-flash.sh`
   - Re-stage: `git add -u`
   - Commit again
