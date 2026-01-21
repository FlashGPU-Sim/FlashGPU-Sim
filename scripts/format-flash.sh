#!/bin/bash

# One-time formatter for Flash module C/C++ files
# Applies LLVM-style formatting using clang-format

set -e

# Get repository root
REPO_ROOT="$(git rev-parse --show-toplevel)"
FLASH_DIR="$REPO_ROOT/src/gpgpu-sim/flash"

# Check clang-format availability
if ! command -v clang-format >/dev/null 2>&1; then
    echo "Error: clang-format not found in PATH"
    exit 1
fi

# Find and format all C/C++ files in Flash directory
echo "Formatting Flash module files..."
find "$FLASH_DIR" -type f \( -name "*.cc" -o -name "*.cpp" -o -name "*.h" -o -name "*.hh" -o -name "*.hpp" \) -print0 | \
    xargs -0 clang-format -i --style=file

echo "Done. Flash module files formatted."
