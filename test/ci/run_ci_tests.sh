#!/bin/bash

# CI test harness for automated testing
# This script runs GPGPU-Sim tests with the reduced configuration
# for resource-efficient CI/CD pipelines.

set -e

# Get repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Source environment setup
# Note: In CI Docker, CUDA_INSTALL_PATH is already set via ENV, so we skip setup.sh
# and only source setup_environment for GPGPU-Sim specific paths
echo "Setting up GPGPU-Sim environment..."
if [ -n "$CI" ] || [ -n "$GITHUB_ACTIONS" ]; then
  echo "CI environment detected, skipping setup.sh (CUDA_INSTALL_PATH already set)"
  source setup_environment

  # Verify simulator environment is set up correctly
  if [ -z "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" ]; then
    echo "ERROR: setup_environment did not set GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN"
    exit 1
  fi
  echo "✓ Simulator environment configured:"
  echo "  GPGPUSIM_ROOT=$GPGPUSIM_ROOT"
  echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
else
  echo "Local environment detected, sourcing both setup.sh and setup_environment"
  source setup.sh
  source setup_environment
fi

# Use CI_TEST_CONFIG environment variable if set, otherwise default to reduced config
TEST_CONFIG="${CI_TEST_CONFIG:-SM120_RTX5090_REDUCED}"

echo "Running tests with configuration: $TEST_CONFIG"

# In CI, clean and force rebuild to ensure we build with container's glibc/toolchain
if [ -n "$CI" ] || [ -n "$GITHUB_ACTIONS" ]; then
  echo "CI environment detected, cleaning build artifacts..."

  # Check what exists before cleaning
  echo "Before clean:"
  [ -d "lib" ] && echo "  lib/ exists: $(find lib -name 'libcudart.so' 2>/dev/null | wc -l) libcudart.so files" || echo "  lib/ does not exist"
  [ -d "test/build" ] && echo "  test/build/ exists" || echo "  test/build/ does not exist"

  # Clean
  make clean 2>&1 | head -10  # Show first 10 lines of clean output
  rm -rf test/build
  rm -rf lib

  echo "After clean:"
  [ -d "lib" ] && echo "  WARNING: lib/ still exists!" || echo "  ✓ lib/ removed"
  [ -d "test/build" ] && echo "  WARNING: test/build/ still exists!" || echo "  ✓ test/build/ removed"

  echo "Forcing full rebuild from scratch..."

  # Force rebuild of GPGPU-Sim library
  echo "Building GPGPU-Sim library..."
  make FLASH=1 -j

  echo "After build:"
  if [ -f "lib/gcc-$(gcc -dumpversion)/cuda-$(nvcc --version | grep -oP 'release \K[0-9.]+')/release/libcudart.so" ] || find lib -name 'libcudart.so' 2>/dev/null | grep -q .; then
    echo "  ✓ libcudart.so built successfully: $(find lib -name 'libcudart.so' 2>/dev/null)"
  else
    echo "  ERROR: libcudart.so not found after build!"
    exit 1
  fi
fi

# Build verification tests only (microbenchmarks not needed in CI)
echo "Building tests..."
./test/run_tests.sh build test

# Run tests with specified configuration.
# test/run_tests.sh owns the default exclusion list so CI and local runs stay
# consistent. Microbenchmarks are already excluded because they live in a
# separate bench binary set.
echo "Running test suite..."

# Use gtest XML output for structured results (absolute path since tests cd into run dir)
export GTEST_OUTPUT="xml:$REPO_ROOT/test_results.xml"

./test/run_tests.sh -c "$TEST_CONFIG" test "$@"

echo "CI tests completed successfully!"
