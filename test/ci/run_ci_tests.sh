#!/bin/bash

# CI test harness for automated testing
# This script runs GPGPU-Sim tests with reduced SM120 and SM90 configurations
# for resource-efficient CI/CD pipelines.

set -euo pipefail

# Get repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CI_LOG_ROOT="$REPO_ROOT/test/logs/ci"
mkdir -p "$CI_LOG_ROOT/logs" "$CI_LOG_ROOT/xml"

run_logged() {
  local label="$1"
  shift
  echo "Running: $label"
  "$@" 2>&1 | tee "$CI_LOG_ROOT/logs/$label.log"
}

run_gtest_group() {
  local config="$1"
  local target="$2"
  local group="$3"
  local label="$4"
  shift 4

  local xml_dir="$CI_LOG_ROOT/xml/$label"
  mkdir -p "$xml_dir"
  export GTEST_OUTPUT="xml:$xml_dir/"
  run_logged "$label" ./test/run_tests.sh -c "$config" run test \
    --target "$target" --group "$group" "$@"
  unset GTEST_OUTPUT
}

run_logged reduced-config-parity-regression \
  python3 test/scripts/test_reduced_config_parity.py
run_logged reduced-config-parity \
  python3 test/scripts/check_reduced_config_parity.py
run_logged ptx-scheduler-operand-regression \
  python3 test/scripts/test_ptx_scheduler_probe_operands.py
run_logged gtest-discovery-output-regression \
  python3 test/scripts/test_gtest_discovery_output.py

# Source environment setup
# Note: In CI Docker, CUDA_INSTALL_PATH is already set via ENV, so we skip setup.sh
# and only source setup_environment for GPGPU-Sim specific paths
echo "Setting up GPGPU-Sim environment..."
if [ -n "${CI:-}" ] || [ -n "${GITHUB_ACTIONS:-}" ]; then
  echo "CI environment detected, skipping setup.sh (CUDA_INSTALL_PATH already set)"
  # setup_environment is a legacy sourced script and is not nounset-safe.
  set +u
  source setup_environment
  set -u

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
  # The legacy setup scripts intentionally probe optional unset variables.
  set +u
  source setup.sh
  source setup_environment
  set -u
fi

# CI_TEST_CONFIG remains a backward-compatible alias for the SM120 config.
SM120_TEST_CONFIG="${CI_SM120_TEST_CONFIG:-${CI_TEST_CONFIG:-SM120_RTX5090_REDUCED}}"
SM90_TEST_CONFIG="${CI_SM90_TEST_CONFIG:-SM90_H100_REDUCED}"

echo "Running SM120 tests with configuration: $SM120_TEST_CONFIG"
echo "Running SM90 tests with configuration: $SM90_TEST_CONFIG"

# In CI, clean and force rebuild to ensure we build with container's glibc/toolchain
if [ -n "${CI:-}" ] || [ -n "${GITHUB_ACTIONS:-}" ]; then
  echo "CI environment detected, cleaning build artifacts..."

  # Check what exists before cleaning
  echo "Before clean:"
  [ -d "lib" ] && echo "  lib/ exists: $(find lib -name 'libcudart.so' 2>/dev/null | wc -l) libcudart.so files" || echo "  lib/ does not exist"
  [ -d "test/build" ] && echo "  test/build/ exists" || echo "  test/build/ does not exist"

  # Clean
  make clean 2>&1 | sed -n '1,10p'  # Show first 10 lines of clean output
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

# Build correctness and smoke groups. Analysis sweeps and microbenchmarks stay
# out of PR CI.
echo "Building tests..."
run_logged build-sm120-unit \
  ./test/run_tests.sh build test --target sm120 --group unit
run_logged build-sm120-integration \
  ./test/run_tests.sh build test --target sm120 --group integration
run_logged build-sm90-instructions \
  ./test/run_tests.sh build test --target sm90 --group instructions
run_logged build-sm90-fa2-smoke \
  ./test/run_tests.sh build test --target sm90 --group fa2-smoke
run_logged build-sm90-fa3-smoke \
  ./test/run_tests.sh build test --target sm90 --group fa3-smoke

# Run tests with specified configuration.
# test/run_tests.sh owns the default exclusion list so CI and local runs stay
# consistent. Microbenchmarks are already excluded because they live in a
# separate microbenchmark binary set.
echo "Running test suite..."

# Directory-valued GTEST_OUTPUT paths give every binary a distinct XML file.
run_gtest_group "$SM120_TEST_CONFIG" sm120 unit sm120-unit
run_gtest_group "$SM120_TEST_CONFIG" sm120 integration sm120-integration "$@"
run_gtest_group "$SM90_TEST_CONFIG" sm90 instructions sm90-instructions
run_gtest_group "$SM90_TEST_CONFIG" sm90 fa2-smoke sm90-fa2-smoke
run_gtest_group "$SM90_TEST_CONFIG" sm90 fa3-smoke sm90-fa3-forward-smoke \
  'Fa3PrefillFp16SmokeTest.*'
run_gtest_group "$SM90_TEST_CONFIG" sm90 fa3-smoke sm90-fa3-fixed-forward \
  'Fa3FwdHdim128Fp16IntegrationTest.FixedForwardCase'
run_gtest_group "$SM90_TEST_CONFIG" sm90 fa3-smoke sm90-fa3-backward-smoke \
  'Fa3PrefillFp16BackwardSmokeTest.*'

# Preserve the existing no-filter CI scope, including GPT-2 trace smoke tests.
if [ "$#" -eq 0 ]; then
  run_logged sm120-gpt2-trace ./test/run_tests.sh -c "$SM120_TEST_CONFIG" \
    run trace --target sm120 --group gpt2
fi

echo "CI tests completed successfully!"
