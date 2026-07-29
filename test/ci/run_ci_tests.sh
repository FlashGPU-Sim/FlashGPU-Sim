#!/bin/bash

# CI test harness for automated testing
# This script runs GPGPU-Sim tests with full SM120 and SM90 configurations
# by default; CI_* environment variables can override them when needed.

set -euo pipefail

# Get repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CI_LOG_ROOT="$REPO_ROOT/test/logs/ci"
mkdir -p "$CI_LOG_ROOT/logs" "$CI_LOG_ROOT/xml"

CI_SHARD="${CI_SHARD:-all}"
CI_BUILD_JOBS="${CI_BUILD_JOBS:-2}"
export GPGPUSIM_BUILD_JOBS="${GPGPUSIM_BUILD_JOBS:-$CI_BUILD_JOBS}"

# A single FA2 NVCC translation unit approaches 5 GiB RSS. Keep FA2 builds
# serial inside the 7 GiB CI cgroup; other Hopper groups may use the common
# build-job limit (FA3 also has an object-level serialization chain).
HOPPER_BUILD_JOBS_DEFAULT="$CI_BUILD_JOBS"
if [ "$CI_SHARD" = all ] || [ "$CI_SHARD" = sm90-fa2 ]; then
  HOPPER_BUILD_JOBS_DEFAULT=1
fi
export HOPPER_BUILD_JOBS="${HOPPER_BUILD_JOBS:-$HOPPER_BUILD_JOBS_DEFAULT}"

case "$CI_SHARD" in
  all|sm120|sm90-fa2|sm90-fa3) ;;
  *)
    echo "ERROR: unsupported CI_SHARD '$CI_SHARD'"
    echo "Expected one of: all, sm120, sm90-fa2, sm90-fa3"
    exit 2
    ;;
esac

run_logged() {
  local label="$1"
  shift
  echo "Running: $label"
  "$@" 2>&1 | tee "$CI_LOG_ROOT/logs/$label.log"
}

log_runner_resources() {
  local phase="$1"
  local metric
  local line

  echo "Runner resource snapshot: $phase"
  echo "  CPUs: $(nproc)"
  if command -v free >/dev/null; then
    free -h
  fi
  df -h "$REPO_ROOT"

  for metric in memory.max memory.current memory.peak memory.events; do
    if [ -r "/sys/fs/cgroup/$metric" ]; then
      while IFS= read -r line; do
        echo "  cgroup $metric: $line"
      done < "/sys/fs/cgroup/$metric"
    fi
  done

  for metric in memory.limit_in_bytes memory.usage_in_bytes \
                memory.max_usage_in_bytes memory.failcnt; do
    if [ -r "/sys/fs/cgroup/memory/$metric" ]; then
      while IFS= read -r line; do
        echo "  cgroup v1 $metric: $line"
      done < "/sys/fs/cgroup/memory/$metric"
    fi
  done
}

log_exit_resources() {
  local status=$?
  trap - EXIT
  set +e
  log_runner_resources exit | tee "$CI_LOG_ROOT/logs/runner-resources-exit.log"
  exit "$status"
}

trap log_exit_resources EXIT

echo "CI shard: $CI_SHARD"
echo "Build jobs: simulator=$GPGPUSIM_BUILD_JOBS hopper=$HOPPER_BUILD_JOBS"
log_runner_resources start | tee "$CI_LOG_ROOT/logs/runner-resources-start.log"

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

run_logged ptx-scheduler-operand-regression \
  python3 test/scripts/test_ptx_scheduler_probe_operands.py
run_logged gtest-discovery-output-regression \
  python3 test/scripts/test_gtest_discovery_output.py

# Source the simulator environment. CI images set CUDA_INSTALL_PATH through
# Docker ENV; local callers must export it before invoking this script.
echo "Setting up GPGPU-Sim environment..."
if [ -z "${CUDA_INSTALL_PATH:-}" ]; then
  echo "ERROR: set CUDA_INSTALL_PATH to the CUDA Toolkit root before running CI tests"
  exit 1
fi

if [ -n "${CI:-}" ] || [ -n "${GITHUB_ACTIONS:-}" ]; then
  echo "CI environment detected"
else
  echo "Local environment detected"
fi

# setup_environment is a legacy sourced script and is not nounset-safe.
set +u
source setup_environment
set -u

if [ -z "$GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN" ]; then
  echo "ERROR: setup_environment did not set GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN"
  exit 1
fi
echo "✓ Simulator environment configured:"
echo "  GPGPUSIM_ROOT=$GPGPUSIM_ROOT"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# CI_TEST_CONFIG remains a backward-compatible alias for the SM120 config.
SM120_TEST_CONFIG="${CI_SM120_TEST_CONFIG:-${CI_TEST_CONFIG:-SM120_RTX5090}}"
SM90_TEST_CONFIG="${CI_SM90_TEST_CONFIG:-SM90_H100}"

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
  run_logged build-gpgpusim make FLASH=1 "-j$GPGPUSIM_BUILD_JOBS"

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
if [ "$CI_SHARD" = all ] || [ "$CI_SHARD" = sm120 ]; then
  run_logged build-sm120-unit \
    ./test/run_tests.sh build test --target sm120 --group unit
  run_logged build-sm120-integration \
    ./test/run_tests.sh build test --target sm120 --group integration
fi
if [ "$CI_SHARD" = all ] || [ "$CI_SHARD" = sm90-fa2 ]; then
  run_logged build-sm90-instructions \
    ./test/run_tests.sh build test --target sm90 --group instructions
  run_logged build-sm90-fa2-smoke \
    ./test/run_tests.sh build test --target sm90 --group fa2-smoke
fi
if [ "$CI_SHARD" = all ] || [ "$CI_SHARD" = sm90-fa3 ]; then
  run_logged build-sm90-fa3-smoke \
    ./test/run_tests.sh build test --target sm90 --group fa3-smoke
fi

# Run tests with specified configuration.
# test/run_tests.sh owns the default exclusion list so CI and local runs stay
# consistent. Microbenchmarks are already excluded because they live in a
# separate microbenchmark binary set.
echo "Running test suite..."

# Directory-valued GTEST_OUTPUT paths give every binary a distinct XML file.
if [ "$CI_SHARD" = all ] || [ "$CI_SHARD" = sm120 ]; then
  run_gtest_group "$SM120_TEST_CONFIG" sm120 unit sm120-unit
  run_gtest_group "$SM120_TEST_CONFIG" sm120 integration sm120-integration "$@"
fi
if [ "$CI_SHARD" = all ] || [ "$CI_SHARD" = sm90-fa2 ]; then
  run_gtest_group "$SM90_TEST_CONFIG" sm90 instructions sm90-instructions
  run_gtest_group "$SM90_TEST_CONFIG" sm90 fa2-smoke sm90-fa2-smoke
fi
if [ "$CI_SHARD" = all ] || [ "$CI_SHARD" = sm90-fa3 ]; then
  run_gtest_group "$SM90_TEST_CONFIG" sm90 fa3-smoke sm90-fa3-forward-smoke \
    'Fa3PrefillFp16SmokeTest.*'
  run_gtest_group "$SM90_TEST_CONFIG" sm90 fa3-smoke sm90-fa3-fixed-forward \
    'Fa3FwdHdim128Fp16IntegrationTest.FixedForwardCase'
  run_gtest_group "$SM90_TEST_CONFIG" sm90 fa3-smoke sm90-fa3-backward-smoke \
    'Fa3PrefillFp16BackwardSmokeTest.*'
fi

# Preserve the existing no-filter CI scope, including GPT-2 trace smoke tests.
if { [ "$CI_SHARD" = all ] || [ "$CI_SHARD" = sm120 ]; } && \
   [ "$#" -eq 0 ]; then
  run_logged sm120-gpt2-trace ./test/run_tests.sh -c "$SM120_TEST_CONFIG" \
    run trace --target sm120 --group gpt2
fi

echo "CI tests completed successfully!"
