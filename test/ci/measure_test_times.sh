#!/bin/bash

# Measure individual test execution times in simulator mode
# Usage: ./measure_test_times.sh [timeout_per_test_seconds]

set -e

# Get repository root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${CUDA_INSTALL_PATH:-}" ]; then
  echo "ERROR: set CUDA_INSTALL_PATH to the CUDA Toolkit root before measuring tests"
  exit 1
fi
source setup_environment

# Default timeout per test (10 minutes)
TIMEOUT_PER_TEST="${1:-600}"

echo "Measuring test execution times in SIMULATOR mode"
echo "Timeout per test: ${TIMEOUT_PER_TEST}s"
echo "========================================================"
echo ""

# Use the same full SM120 configuration as CI by default
TEST_CONFIG="${CI_TEST_CONFIG:-SM120_RTX5090}"

# Array of test patterns to run (focusing on major test suites)
TEST_SUITES=(
  "GPGPUSimBasicTest.*"
  "MathTest.*"
  "LdMatrixX1Test.*"
  "LdMatrixX2Test.*"
  "LdMatrixX4Test.*"
  "StMatrixX1Test.*"
  "StMatrixX2Test.*"
  "StMatrixX4Test.*"
  "LdMatrixM8N8TransTest.*"
  "StMatrixM8N8TransTest.*"
  "CudaVectorAddTest.*"
  "CudaTensorMMATest.*"
  "MMATF32M16N8K4IntegrationTest.*"
  "MMATF32M16N8K8IntegrationTest.*"
  "MMAF16M16N8K8IntegrationTest.*"
  "MMAF16M8N8K4IntegrationTest.*"
  "MMAF16M16N8K16Test.*"
  "MMAS8M16N8K16IntegrationTest.*"
  "MMAS8M16N8K32IntegrationTest.*"
  "MMAS8M8N8K16IntegrationTest.*"
  "MMABF16M16N8K8IntegrationTest.*"
  "MMABF16M16N8K16Test.*"
  "MBarrierThreadLevelTest.*"
  "MBarrierSanityTest.*"
  "GPGPUSimIntegrationTest.*"
  "CudaTMATest.*"
  "BasicValues/ParameterizedTest.*"
  "VariousSizes/CudaVectorAddParameterizedTest.*"
)

# Results file
RESULTS_FILE="test_timing_results.txt"
echo "Test Suite,Time (seconds),Status" > "$RESULTS_FILE"

# Run each test suite and measure time
for test_pattern in "${TEST_SUITES[@]}"; do
  echo "Running: $test_pattern"

  start_time=$(date +%s)
  group="unit"

  # The current runner requires an explicit registry group. CUDA integration
  # suites and the standalone MMA integration cases live in the integration
  # binary; the remaining instruction/unit suites live in the unit binary.
  case "$test_pattern" in
    Cuda*|GPGPUSimIntegrationTest.*|MMA*|MBarrier*|LdMatrix*|StMatrix*|VariousSizes/*)
      group="integration"
      ;;
  esac

  # Run test in a subshell with proper environment and capture result
  # Skip CudaTMATest.PerformanceComparison for CudaTMATest suite
  if [[ "$test_pattern" == "CudaTMATest.*" ]]; then
    filter="CudaTMATest.*:-CudaTMATest.PerformanceComparison"
  else
    filter="$test_pattern"
  fi

  if timeout "$TIMEOUT_PER_TEST" ./test/run_tests.sh -c "$TEST_CONFIG" \
      run test --target sm120 --group "$group" "$filter" &>/dev/null; then
    status="PASS"
  else
    exit_code=$?
    if [ $exit_code -eq 124 ]; then
      status="TIMEOUT (${TIMEOUT_PER_TEST}s)"
    else
      status="FAIL"
    fi
  fi

  end_time=$(date +%s)
  duration=$((end_time - start_time))

  echo "  → ${duration}s [$status]"
  echo "$test_pattern,$duration,$status" >> "$RESULTS_FILE"
done

echo ""
echo "========================================================"
echo "Test timing results saved to: $RESULTS_FILE"
echo ""
echo "Summary (sorted by time, top 20):"
sort -t',' -k2 -rn "$RESULTS_FILE" | head -20
