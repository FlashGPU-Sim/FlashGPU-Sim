#!/bin/bash

# CI planner driven by independent architecture and test-set selections.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CI_LOG_ROOT="$REPO_ROOT/tests/logs/ci"

CI_ARCH="${CI_ARCH:-all}"
CI_TEST_SET="${CI_TEST_SET:-all}"
CI_LIST_JOBS="${CI_LIST_JOBS:-0}"
CI_BUILD_JOBS="${CI_BUILD_JOBS:-2}"
export GPGPUSIM_BUILD_JOBS="${GPGPUSIM_BUILD_JOBS:-$CI_BUILD_JOBS}"

# A single FA2 translation unit approaches 5 GiB RSS. Keep FA2 serial in the
# 7 GiB CI cgroup; other test sets may use the common build-job limit.
TEST_BUILD_JOBS_DEFAULT="$CI_BUILD_JOBS"
if [ "$CI_TEST_SET" = all ] || [ "$CI_TEST_SET" = fa2 ]; then
  TEST_BUILD_JOBS_DEFAULT=1
fi
export TEST_BUILD_JOBS="${TEST_BUILD_JOBS:-$TEST_BUILD_JOBS_DEFAULT}"

case "$CI_TEST_SET" in
  all|core|fa2|fa3) ;;
  *)
    echo "ERROR: unsupported CI_TEST_SET '$CI_TEST_SET'"
    echo "Expected one of: all, core, fa2, fa3"
    exit 2
    ;;
esac

if [[ ! "$CI_LIST_JOBS" =~ ^[01]$ ]]; then
  echo "ERROR: CI_LIST_JOBS must be 0 or 1"
  exit 2
fi

CORE_TEST_GROUPS=(unit integration barrier tma mma wgmma trace)

mapfile -t MANIFEST_ARCHITECTURES < <(
  make -s --no-print-directory -C tests list-architectures
)

line_array_contains() {
  local expected="$1"
  shift
  local item=""
  for item in "$@"; do
    [ "$item" != "$expected" ] || return 0
  done
  return 1
}

declare -a SELECTED_ARCHITECTURES=()
if [ "$CI_ARCH" = all ]; then
  SELECTED_ARCHITECTURES=("${MANIFEST_ARCHITECTURES[@]}")
elif line_array_contains "$CI_ARCH" "${MANIFEST_ARCHITECTURES[@]}"; then
  SELECTED_ARCHITECTURES=("$CI_ARCH")
else
  echo "ERROR: unsupported CI_ARCH '$CI_ARCH'"
  echo "Available architectures: ${MANIFEST_ARCHITECTURES[*]}"
  exit 2
fi

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

architecture_config() {
  local arch="$1"
  local metadata=""
  local manifest_config=""

  metadata="$(make -s --no-print-directory -C tests \
    print-architecture-metadata ARCH="$arch")"
  manifest_config="${metadata%%|*}"
  case "$arch" in
    sm90)
      printf '%s\n' "${CI_SM90_TEST_CONFIG:-${CI_TEST_CONFIG:-$manifest_config}}"
      ;;
    sm120)
      printf '%s\n' "${CI_SM120_TEST_CONFIG:-${CI_TEST_CONFIG:-$manifest_config}}"
      ;;
    *)
      printf '%s\n' "${CI_TEST_CONFIG:-$manifest_config}"
      ;;
  esac
}

test_group_supported() {
  local arch="$1"
  local expected="$2"
  local test_group=""

  while IFS= read -r test_group; do
    [ "$test_group" != "$expected" ] || return 0
  done < <(make -s --no-print-directory -C tests list-test-groups ARCH="$arch")
  return 1
}

plan_ci_jobs() {
  local arch=""
  local test_group=""

  for arch in "${SELECTED_ARCHITECTURES[@]}"; do
    if [ "$CI_TEST_SET" = all ] || [ "$CI_TEST_SET" = core ]; then
      for test_group in "${CORE_TEST_GROUPS[@]}"; do
        if test_group_supported "$arch" "$test_group"; then
          printf '%s|core|%s\n' "$arch" "$test_group"
        fi
      done
    fi

    if { [ "$CI_TEST_SET" = all ] || [ "$CI_TEST_SET" = fa2 ]; } && \
       test_group_supported "$arch" fa2; then
      printf '%s|fa2|fa2\n' "$arch"
    fi

    if { [ "$CI_TEST_SET" = all ] || [ "$CI_TEST_SET" = fa3 ]; } && \
       test_group_supported "$arch" fa3; then
      printf '%s|fa3|fa3\n' "$arch"
    fi
  done
}

PLANNED_JOBS="$(plan_ci_jobs)"
if [ -z "$PLANNED_JOBS" ]; then
  echo "ERROR: test set '$CI_TEST_SET' has no jobs for architecture selector '$CI_ARCH'"
  exit 2
fi

if [ "$CI_LIST_JOBS" -eq 1 ]; then
  printf '%s\n' "$PLANNED_JOBS"
  exit 0
fi

mkdir -p "$CI_LOG_ROOT/logs" "$CI_LOG_ROOT/xml"
trap log_exit_resources EXIT

echo "CI architecture selector: $CI_ARCH"
echo "CI test set: $CI_TEST_SET"
echo "Resolved architectures: ${SELECTED_ARCHITECTURES[*]}"
echo "Build jobs: simulator=$GPGPUSIM_BUILD_JOBS tests=$TEST_BUILD_JOBS"
echo "Planned jobs:"
while IFS= read -r planned_job; do
  printf '  %s\n' "$planned_job"
done <<< "$PLANNED_JOBS"
log_runner_resources start | tee "$CI_LOG_ROOT/logs/runner-resources-start.log"

run_gtest_selection() {
  local config="$1"
  local arch="$2"
  local test_group="$3"
  local label="$4"
  shift 4

  local xml_dir="$CI_LOG_ROOT/xml/$label"
  mkdir -p "$xml_dir"
  export GTEST_OUTPUT="xml:$xml_dir/"
  run_logged "$label" ./tests/run_tests.py -c "$config" run \
    --arch "$arch" --group "$test_group" "$@"
  unset GTEST_OUTPUT
}

build_selection() {
  local arch="$1"
  local test_group="$2"
  local label="$3"
  shift 3
  run_logged "$label" ./tests/run_tests.py build \
    --arch "$arch" --group "$test_group" "$@"
}

# Run repository-level planner and discovery regressions once per normal matrix.
if { [ "$CI_ARCH" = all ] || [ "$CI_ARCH" = sm120 ]; } && \
   { [ "$CI_TEST_SET" = all ] || [ "$CI_TEST_SET" = core ]; }; then
  run_logged architecture-manifest-regression \
    python3 tests/ci/test_arch_manifest.py
  run_logged ci-planner-regression \
    python3 tests/ci/test_ci_planner.py
  run_logged gtest-regression \
    python3 tests/ci/test_gtest.py
fi

echo "Setting up FlashGPU-Sim environment..."
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
  echo "ERROR: setup_environment did not configure the simulator"
  exit 1
fi
echo "✓ Simulator environment configured:"
echo "  GPGPUSIM_ROOT=$GPGPUSIM_ROOT"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

for arch in "${SELECTED_ARCHITECTURES[@]}"; do
  echo "$arch configuration: $(architecture_config "$arch")"
done

# CI containers always rebuild with their own glibc and CUDA toolchain.
if [ -n "${CI:-}" ] || [ -n "${GITHUB_ACTIONS:-}" ]; then
  echo "CI environment detected, cleaning build artifacts..."
  make clean 2>&1 | sed -n '1,10p'
  rm -rf tests/build
  rm -rf lib

  run_logged build-flashgpu-sim make FLASH=1 "-j$GPGPUSIM_BUILD_JOBS"
  if ! find lib -name 'libcudart.so' 2>/dev/null | grep -q .; then
    echo "ERROR: libcudart.so not found after build"
    exit 1
  fi
fi

run_core_test_group() {
  local arch="$1"
  local test_group="$2"
  local config=""
  local label="$arch-$test_group"
  config="$(architecture_config "$arch")"

  if [ "$test_group" = trace ]; then
    run_logged "$label" ./tests/run_tests.py -c "$config" run \
      --arch "$arch" --group trace --profile gpt2
    return
  fi

  build_selection "$arch" "$test_group" "build-$label"
  case "$test_group" in
    barrier)
      run_gtest_selection "$config" "$arch" "$test_group" "$label" \
        --gtest-filter '*-MBarrierSanityTest.*'
      ;;
    tma)
      run_gtest_selection "$config" "$arch" "$test_group" "$label" \
        --gtest-filter '*-*CPAsyncMethod*:*PerformanceComparison*'
      ;;
    *)
      run_gtest_selection "$config" "$arch" "$test_group" "$label"
      ;;
  esac

  if [ "$arch" = sm90 ] && [ "$test_group" = integration ]; then
    run_logged sm90-cp-async-src-size-stats \
      python3 tests/validators/cp_async_src_size_stats.py \
      "$CI_LOG_ROOT/logs/$label.log"
  fi
}

run_fa2_test_set() {
  local arch="$1"
  local config=""
  config="$(architecture_config "$arch")"
  build_selection "$arch" fa2 "build-$arch-fa2-smoke" --profile smoke
  run_gtest_selection "$config" "$arch" fa2 "$arch-fa2-smoke" \
    --profile smoke
}

run_fa3_test_set() {
  local arch="$1"
  local config=""
  config="$(architecture_config "$arch")"
  build_selection "$arch" fa3 "build-$arch-fa3-smoke" --profile smoke
  build_selection "$arch" fa3 "build-$arch-fa3-packgqa" --profile packgqa

  run_gtest_selection "$config" "$arch" fa3 "$arch-fa3-forward-smoke" \
    --profile smoke --gtest-filter 'Fa3PrefillFp16SmokeTest.*'
  run_gtest_selection "$config" "$arch" fa3 "$arch-fa3-fixed-forward" \
    --profile smoke \
    --gtest-filter 'Fa3FwdHdim128Fp16IntegrationTest.FixedForwardCase'
  run_gtest_selection "$config" "$arch" fa3 "$arch-fa3-backward-smoke" \
    --profile smoke --gtest-filter 'Fa3PrefillFp16BackwardSmokeTest.*'
  run_gtest_selection "$config" "$arch" fa3 "$arch-fa3-packgqa" \
    --profile packgqa \
    --gtest-filter 'Fa3FwdPackGqaFp16IntegrationTest.Smoke'
}

while IFS='|' read -r arch planned_test_set test_group; do
  case "$planned_test_set" in
    core) run_core_test_group "$arch" "$test_group" ;;
    fa2) run_fa2_test_set "$arch" ;;
    fa3) run_fa3_test_set "$arch" ;;
    *)
      echo "ERROR: internal planner emitted unknown test set '$planned_test_set'"
      exit 2
      ;;
  esac
done <<< "$PLANNED_JOBS"

echo "CI tests completed successfully!"
