#!/bin/bash

# Execute one TOML-defined CI job, or every job for local validation.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CI_LOG_ROOT="$REPO_ROOT/tests/logs/ci"
CI_PLANNER="$REPO_ROOT/tests/ci/planner.py"
CI_FAILURE_CONTEXT="$CI_LOG_ROOT/failure-context.txt"

CI_JOB="${CI_JOB:-all}"
CI_LIST_JOBS="${CI_LIST_JOBS:-0}"
CI_BUILD_JOBS="${CI_BUILD_JOBS:-2}"
export GPGPUSIM_BUILD_JOBS="${GPGPUSIM_BUILD_JOBS:-$CI_BUILD_JOBS}"

if [[ ! "$CI_LIST_JOBS" =~ ^[01]$ ]]; then
  echo "ERROR: CI_LIST_JOBS must be 0 or 1"
  exit 2
fi

PLANNED_TESTS="$(python3 "$CI_PLANNER" plan --job "$CI_JOB")"

if [ "$CI_LIST_JOBS" -eq 1 ]; then
  printf '%s\n' "$PLANNED_TESTS"
  exit 0
fi

mapfile -t SELECTED_ARCHITECTURES < <(
  python3 "$CI_PLANNER" architectures --job "$CI_JOB"
)
mapfile -t PRE_CHECKS < <(
  python3 "$CI_PLANNER" checks --job "$CI_JOB"
)

# A single FA2 translation unit approaches 5 GiB RSS. Keep any job containing
# FA2 serial in the 7 GiB CI cgroup; other jobs use the common build limit.
TEST_BUILD_JOBS_DEFAULT="$CI_BUILD_JOBS"
while IFS='|' read -r planned_job planned_arch planned_name test_group remainder; do
  if [ "$test_group" = fa2 ]; then
    TEST_BUILD_JOBS_DEFAULT=1
    break
  fi
done <<< "$PLANNED_TESTS"
export TEST_BUILD_JOBS="${TEST_BUILD_JOBS:-$TEST_BUILD_JOBS_DEFAULT}"

mkdir -p "$CI_LOG_ROOT/logs" "$CI_LOG_ROOT/xml"
rm -f "$CI_FAILURE_CONTEXT"

run_logged() {
  local label="$1"
  local failure_group="$2"
  local status
  local -a pipeline_status
  shift 2
  echo "Running: $label"
  if "$@" 2>&1 | tee "$CI_LOG_ROOT/logs/$label.log"; then
    return 0
  else
    pipeline_status=("${PIPESTATUS[@]}")
    status="${pipeline_status[0]}"
    if [ "$status" -eq 0 ]; then
      status="${pipeline_status[1]}"
    fi
    {
      printf 'phase=%s\nexit_code=%s\n' "$label" "$status"
      if [ -n "$failure_group" ]; then
        printf 'group=%s\n' "$failure_group"
      fi
    } > "$CI_FAILURE_CONTEXT"
    return "$status"
  fi
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

trap log_exit_resources EXIT

echo "CI job selector: $CI_JOB"
echo "Resolved architectures: ${SELECTED_ARCHITECTURES[*]}"
echo "Build jobs: simulator=$GPGPUSIM_BUILD_JOBS tests=$TEST_BUILD_JOBS"
echo "Planned tests:"
while IFS= read -r planned_test; do
  printf '  %s\n' "$planned_test"
done <<< "$PLANNED_TESTS"
log_runner_resources start | tee "$CI_LOG_ROOT/logs/runner-resources-start.log"

for check_script in "${PRE_CHECKS[@]}"; do
  check_name="$(basename "$check_script" .py)"
  run_logged "${check_name//_/-}" "" python3 "$check_script"
done

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

  run_logged build-flashgpu-sim "" make FLASH=1 "-j$GPGPUSIM_BUILD_JOBS"
  if ! find lib -name 'libcudart.so' 2>/dev/null | grep -q .; then
    echo "ERROR: libcudart.so not found after build"
    exit 1
  fi
fi

run_ci_test() {
  local job_name="$1"
  local arch="$2"
  local test_name="$3"
  local test_group="$4"
  local profile="$5"
  local mode="$6"
  local gtest_filter="$7"
  local post_check="$8"
  local config=""
  local label="$job_name-$test_name"
  local post_name=""
  local -a selectors=(--arch "$arch" --group "$test_group")

  config="$(architecture_config "$arch")"
  if [ -n "$profile" ]; then
    selectors+=(--profile "$profile")
  fi
  if [ -n "$mode" ]; then
    selectors+=(--mode "$mode")
  fi
  if [ -n "$gtest_filter" ]; then
    selectors+=(--gtest-filter "$gtest_filter")
  fi

  mkdir -p "$CI_LOG_ROOT/xml/$label"
  export GTEST_OUTPUT="xml:$CI_LOG_ROOT/xml/$label/"
  run_logged "$label" "$test_group" \
    ./tests/run_tests.py -c "$config" run "${selectors[@]}"
  unset GTEST_OUTPUT

  if [ -n "$post_check" ]; then
    post_name="$(basename "$post_check" .py)"
    run_logged "$label-${post_name//_/-}" "$test_group" \
      python3 "$post_check" "$CI_LOG_ROOT/logs/$label.log"
  fi
}

while IFS='|' read -r job_name arch test_name test_group profile mode \
  gtest_filter post_check; do
  run_ci_test "$job_name" "$arch" "$test_name" "$test_group" "$profile" \
    "$mode" "$gtest_filter" "$post_check"
done <<< "$PLANNED_TESTS"

echo "CI tests completed successfully!"
