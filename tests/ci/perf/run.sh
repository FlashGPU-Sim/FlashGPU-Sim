#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
CONFIG="${1:-}"
JOB="${2:-}"
BUILD_JOBS="${CI_BUILD_JOBS:-2}"
LOG_ROOT="${REPO_ROOT}/tests/logs/ci/perf"
BUILD_LOG="${LOG_ROOT}/${CONFIG}/${JOB}/build.log"

if [[ -z "${CONFIG}" || -z "${JOB}" ]]; then
  echo "Usage: run.sh CONFIG JOB" >&2
  exit 2
fi

cd "${REPO_ROOT}"
python3 tests/ci/perf/perf.py init \
  --config "${CONFIG}" \
  --job "${JOB}" \
  --log-root "${LOG_ROOT}"
mkdir -p "$(dirname -- "${BUILD_LOG}")"

if [[ "${CI:-}" == "true" || "${GITHUB_ACTIONS:-}" == "true" ]]; then
  (
    set +u
    source setup_environment
    set -u
    make clean
  ) 2>&1 | sed -n '1,10p'
  rm -rf tests/build
  rm -rf lib
fi

# Build FlashGPU-Sim in an isolated sourced environment.
if (
  set +u
  source setup_environment
  set -u
  make FLASH=1 "-j${BUILD_JOBS}"
) 2>&1 | tee "${BUILD_LOG}"; then
  :
else
  status="${PIPESTATUS[0]}"
  echo "FlashGPU-Sim build failed with status ${status}" >&2
  exit "${status}"
fi

if ! find lib -name libcudart.so -print -quit | grep -q .; then
  echo "FlashGPU-Sim build did not produce libcudart.so" >&2
  exit 1
fi

# Performance cases only replay checked-in workloads, so activate the simulator
# once for both tutorial scripts and frozen Triton launchers.
if [[ "${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" != "1" ]]; then
  set +u
  source setup_environment
  set -u
fi

python3 tests/ci/perf/perf.py run \
  --config "${CONFIG}" \
  --job "${JOB}" \
  --log-root "${LOG_ROOT}"
