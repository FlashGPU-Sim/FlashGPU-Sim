#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
CONFIG="${1:-}"
BUILD_JOBS="${CI_BUILD_JOBS:-2}"
LOG_ROOT="${REPO_ROOT}/tests/logs/ci/perf"
BUILD_LOG="${LOG_ROOT}/${CONFIG}/build.log"

if [[ -z "${CONFIG}" ]]; then
  echo "Usage: run.sh CONFIG" >&2
  exit 2
fi

cd "${REPO_ROOT}"
python3 tests/ci/perf/perf.py init --config "${CONFIG}" --log-root "${LOG_ROOT}"
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

# Keep setup_environment inside the build subprocess. The parent must remain a
# native CUDA environment so an offline capture script can run before its
# corresponding simulator replay script.
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

python3 tests/ci/perf/perf.py run --config "${CONFIG}" --log-root "${LOG_ROOT}"
