#!/usr/bin/env bash
# Run an optional H200 CLUSTER132 calibration binary under FlashGPU-Sim.
# Default config is SM90_H200_CLUSTER132, 4 OpenMP threads.
# Functional cluster tests do not use this script; they use test/run_tests.sh.
#
# Usage:
#   bash scripts/run_calibration_sim.sh [--config NAME] [--run-dir DIR] \
#       [--loops N] -- <binary> [args...]
#   bash scripts/run_calibration_sim.sh build [dsm_bw|tma_bw|h200_probes]
#
# Examples:
#   bash scripts/run_calibration_sim.sh -- ./dsm_h200.out --suite smoke
#   CALIB_LOOPS=32 bash scripts/run_calibration_sim.sh --loops 32 -- ./dsm_h200.out --suite smoke
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_NAME="${CALIB_CONFIG:-SM90_H200_CLUSTER132}"
RUN_DIR=""
LOOPS="${CALIB_LOOPS:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export FLASHGPU_ALLOW_CC_MISMATCH="${FLASHGPU_ALLOW_CC_MISMATCH:-1}"
export FLASHGPU_SIM_CLOCK_FROM_PROP=1

usage() {
  sed -n '2,14p' "$0" | sed 's/^# \?//'
}

ensure_env() {
  if [[ -z "${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" ]]; then
    set +u
    source "${ROOT}/setup.sh"
    source "${ROOT}/setup_environment"
    set -u
  fi
}

sync_kernels() {
  if [[ ! -d "${ROOT}/calibration/kernels/dsm_bw" ]]; then
    bash "${ROOT}/scripts/sync_calibration_kernels.sh"
  fi
}

cmd_build() {
  ensure_env
  sync_kernels
  local which="${1:-dsm_bw}"
  local kdir="${ROOT}/calibration/kernels/${which}"
  if [[ ! -d "${kdir}" ]]; then
    echo "missing ${kdir}; run bash scripts/sync_calibration_kernels.sh" >&2
    exit 1
  fi
  make -C "${kdir}"
  echo "built ${which} under ${kdir}"
}

setup_run_dir() {
  local dest="$1"
  mkdir -p "${dest}"
  local src="${ROOT}/configs/${CONFIG_NAME}"
  if [[ ! -f "${src}/gpgpusim.config" ]]; then
    echo "missing config ${src}/gpgpusim.config" >&2
    exit 1
  fi
  cp -a "${src}/." "${dest}/"
  if [[ -n "${CALIB_MAX_CYCLES:-}" ]]; then
    printf '\n-gpgpu_max_cycle %s\n' "${CALIB_MAX_CYCLES}" >> "${dest}/gpgpusim.config"
  fi
  if [[ "${CALIB_LONG_TMA_DRAIN:-0}" == "1" ]]; then
    printf '\n-gpgpu_deadlock_detect 0\n' >> "${dest}/gpgpusim.config"
  fi
  # GPGPU-Sim reads gpgpusim.config from the process cwd.
}

# --- parse ---
if [[ $# -ge 1 && "$1" == "build" ]]; then
  shift
  cmd_build "${1:-dsm_bw}"
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_NAME="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --loops)
      LOOPS="$2"
      export CALIB_LOOPS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "unknown flag $1 (use -- before the binary)" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  echo "missing binary. Example: $0 -- ./dsm_h200.out --suite smoke" >&2
  usage
  exit 2
fi

ensure_env
sync_kernels

if [[ -z "${RUN_DIR}" ]]; then
  RUN_DIR="${ROOT}/calibration/runs/${CONFIG_NAME}_$(date +%Y%m%d_%H%M%S)"
fi
setup_run_dir "${RUN_DIR}"

echo "config=${CONFIG_NAME}"
echo "run_dir=${RUN_DIR}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
if [[ -n "${LOOPS}" ]]; then
  echo "CALIB_LOOPS=${LOOPS}"
fi

# Resolve binary to an absolute path so cwd can change.
bin="$1"
shift
if [[ "${bin}" != /* ]]; then
  if [[ -x "${bin}" ]]; then
    bin="$(cd "$(dirname "${bin}")" && pwd)/$(basename "${bin}")"
  elif [[ -x "${ROOT}/calibration/kernels/dsm_bw/${bin}" ]]; then
    bin="${ROOT}/calibration/kernels/dsm_bw/${bin}"
  elif [[ -x "${PWD}/${bin}" ]]; then
    bin="${PWD}/${bin}"
  fi
fi

cd "${RUN_DIR}"
echo "cwd=$(pwd)"
echo "exec: ${bin} $*"
if [[ -n "${CALIB_TIMEOUT_S:-}" ]]; then
  exec timeout "${CALIB_TIMEOUT_S}" "${bin}" "$@"
fi
exec "${bin}" "$@"
