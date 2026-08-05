#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CONFIG_NAME="SM120_RTX5090"
CONFIG_DIR="${REPO_ROOT}/configs/${CONFIG_NAME}"
RUN_DIR="${SCRIPT_DIR}/run"
APP="${RUN_DIR}/vectorAdd"
LOG_FILE="${RUN_DIR}/simulation.log"

echo "[1/4] Configuring FlashGPU-Sim"

if [[ "${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" != "1" ]]; then
  set +u
  source "${REPO_ROOT}/setup_environment"
  set -u
fi

if [[ -z "${GPGPUSIM_ROOT:-}" ]] ||
  [[ "$(cd -- "${GPGPUSIM_ROOT}" && pwd -P)" != "${REPO_ROOT}" ]]; then
  echo "Error: GPGPUSIM_ROOT does not point to this FlashGPU-Sim checkout." >&2
  echo "Open a new shell and run this script again." >&2
  exit 1
fi

if [[ -z "${GPGPUSIM_CONFIG:-}" ]]; then
  echo "Error: GPGPUSIM_CONFIG is not set by setup_environment." >&2
  exit 1
fi

SIM_LIB_DIR="${REPO_ROOT}/lib/${GPGPUSIM_CONFIG}"
if [[ ! -f "${SIM_LIB_DIR}/libcudart.so" ]]; then
  echo "Error: FlashGPU-Sim has not been built for the current environment." >&2
  echo "Complete the Quick Start build before running this tutorial." >&2
  exit 1
fi

if [[ ! -d "${CONFIG_DIR}" ]]; then
  echo "Error: GPU configuration not found: ${CONFIG_DIR}" >&2
  exit 1
fi

echo "[2/4] Preparing ${CONFIG_NAME} in ${RUN_DIR}"
mkdir -p "${RUN_DIR}"
cp -a "${CONFIG_DIR}/." "${RUN_DIR}/"

echo "[3/4] Building the CUDA vector addition workload"
make --no-print-directory -C "${SCRIPT_DIR}" TARGET="${APP}"

if ! ldd "${APP}" | grep -Fq "${SIM_LIB_DIR}/libcudart"; then
  echo "Error: vectorAdd is not linked to FlashGPU-Sim's CUDA runtime." >&2
  echo "Re-source setup_environment and rebuild the workload." >&2
  exit 1
fi

echo "[4/4] Running vectorAdd with FlashGPU-Sim"
(
  cd "${RUN_DIR}"
  ./vectorAdd 2>&1 | tee "${LOG_FILE}"
)

echo "Simulation log: ${LOG_FILE}"
