#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUN_DIR="${SCRIPT_DIR}/run"
TRACKING_DIR="${RUN_DIR}/tracking"
LAUNCHER_DIR="${TRACKING_DIR}/launchers"
CONFIG_DIR="${REPO_ROOT}/configs/SM120_RTX5090"
SIMULATION_LOG="${RUN_DIR}/simulation.log"

if [[ ! -d "${CONFIG_DIR}" ]]; then
  echo "Error: GPU configuration not found: ${CONFIG_DIR}" >&2
  exit 1
fi

shopt -s nullglob
MAKEFILES=("${LAUNCHER_DIR}"/*_launch1_Makefile)
shopt -u nullglob

if (( ${#MAKEFILES[@]} != 1 )); then
  echo "Error: captured launcher not found." >&2
  echo "Run ${SCRIPT_DIR}/capture.sh first." >&2
  exit 1
fi

LAUNCHER_MAKEFILE="$(basename -- "${MAKEFILES[0]}")"
LAUNCHER_NAME="${LAUNCHER_MAKEFILE%_Makefile}"
LAUNCHER="${LAUNCHER_DIR}/${LAUNCHER_NAME}"

if [[ ! -x "${LAUNCHER}" ]]; then
  echo "Error: captured launcher not found: ${LAUNCHER}" >&2
  echo "Run ${SCRIPT_DIR}/capture.sh first." >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"

echo "[1/3] Preparing the SM120_RTX5090 configuration"
cp -a "${CONFIG_DIR}/." "${LAUNCHER_DIR}/"

echo "[2/3] Configuring FlashGPU-Sim"
if [[ "${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" != "1" ]]; then
  set +u
  source "${REPO_ROOT}/setup_environment"
  set -u
fi

if [[ "${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" != "1" ]] ||
  [[ -z "${GPGPUSIM_CONFIG:-}" ]]; then
  echo "Error: setup_environment did not configure FlashGPU-Sim." >&2
  exit 1
fi

SIM_LIB_DIR="${REPO_ROOT}/lib/${GPGPUSIM_CONFIG}"
if [[ ! -f "${SIM_LIB_DIR}/libcudart.so" ]]; then
  echo "Error: FlashGPU-Sim has not been built for the current environment." >&2
  echo "Complete the Quick Start build before running this tutorial." >&2
  exit 1
fi

LDD_OUTPUT="$(ldd "${LAUNCHER}")"
CUDART_PATH="$(awk '$1 ~ /^libcudart/ {print $3; exit}' <<<"${LDD_OUTPUT}")"
CUDA_DRIVER_PATH="$(awk '$1 ~ /^libcuda\.so/ {print $3; exit}' <<<"${LDD_OUTPUT}")"

if [[ "${CUDART_PATH}" != "${SIM_LIB_DIR}/"* ]] ||
  { [[ -n "${CUDA_DRIVER_PATH}" ]] &&
    [[ "${CUDA_DRIVER_PATH}" != "${SIM_LIB_DIR}/"* ]]; }; then
  echo "Error: launcher is not using FlashGPU-Sim's CUDA libraries." >&2
  printf '%s\n' "${LDD_OUTPUT}" >&2
  exit 1
fi

echo "[3/3] Simulating Triton FlashAttention with FlashGPU-Sim"
cd "${LAUNCHER_DIR}"
"./${LAUNCHER_NAME}" 2>&1 | tee "${SIMULATION_LOG}"

if ! grep -q "Validation PASSED" "${SIMULATION_LOG}"; then
  echo "Error: standalone launcher validation did not pass." >&2
  exit 1
fi

if ! grep -q "^gpu_tot_sim_cycle" "${SIMULATION_LOG}"; then
  echo "Error: simulator cycle statistics were not found." >&2
  exit 1
fi

echo "Simulation completed successfully."
grep -m1 "^gpu_tot_sim_cycle" "${SIMULATION_LOG}"
echo "Simulation log: ${SIMULATION_LOG}"
