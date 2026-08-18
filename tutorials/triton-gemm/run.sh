#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUN_DIR="${SCRIPT_DIR}/run"
TRACKING_DIR="${RUN_DIR}/tracking"
LAUNCHER_DIR="${TRACKING_DIR}/launchers"
BUNDLED_TRACE_DIR="${SCRIPT_DIR}/trace"
CONFIG_NAME="${PERF_SIM_CONFIG:-SM120_RTX5090}"
CONFIG_DIR="${REPO_ROOT}/configs/${CONFIG_NAME}"
SIMULATION_LOG="${RUN_DIR}/simulation.log"
LAUNCHER_NAME="kernel_tma_gemm_launch1"
LAUNCHER_MAKEFILE="${LAUNCHER_NAME}_Makefile"
LAUNCHER="${LAUNCHER_DIR}/${LAUNCHER_NAME}"

if [[ ! -d "${CONFIG_DIR}" ]]; then
  echo "Error: GPU configuration not found: ${CONFIG_DIR}" >&2
  exit 1
fi

echo "[1/4] Preparing the Triton GEMM replay"
if [[ ! -f "${LAUNCHER_DIR}/${LAUNCHER_MAKEFILE}" ]]; then
  if [[ ! -f "${BUNDLED_TRACE_DIR}/launchers/${LAUNCHER_MAKEFILE}" ]]; then
    echo "Error: bundled GEMM replay is incomplete: ${BUNDLED_TRACE_DIR}" >&2
    exit 1
  fi
  mkdir -p "${TRACKING_DIR}/data" "${LAUNCHER_DIR}"
  cp -a "${BUNDLED_TRACE_DIR}/data/." "${TRACKING_DIR}/data/"
  cp -a "${BUNDLED_TRACE_DIR}/launchers/." "${LAUNCHER_DIR}/"
  echo "  Staged the checked-in online capture."
else
  echo "  Using the existing capture under ${TRACKING_DIR}."
fi

mkdir -p "${RUN_DIR}"

echo "[2/4] Configuring FlashGPU-Sim for ${CONFIG_NAME}"
cp -a "${CONFIG_DIR}/." "${LAUNCHER_DIR}/"

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

echo "[3/4] Building the standalone launcher"
make --no-print-directory -C "${LAUNCHER_DIR}" -f "${LAUNCHER_MAKEFILE}"

if [[ ! -x "${LAUNCHER}" ]]; then
  echo "Error: launcher build did not produce ${LAUNCHER}." >&2
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

echo "[4/4] Replaying Triton GEMM with FlashGPU-Sim"
(
  cd "${LAUNCHER_DIR}"
  "./${LAUNCHER_NAME}" 2>&1 | tee "${SIMULATION_LOG}"
)

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
