#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PTX_FILE="${1:-${ROOT_DIR}/test/src/trace/ptx/tcgen05_phase1_smoke.ptx}"
PTX_FILE="$(readlink -f "${PTX_FILE}")"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/test/build/tcgen05_phase1}"
GPGPUSIM_CONFIG="${GPGPUSIM_CONFIG:-gcc-13.3.0/cuda-12080/release}"
SIM_LIB_DIR="${SIM_LIB_DIR:-${ROOT_DIR}/lib/${GPGPUSIM_CONFIG}}"
SIM_BUILD_DIR="${SIM_BUILD_DIR:-${ROOT_DIR}/build/${GPGPUSIM_CONFIG}}"
GPU_CONFIG="${GPU_CONFIG:-SM100_B200_REDUCED}"
GPU_CONFIG_DIR="${ROOT_DIR}/configs/${GPU_CONFIG}"
CUDA_HOME="${CUDA_INSTALL_PATH:-/home/wzr/cuda}"
CXX="${CXX:-g++}"

if [[ ! -f "${SIM_LIB_DIR}/libcudart.so" ]]; then
  echo "missing ${SIM_LIB_DIR}/libcudart.so; build flashgpu-sim first" >&2
  exit 1
fi

if [[ ! -f "${CUDA_HOME}/include/cuda.h" ]]; then
  echo "missing ${CUDA_HOME}/include/cuda.h; set CUDA_INSTALL_PATH" >&2
  exit 1
fi

if [[ ! -f "${SIM_BUILD_DIR}/cuda-sim/ptx.tab.h" ]]; then
  echo "missing ${SIM_BUILD_DIR}/cuda-sim/ptx.tab.h; build flashgpu-sim first" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"
cp "${GPU_CONFIG_DIR}/gpgpusim.config" "${BUILD_DIR}/"
cp "${GPU_CONFIG_DIR}"/config_*.icnt "${BUILD_DIR}/" 2>/dev/null || true

"${CXX}" -std=c++17 \
  -I"${ROOT_DIR}" \
  -I"${ROOT_DIR}/src" \
  -I"${SIM_BUILD_DIR}/cuda-sim" \
  -I"${ROOT_DIR}/libcuda" \
  -I"${CUDA_HOME}/include" \
  "${ROOT_DIR}/test/src/unit/tcgen05_parser_smoke.cc" \
  -L"${SIM_LIB_DIR}" \
  -Wl,-rpath,"${SIM_LIB_DIR}" \
  -lcudart \
  -o "${BUILD_DIR}/tcgen05_parser_smoke"

(
  cd "${BUILD_DIR}"
  LD_LIBRARY_PATH="${SIM_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
    ./tcgen05_parser_smoke "${PTX_FILE}"
)
