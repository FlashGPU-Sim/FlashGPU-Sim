#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/tests/build/vector_literal}"
GPGPUSIM_CONFIG="${GPGPUSIM_CONFIG:-gcc-13.3.0/cuda-12080/release}"
SIM_LIB_DIR="${SIM_LIB_DIR:-${ROOT_DIR}/lib/${GPGPUSIM_CONFIG}}"
SIM_BUILD_DIR="${SIM_BUILD_DIR:-${ROOT_DIR}/build/${GPGPUSIM_CONFIG}}"
GPU_CONFIG_DIR="${ROOT_DIR}/configs/${GPU_CONFIG:-SM100_B200_REDUCED}"
CUDA_HOME="${CUDA_INSTALL_PATH:-/usr/local/cuda-12.8}"
CXX="${CXX:-g++}"

mkdir -p "${BUILD_DIR}"
cp "${GPU_CONFIG_DIR}/gpgpusim.config" "${BUILD_DIR}/"
cp "${GPU_CONFIG_DIR}"/config_*.icnt "${BUILD_DIR}/" 2>/dev/null || true

"${CXX}" -std=c++17 \
  -I"${ROOT_DIR}" -I"${ROOT_DIR}/src" \
  -I"${SIM_BUILD_DIR}/cuda-sim" -I"${ROOT_DIR}/libcuda" \
  -I"${CUDA_HOME}/include" \
  "${SCRIPT_DIR}/vector_literal_parser_smoke.cc" \
  -L"${SIM_LIB_DIR}" -Wl,-rpath,"${SIM_LIB_DIR}" -lcudart \
  -o "${BUILD_DIR}/vector_literal_parser_smoke"

run_fixture() {
  local fixture="$1"
  (
    cd "${BUILD_DIR}"
    LD_LIBRARY_PATH="${SIM_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
      ./vector_literal_parser_smoke "${fixture}"
  )
}

run_fixture "${SCRIPT_DIR}/fixtures/vector_literal_valid.ptx"

for fixture in "${SCRIPT_DIR}"/fixtures/vector_literal_invalid_*.ptx; do
  log="${BUILD_DIR}/$(basename "${fixture}").log"
  if run_fixture "${fixture}" >"${log}" 2>&1; then
    echo "invalid fixture unexpectedly parsed: ${fixture}" >&2
    exit 1
  fi
  if ! grep -q "Parse error:" "${log}"; then
    echo "invalid fixture failed without parser diagnostic: ${fixture}" >&2
    exit 1
  fi
done

echo "vector literal parser positive/negative smoke passed"
