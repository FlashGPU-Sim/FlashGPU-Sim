#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
usage() {
  cat <<EOF
usage: $(basename "$0") [--check-inline] [ptx-file]

Validates TCGen05 parser coverage on a checked-in PTX file. With
--check-inline, also compiles
tests/dev/tcgen05/fixtures/tcgen05_instruction_surface_inline.cu
with a Blackwell-capable CUDA toolchain and validates the generated PTX.

Environment:
  TCGEN05_INLINE_NVCC  explicit nvcc for --check-inline
  FA4_CU13_ROOT        CUDA 13 root containing bin/nvcc
  FA4_PYTHON           Python env containing nvidia-cu13
EOF
}

CHECK_INLINE=0
PTX_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-inline)
      CHECK_INLINE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "${PTX_FILE}" ]]; then
        echo "only one ptx-file argument is supported" >&2
        exit 2
      fi
      PTX_FILE="$1"
      shift
      ;;
  esac
done
if [[ $# -gt 0 ]]; then
  if [[ -n "${PTX_FILE}" || $# -gt 1 ]]; then
    echo "only one ptx-file argument is supported" >&2
    exit 2
  fi
  PTX_FILE="$1"
fi

PTX_FILE="${PTX_FILE:-${ROOT_DIR}/tests/dev/tcgen05/fixtures/tcgen05_phase1_smoke.ptx}"
PTX_FILE="$(readlink -f "${PTX_FILE}")"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/tests/build/tcgen05_phase1}"
GPGPUSIM_CONFIG="${GPGPUSIM_CONFIG:-gcc-13.3.0/cuda-12080/release}"
SIM_LIB_DIR="${SIM_LIB_DIR:-${ROOT_DIR}/lib/${GPGPUSIM_CONFIG}}"
SIM_BUILD_DIR="${SIM_BUILD_DIR:-${ROOT_DIR}/build/${GPGPUSIM_CONFIG}}"
GPU_CONFIG="${GPU_CONFIG:-SM100_B200_REDUCED}"
GPU_CONFIG_DIR="${ROOT_DIR}/configs/${GPU_CONFIG}"
CUDA_HOME="${CUDA_INSTALL_PATH:-/home/wzr/cuda}"
CXX="${CXX:-g++}"

detect_fa4_python() {
  if [[ -n "${FA4_PYTHON:-}" ]]; then
    echo "${FA4_PYTHON}"
  elif [[ -x "${ROOT_DIR}/../fa4-env-cu133/bin/python" ]]; then
    echo "${ROOT_DIR}/../fa4-env-cu133/bin/python"
  else
    echo python3
  fi
}

detect_inline_nvcc() {
  if [[ -n "${TCGEN05_INLINE_NVCC:-}" ]]; then
    echo "${TCGEN05_INLINE_NVCC}"
    return
  fi
  if [[ -n "${FA4_CU13_ROOT:-}" && -x "${FA4_CU13_ROOT}/bin/nvcc" ]]; then
    echo "${FA4_CU13_ROOT}/bin/nvcc"
    return
  fi

  local python_bin
  python_bin="$(detect_fa4_python)"
  if command -v "${python_bin}" >/dev/null 2>&1 ||
      [[ -x "${python_bin}" ]]; then
    "${python_bin}" <<'PY' || true
import sysconfig
from pathlib import Path

root = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13"
nvcc = root / "bin" / "nvcc"
if nvcc.exists():
    print(nvcc)
PY
  fi
}

run_parser_smoke() {
  local ptx="$1"
  (
    cd "${BUILD_DIR}"
    LD_LIBRARY_PATH="${SIM_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
      ./tcgen05_parser_smoke "${ptx}"
  )
}

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
  "${ROOT_DIR}/tests/dev/tcgen05/tcgen05_parser_smoke.cc" \
  -L"${SIM_LIB_DIR}" \
  -Wl,-rpath,"${SIM_LIB_DIR}" \
  -lcudart \
  -o "${BUILD_DIR}/tcgen05_parser_smoke"

run_parser_smoke "${PTX_FILE}"

if [[ "${CHECK_INLINE}" -eq 1 ]]; then
  INLINE_NVCC="$(detect_inline_nvcc)"
  if [[ -z "${INLINE_NVCC}" || ! -x "${INLINE_NVCC}" ]]; then
    echo "missing Blackwell-capable nvcc for --check-inline" >&2
    exit 1
  fi
  INLINE_SRC="${ROOT_DIR}/tests/dev/tcgen05/fixtures/tcgen05_instruction_surface_inline.cu"
  INLINE_PTX="${BUILD_DIR}/tcgen05_instruction_surface_inline.generated.ptx"
  "${INLINE_NVCC}" -std=c++17 -arch=sm_100a -ptx "${INLINE_SRC}" \
    -o "${INLINE_PTX}"
  run_parser_smoke "${INLINE_PTX}"
fi
