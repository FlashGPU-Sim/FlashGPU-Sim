#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUN_DIR="${SCRIPT_DIR}/run"
TRACKING_DIR="${RUN_DIR}/tracking"
LAUNCHER_DIR="${TRACKING_DIR}/launchers"
CAPTURE_LOG="${RUN_DIR}/capture.log"

M="${GEMM_M:-2560}"
N="${GEMM_N:-64}"
K="${GEMM_K:-2560}"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${REPO_ROOT}/tutorials/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/tutorials/.venv/bin/python"
else
  PYTHON="$(command -v python3 || true)"
fi

echo "[1/3] Checking the native GPU environment"

if [[ -z "${CUDA_INSTALL_PATH:-}" ]] ||
  [[ ! -x "${CUDA_INSTALL_PATH}/bin/nvcc" ]]; then
  echo "Error: CUDA_INSTALL_PATH must point to a CUDA Toolkit containing nvcc." >&2
  exit 1
fi

if [[ -n "${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" ]] ||
  [[ "${LD_LIBRARY_PATH:-}" == *"${REPO_ROOT}/lib/"* ]]; then
  echo "Error: Triton capture requires a clean shell using the real GPU." >&2
  echo "Open a new shell, set CUDA_INSTALL_PATH, and run this script again." >&2
  exit 1
fi

if [[ -z "${PYTHON}" ]]; then
  echo "Error: Python 3 was not found." >&2
  exit 1
fi

if ! "${PYTHON}" -c "import numpy, torch, triton" >/dev/null 2>&1; then
  echo "Error: the selected Python environment must provide NumPy, PyTorch, and Triton." >&2
  echo "Activate a compatible virtual environment and run this script again." >&2
  exit 1
fi

if ! "${PYTHON}" -c "import tritontrace" >/dev/null 2>&1; then
  echo "Error: TritonTrace is not installed in the selected Python environment." >&2
  echo "Install it with: ${PYTHON} -m pip install -e ${REPO_ROOT}/tools" >&2
  exit 1
fi

if ! "${PYTHON}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  echo "Error: PyTorch cannot access a CUDA-capable GPU." >&2
  exit 1
fi

export PATH="${CUDA_INSTALL_PATH}/bin:${PATH}"
mkdir -p "${RUN_DIR}"

echo "[2/3] Capturing Triton GEMM (${M}x${N}x${K})"
"${PYTHON}" "${SCRIPT_DIR}/gemm.py" --m "${M}" --n "${N}" --k "${K}" \
  2>&1 | tee "${CAPTURE_LOG}"

LAUNCHER_NAME="kernel_tma_gemm_launch1"
LAUNCHER_MAKEFILE="${LAUNCHER_NAME}_Makefile"
LAUNCHER="${LAUNCHER_DIR}/${LAUNCHER_NAME}"

if [[ ! -f "${LAUNCHER_DIR}/${LAUNCHER_MAKEFILE}" ]]; then
  echo "Error: tracker did not generate ${LAUNCHER_MAKEFILE}." >&2
  exit 1
fi

if [[ ! -f "${LAUNCHER_DIR}/${LAUNCHER_NAME}_kernel.ptx" ]]; then
  echo "Error: tracker did not generate the captured PTX." >&2
  exit 1
fi

echo "[3/3] Building the standalone launcher"
make --no-print-directory -C "${LAUNCHER_DIR}" -f "${LAUNCHER_MAKEFILE}"

if [[ ! -x "${LAUNCHER}" ]]; then
  echo "Error: launcher build did not produce ${LAUNCHER}." >&2
  exit 1
fi

echo "Capture completed successfully."
echo "Capture log: ${CAPTURE_LOG}"
