#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
RUN_DIR="${SCRIPT_DIR}/run"
TRACKING_DIR="${RUN_DIR}/tracking"
LAUNCHER_DIR="${TRACKING_DIR}/launchers"
CAPTURE_LOG="${RUN_DIR}/capture.log"

BATCH="${FLASH_ATTN_BATCH:-32}"
HEADS="${FLASH_ATTN_HEADS:-32}"
SEQ_LEN="${FLASH_ATTN_SEQ_LEN:-512}"
HEAD_DIM="${FLASH_ATTN_HEAD_DIM:-64}"
CAUSAL="${FLASH_ATTN_CAUSAL:-1}"
TRITON_TARGET="${TRITON_TARGET:-sm120}"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${REPO_ROOT}/tutorials/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/tutorials/.venv/bin/python"
else
  PYTHON="$(command -v python3 || true)"
fi

echo "[1/3] Checking the offline Triton environment"

if [[ -z "${CUDA_INSTALL_PATH:-}" ]] ||
  [[ ! -x "${CUDA_INSTALL_PATH}/bin/nvcc" ]]; then
  echo "Error: CUDA_INSTALL_PATH must point to a CUDA Toolkit containing nvcc." >&2
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

if ! "${PYTHON}" -c "import TritonTrace" >/dev/null 2>&1; then
  echo "Error: TritonTrace is not installed in the selected Python environment." >&2
  echo "Install it with: ${PYTHON} -m pip install -e ${REPO_ROOT}/tools" >&2
  exit 1
fi

case "${CAUSAL}" in
  0)
    CAUSAL_ARGS=(--no-causal)
    ;;
  1)
    CAUSAL_ARGS=(--causal)
    ;;
  *)
    echo "Error: FLASH_ATTN_CAUSAL must be 0 or 1." >&2
    exit 1
    ;;
esac

export PATH="${CUDA_INSTALL_PATH}/bin:${PATH}"
mkdir -p "${RUN_DIR}"

echo "[2/3] Compiling Triton FlashAttention for ${TRITON_TARGET}"
"${PYTHON}" "${SCRIPT_DIR}/flash_attention.py" \
  --batch "${BATCH}" \
  --heads "${HEADS}" \
  --seq-len "${SEQ_LEN}" \
  --head-dim "${HEAD_DIM}" \
  --target "${TRITON_TARGET}" \
  "${CAUSAL_ARGS[@]}" \
  2>&1 | tee "${CAPTURE_LOG}"

shopt -s nullglob
MAKEFILES=("${LAUNCHER_DIR}"/*_launch1_Makefile)
shopt -u nullglob

if (( ${#MAKEFILES[@]} != 1 )); then
  echo "Error: expected one captured launcher, found ${#MAKEFILES[@]}." >&2
  exit 1
fi

LAUNCHER_MAKEFILE="$(basename -- "${MAKEFILES[0]}")"
LAUNCHER_NAME="${LAUNCHER_MAKEFILE%_Makefile}"
LAUNCHER="${LAUNCHER_DIR}/${LAUNCHER_NAME}"

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
