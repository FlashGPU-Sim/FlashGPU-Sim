# Set CUDA_INSTALL_PATH only if not already set (respects Docker ENV or user override).
# For SM120/Triton flows, prefer the repo-local Python venv CUDA toolkit because
# it carries CUDA 13 nvcc/ptxas/cuobjdump for PTX 9.x and sm_120a kernels.
if [ -z "${CUDA_INSTALL_PATH:-}" ]; then
  SETUP_SEARCH_DIR="$(pwd)"
  while [ "$SETUP_SEARCH_DIR" != "/" ] && [ ! -d "$SETUP_SEARCH_DIR/test/triton_trace/.venv" ]; do
    SETUP_SEARCH_DIR="$(dirname "$SETUP_SEARCH_DIR")"
  done

  VENV_CUDA_NVCC=""
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    VENV_CUDA_NVCC=$(find "$VIRTUAL_ENV/lib" -path '*/site-packages/nvidia/cu*/bin/nvcc' -type f 2>/dev/null | sort -V | tail -n 1)
  fi
  if [ -z "$VENV_CUDA_NVCC" ]; then
    VENV_CUDA_NVCC=$(find "$SETUP_SEARCH_DIR/test/triton_trace/.venv/lib" -path '*/site-packages/nvidia/cu*/bin/nvcc' -type f 2>/dev/null | sort -V | tail -n 1)
  fi

  if [ -n "$VENV_CUDA_NVCC" ]; then
    export CUDA_INSTALL_PATH="$(dirname "$(dirname "$VENV_CUDA_NVCC")")"
    echo "Using venv CUDA_INSTALL_PATH: $CUDA_INSTALL_PATH"
  elif [ -d "/home/wzr/cuda" ]; then
    export CUDA_INSTALL_PATH=/home/wzr/cuda
  elif [ -d "/usr/local/cuda" ]; then
    export CUDA_INSTALL_PATH=/usr/local/cuda
  else
    echo "WARNING: CUDA not found at common locations. Please set CUDA_INSTALL_PATH manually."
  fi
else
  echo "Using existing CUDA_INSTALL_PATH: $CUDA_INSTALL_PATH"
fi
