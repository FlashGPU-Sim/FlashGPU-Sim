# Set CUDA_INSTALL_PATH only if not already set (respects Docker ENV or user override).
# If the caller has activated a Python venv with NVIDIA CUDA wheels, use its
# toolkit.  Otherwise prefer the local CUDA 12.8 install used to build the
# simulator; FA4 scripts set PTXAS_CUDA_INSTALL_PATH separately for CUDA 13 PTX.
if [ -z "${CUDA_INSTALL_PATH:-}" ]; then
  VENV_CUDA_NVCC=""
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    VENV_CUDA_NVCC=$(find "$VIRTUAL_ENV/lib" -path '*/site-packages/nvidia/cu*/bin/nvcc' -type f 2>/dev/null | sort -V | tail -n 1)
  fi

  if [ -n "$VENV_CUDA_NVCC" ]; then
    export CUDA_INSTALL_PATH="$(dirname "$(dirname "$VENV_CUDA_NVCC")")"
    echo "Using venv CUDA_INSTALL_PATH: $CUDA_INSTALL_PATH"
  elif [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_INSTALL_PATH=/usr/local/cuda-12.8
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
