# Set CUDA_INSTALL_PATH only if not already set (respects Docker ENV or user override)
if [ -z "$CUDA_INSTALL_PATH" ]; then
  # Try common CUDA installation locations
  if [ -d "/home/wzr/cuda" ]; then
    export CUDA_INSTALL_PATH=/home/wzr/cuda
  elif [ -d "/usr/local/cuda" ]; then
    export CUDA_INSTALL_PATH=/usr/local/cuda
  else
    echo "WARNING: CUDA not found at common locations. Please set CUDA_INSTALL_PATH manually."
  fi
else
  echo "Using existing CUDA_INSTALL_PATH: $CUDA_INSTALL_PATH"
fi
