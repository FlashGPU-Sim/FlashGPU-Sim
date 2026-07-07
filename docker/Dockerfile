# GPGPU-Sim Development Docker Image
# Based on NVIDIA CUDA 12.8 Official Image

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

LABEL maintainer="GPGPU-Sim Development Team"
LABEL description="GPGPU-Sim development environment with CUDA 12.8"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set CUDA environment variables
ENV CUDA_INSTALL_PATH=/usr/local/cuda
ENV PATH="${CUDA_INSTALL_PATH}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_INSTALL_PATH}/lib64:${LD_LIBRARY_PATH}"

# Install GPGPU-Sim dependencies (versions pinned for Ubuntu 22.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    g++ \
    gcc \
    make \
    cmake \
    # Required by GPGPU-Sim
    xutils-dev \
    bison \
    flex \
    zlib1g-dev \
    libglu1-mesa-dev \
    # Documentation
    doxygen \
    graphviz \
    # Version control
    git \
    # Additional utilities
    wget \
    curl \
    vim \
    nano \
    gdb \
    valgrind \
    # Python for scripts
    python3 \
    python3-pip \
    python3-dev \
    # AerialVision dependencies (optional)
    python3-numpy \
    python3-matplotlib \
    libpng-dev \
    # Debugging and utilities
    htop \
    less \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for AerialVision
RUN pip3 install --no-cache-dir \
    ply \
    pmw || true

# Create working directory
WORKDIR /gpgpu-sim

# Set up environment variables for GPGPU-Sim
ENV GPGPUSIM_ROOT=/gpgpu-sim
ENV PTXAS_CUDA_INSTALL_PATH=${CUDA_INSTALL_PATH}

# Create necessary directories
RUN mkdir -p /gpgpu-sim/lib /gpgpu-sim/build

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
