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
    build-essential=12.9ubuntu3 \
    g++=4:11.2.0-1ubuntu1 \
    gcc=4:11.2.0-1ubuntu1 \
    make=4.3-4.1build1 \
    cmake=3.22.1-1ubuntu1.22.04.2 \
    # Required by GPGPU-Sim
    xutils-dev=1:7.7+6ubuntu1 \
    bison=2:3.8.2+dfsg-1build1 \
    flex=2.6.4-8build2 \
    zlib1g-dev=1:1.2.11.dfsg-2ubuntu9.2 \
    libglu1-mesa-dev=9.0.2-1 \
    # Documentation
    doxygen=1.9.1-2ubuntu2 \
    graphviz=2.42.2-6ubuntu0.1 \
    # Version control
    git=1:2.34.1-1ubuntu1.15 \
    # Additional utilities
    wget=1.21.2-2ubuntu1.1 \
    curl=7.81.0-1ubuntu1.21 \
    vim=2:8.2.3995-1ubuntu2.24 \
    nano \
    gdb=12.1-0ubuntu1~22.04.2 \
    valgrind=1:3.18.1-1ubuntu2 \
    # Python for scripts
    python3=3.10.6-1~22.04.1 \
    python3-pip=22.0.2+dfsg-1ubuntu0.7 \
    python3-dev=3.10.6-1~22.04.1 \
    # AerialVision dependencies (optional)
    python3-numpy=1:1.21.5-1ubuntu22.04.1 \
    python3-matplotlib=3.5.1-2build1 \
    libpng-dev=1.6.37-3ubuntu0.1 \
    # Debugging and utilities
    htop=3.0.5-7build2 \
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
