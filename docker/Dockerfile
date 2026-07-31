# FlashGPU-Sim development image

ARG CUDA_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu22.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8.0

FROM ${UV_IMAGE} AS uv
FROM ${CUDA_IMAGE}

LABEL org.opencontainers.image.title="FlashGPU-Sim development environment"
LABEL org.opencontainers.image.description="CUDA 12.8 development environment for FlashGPU-Sim and TritonTrace"

ARG DEV_USER=flashgpusim
ARG DEV_UID=1000
ARG DEV_GID=1000

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bison \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    flex \
    gdb \
    git \
    htop \
    less \
    libglu1-mesa-dev \
    ninja-build \
    pkg-config \
    sudo \
    tree \
    valgrind \
    vim \
    xutils-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_LINK_MODE=copy
ENV VIRTUAL_ENV=/opt/flashgpusim-venv

RUN uv python install 3.12.3 \
    && uv venv --python 3.12.3 --seed "${VIRTUAL_ENV}" \
    && uv pip install --python "${VIRTUAL_ENV}/bin/python" \
        numpy==2.4.0 \
        torch==2.9.0 \
        triton==3.5.0 \
    && uv cache clean

RUN if ! getent group "${DEV_GID}" >/dev/null; then \
        groupadd --gid "${DEV_GID}" "${DEV_USER}"; \
    fi \
    && useradd --uid "${DEV_UID}" --gid "${DEV_GID}" \
        --create-home --shell /bin/bash "${DEV_USER}" \
    && printf '%s ALL=(ALL) NOPASSWD:ALL\n' "${DEV_USER}" \
        > "/etc/sudoers.d/${DEV_USER}" \
    && chmod 0440 "/etc/sudoers.d/${DEV_USER}" \
    && install -d -o "${DEV_UID}" -g "${DEV_GID}" \
        "/home/${DEV_USER}/.cache" \
        /workspace/flashgpu-sim

ENV CUDA_INSTALL_PATH=/usr/local/cuda
ENV PTXAS_CUDA_INSTALL_PATH=/usr/local/cuda
ENV PATH="${VIRTUAL_ENV}/bin:/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64
ENV HOME="/home/${DEV_USER}"
ENV XDG_CACHE_HOME="/home/${DEV_USER}/.cache"
ENV TRITON_CACHE_DIR="/home/${DEV_USER}/.cache/triton"
ENV TORCH_HOME="/home/${DEV_USER}/.cache/torch"
ENV PYTHONPATH=/workspace/flashgpu-sim/tools
ENV FLASHGPUSIM_WORKSPACE=/workspace/flashgpu-sim

COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/flashgpusim-entrypoint

USER ${DEV_USER}
WORKDIR /workspace/flashgpu-sim

ENTRYPOINT ["/usr/local/bin/flashgpusim-entrypoint"]
CMD ["bash"]
