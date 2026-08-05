#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
compose_file="${repo_root}/docker/docker-compose.yml"
gpu_compose_file="${repo_root}/docker/docker-compose.gpu.yml"
service="flashgpusim"
image="flashgpu-sim-dev:cuda12.8"

export FLASHGPUSIM_UID="${FLASHGPUSIM_UID:-$(id -u)}"
export FLASHGPUSIM_GID="${FLASHGPUSIM_GID:-$(id -g)}"

usage() {
  cat <<EOF
Usage:
  ./docker.sh build
  ./docker.sh shell
  ./docker.sh shell --gpu [DEVICE]

Commands:
  build                 Build the development image.
  shell                 Start a disposable CPU-only development shell.
  shell --gpu           Start a shell with all NVIDIA GPUs available.
  shell --gpu DEVICE    Start a GPU-enabled shell and select DEVICE with
                        CUDA_VISIBLE_DEVICES.
EOF
}

require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not installed." >&2
    exit 1
  fi

  if ! docker info >/dev/null 2>&1; then
    echo "Error: the Docker daemon is unavailable." >&2
    exit 1
  fi
}

build_image() {
  require_docker
  docker compose -f "${compose_file}" build "${service}"
}

open_shell() {
  local use_gpu=false
  local device=""

  if [[ $# -gt 0 ]]; then
    if [[ "$1" != "--gpu" ]]; then
      echo "Error: unknown shell option '$1'." >&2
      usage >&2
      exit 2
    fi
    use_gpu=true
    shift

    if [[ $# -gt 0 ]]; then
      device="$1"
      shift
      if [[ ! "${device}" =~ ^[0-9]+$ ]]; then
        echo "Error: GPU device must be a non-negative integer." >&2
        exit 2
      fi
    fi
  fi

  if [[ $# -gt 0 ]]; then
    echo "Error: too many shell arguments." >&2
    usage >&2
    exit 2
  fi

  require_docker

  if ! docker image inspect "${image}" >/dev/null 2>&1; then
    echo "Error: development image '${image}' was not found." >&2
    echo "Run './docker.sh build' first." >&2
    exit 1
  fi

  local -a compose=(docker compose -f "${compose_file}")
  local -a run_args=(run --rm)

  if [[ "${use_gpu}" == true ]]; then
    compose+=(-f "${gpu_compose_file}")
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      echo "Warning: nvidia-smi was not found on the host." >&2
    fi
  fi

  if [[ -n "${device}" ]]; then
    run_args+=(-e "CUDA_VISIBLE_DEVICES=${device}")
  fi

  "${compose[@]}" "${run_args[@]}" "${service}" bash
}

case "${1:-help}" in
  build)
    shift
    if [[ $# -ne 0 ]]; then
      echo "Error: build does not accept arguments." >&2
      exit 2
    fi
    build_image
    ;;
  shell)
    shift
    open_shell "$@"
    ;;
  help | --help | -h)
    usage
    ;;
  *)
    echo "Error: unknown command '${1}'." >&2
    usage >&2
    exit 2
    ;;
esac
