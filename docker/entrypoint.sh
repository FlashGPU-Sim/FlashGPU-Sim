#!/usr/bin/env bash

set -e

workspace="${FLASHGPUSIM_WORKSPACE:-/workspace/flashgpu-sim}"
if [[ -d "${workspace}" ]]; then
  cd "${workspace}"
fi

if [[ -t 1 && "${1:-}" == "bash" ]]; then
  printf '%s\n' \
    "FlashGPU-Sim development environment (CUDA 12.8)" \
    "  Simulator: source setup_environment" \
    "  Triton capture: keep this native CUDA environment"
fi

exec "$@"
