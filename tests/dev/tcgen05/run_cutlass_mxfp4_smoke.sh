#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)

: "${CUDA_INSTALL_PATH:?set CUDA_INSTALL_PATH to a CUDA 13.3+ toolkit}"
: "${CUTLASS_ROOT:?set CUTLASS_ROOT to an NVIDIA CUTLASS checkout}"

nvcc="$CUDA_INSTALL_PATH/bin/nvcc"
cuobjdump="$CUDA_INSTALL_PATH/bin/cuobjdump"
source_file="$script_dir/cutlass_mxfp4_smoke.cu"
build_dir="${MXFP4_BUILD_DIR:-$repo_root/tests/build/tcgen05_cutlass_mxfp4}"
run_dir="$build_dir/run"
binary="$build_dir/cutlass_mxfp4_smoke_shared"
ptx_name=cutlass_mxfp4_smoke.sm_100a.ptx
ptx_file="$run_dir/$ptx_name"
gtest_filter="${GTEST_FILTER:-FlashGpuSimCutlassMxfp4.*}"
timeout_seconds="${SIM_TIMEOUT_SECONDS:-600}"

if [[ ! -x "$nvcc" || ! -x "$cuobjdump" ]]; then
  echo "CUDA tools not found under CUDA_INSTALL_PATH=$CUDA_INSTALL_PATH" >&2
  exit 1
fi
if [[ ! -f "$CUTLASS_ROOT/include/cutlass/cutlass.h" ]]; then
  echo "CUTLASS_ROOT does not contain include/cutlass/cutlass.h" >&2
  exit 1
fi

mkdir -p "$build_dir" "$run_dir"
env PATH="$CUDA_INSTALL_PATH/bin:$PATH" \
  make -C "$repo_root/tests" setup-gtest build/obj/gtest_main.a

"$nvcc" -std=c++17 -O2 -lineinfo --expt-relaxed-constexpr \
  -arch=sm_100a -cudart shared \
  -I"$CUTLASS_ROOT/include" \
  -I"$CUTLASS_ROOT/tools/util/include" \
  -I"$CUTLASS_ROOT/test/unit/gemm/device" \
  -I"$repo_root/tests/gtest/googletest/include" \
  "$source_file" "$repo_root/tests/build/obj/gtest_main.a" \
  -lpthread -o "$binary"

"$cuobjdump" --dump-ptx "$binary" > "$ptx_file"
if ! rg -q \
  'tcgen05\.mma\.cta_group::1\.kind::mxf4\.block_scale\.block32' \
  "$ptx_file"; then
  echo "compiled PTX does not contain strict MXFP4 tcgen05.mma" >&2
  exit 1
fi

cp -a "$repo_root/configs/SM100_B200_REDUCED/." "$run_dir/"

export PTXAS_CUDA_INSTALL_PATH="${PTXAS_CUDA_INSTALL_PATH:-$CUDA_INSTALL_PATH}"
export PTX_SIM_USE_PTX_FILE=1
export PTX_SIM_KERNELFILE="$ptx_name"
export CUOBJDUMP_SIM_FILE=jj

export CUDA_INSTALL_PATH
set +u
source "$repo_root/setup_environment"
set -u

cd "$run_dir"
timeout "$timeout_seconds" "$binary" --gtest_filter="$gtest_filter"
