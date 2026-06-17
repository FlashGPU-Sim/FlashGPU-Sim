#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-./mma_saturation_bench_cuda128_sm90a}"
CUDA128_LIB="${CUDA128_LIB:-${ROOT}/prebuilt_cuda128/lib64}"
OUT_DIR="${OUT_DIR:-mma_saturation_h100_$(date +%Y%m%d_%H%M%S)}"

WARPS="${WARPS:-8}"
UNROLL="${UNROLL:-16}"
BLOCKS_PER_SM="${BLOCKS_PER_SM:-8}"
REPEAT="${REPEAT:-32768}"
RUN_NCU="${RUN_NCU:-1}"

mkdir -p "${OUT_DIR}"

export LD_LIBRARY_PATH="${CUDA128_LIB}:${LD_LIBRARY_PATH:-}"

{
  echo "date=$(date -Is)"
  echo "host=$(hostname)"
  echo "bin=${BIN}"
  echo "cuda128_lib=${CUDA128_LIB}"
  echo "ld_library_path=${LD_LIBRARY_PATH}"
  echo "warps=${WARPS}"
  echo "unroll=${UNROLL}"
  echo "blocks_per_sm=${BLOCKS_PER_SM}"
  echo "repeat=${REPEAT}"
  echo "run_ncu=${RUN_NCU}"
  if [[ -x "${BIN}" ]]; then
    sha256sum "${BIN}"
  fi
  nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.current.sm,clocks.current.memory,pstate,driver_version --format=csv,noheader || true
} > "${OUT_DIR}/env.txt"

ldd "${BIN}" > "${OUT_DIR}/ldd.txt"

COMMON_ARGS=(
  --warps "${WARPS}"
  --unroll "${UNROLL}"
  --blocks-per-sm "${BLOCKS_PER_SM}"
  --repeat "${REPEAT}"
)

"${BIN}" "${COMMON_ARGS[@]}" --warmup 2 --samples 5 \
  > "${OUT_DIR}/event_timing.csv"

if [[ "${RUN_NCU}" == "1" ]] && command -v ncu >/dev/null 2>&1; then
  ncu --target-processes all \
    --set full \
    --force-overwrite \
    --export "${OUT_DIR}/ncu_full" \
    "${BIN}" "${COMMON_ARGS[@]}" --warmup 0 --samples 1 \
    > "${OUT_DIR}/ncu_full.stdout" 2> "${OUT_DIR}/ncu_full.stderr"

  ncu --target-processes all \
    --csv --page raw \
    --metrics sm__cycles_elapsed.avg.per_second,smsp__inst_executed_pipe_tensor.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active \
    "${BIN}" "${COMMON_ARGS[@]}" --warmup 0 --samples 1 \
    > "${OUT_DIR}/ncu_tensor_metrics.csv" 2> "${OUT_DIR}/ncu_tensor_metrics.stderr" || true
else
  echo "ncu skipped or not found" > "${OUT_DIR}/ncu_full.stderr"
fi

{
  echo "after=$(date -Is)"
  nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.current.sm,clocks.current.memory,pstate,driver_version --format=csv,noheader || true
} >> "${OUT_DIR}/env.txt"

echo "${OUT_DIR}"
