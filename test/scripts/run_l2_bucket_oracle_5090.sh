#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
ARCH="${ARCH:-sm_120a}"
PTX_PROFILE="${PTX_PROFILE:-compute_120a}"
BIN="${BIN:-${ROOT_DIR}/test/build/bin/microbench/memory/l2_hbm_interleave_bench_cuda128_${ARCH}}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/test/run/L2_BUCKET_ORACLE_5090_$(date +%Y%m%d_%H%M%S)}"
NCU="${NCU:-ncu}"
RUN_NCU="${RUN_NCU:-1}"
PLAN_ONLY="${PLAN_ONLY:-0}"
BUILD="${BUILD:-1}"

PAGES="${PAGES:-4}"
SAMPLE_LINES="${SAMPLE_LINES:-32}"
PAIRS_PER_BIT="${PAIRS_PER_BIT:-4}"
RANDOM_PAIRS="${RANDOM_PAIRS:-0}"
MAX_PAIRS="${MAX_PAIRS:-0}"
BIT_FIRST="${BIT_FIRST:-7}"
BIT_LAST="${BIT_LAST:-20}"
SEED="${SEED:-1}"

BLOCKS="${BLOCKS:-170}"
THREADS="${THREADS:-256}"
ITERS="${ITERS:-64}"
WARMUP="${WARMUP:-0}"
TILE_BYTES="${TILE_BYTES:-4096}"
DATA_BYTES="${DATA_BYTES:-128M}"
SMEM_BYTES="${SMEM_BYTES:-32768}"

NCU_METRICS="${NCU_METRICS:-gpu__time_duration.sum,sm__cycles_elapsed.avg,lts__t_sectors_srcunit_tex.sum,lts__t_sectors_srcunit_tex.max,lts__t_sectors_srcunit_tex.avg,lts__t_sectors_srcunit_tex.min,lts__t_sectors_srcunit_tex_lookup_hit.sum,lts__t_sectors_srcunit_tex_lookup_miss.sum}"

usage() {
  cat <<EOF
Usage: $(basename "$0")

Environment controls:
  OUT_DIR=${OUT_DIR}
  RUN_NCU=${RUN_NCU} PLAN_ONLY=${PLAN_ONLY} BUILD=${BUILD}
  PAGES=${PAGES} SAMPLE_LINES=${SAMPLE_LINES} PAIRS_PER_BIT=${PAIRS_PER_BIT}
  RANDOM_PAIRS=${RANDOM_PAIRS} MAX_PAIRS=${MAX_PAIRS}
  BLOCKS=${BLOCKS} THREADS=${THREADS} ITERS=${ITERS} TILE_BYTES=${TILE_BYTES}

This keeps one allocation alive and launches one fixed-pair kernel per pair.
NCU raw rows are joined with the emitted batch CSV by launch order.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

mkdir -p "${OUT_DIR}/ncu" "${OUT_DIR}/logs" "${OUT_DIR}/csv" \
         "${OUT_DIR}/provenance"

PAIR_PLAN="${OUT_DIR}/csv/pair_plan.csv"
BATCH_CSV="${OUT_DIR}/csv/batch_observed.csv"
REP="${OUT_DIR}/ncu/l2_bucket_oracle.ncu-rep"
RAW_CSV="${OUT_DIR}/ncu/l2_bucket_oracle.raw.csv"
JOINED_CSV="${OUT_DIR}/csv/l2_bucket_oracle.joined.csv"
SUMMARY_TXT="${OUT_DIR}/summary.txt"

{
  echo "date=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "root=${ROOT_DIR}"
  echo "cuda_home=${CUDA_HOME}"
  echo "arch=${ARCH}"
  echo "ptx_profile=${PTX_PROFILE}"
  echo "bin=${BIN}"
  echo "pages=${PAGES}"
  echo "sample_lines=${SAMPLE_LINES}"
  echo "pairs_per_bit=${PAIRS_PER_BIT}"
  echo "random_pairs=${RANDOM_PAIRS}"
  echo "max_pairs=${MAX_PAIRS}"
  echo "bit_first=${BIT_FIRST}"
  echo "bit_last=${BIT_LAST}"
  echo "ncu_metrics=${NCU_METRICS}"
} >"${OUT_DIR}/provenance/run_env.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q >"${OUT_DIR}/provenance/nvidia_smi_q.txt" || true
fi
"${NCU}" --version >"${OUT_DIR}/provenance/ncu_version.txt" 2>&1 || true

if [[ "${BUILD}" == "1" ]]; then
  make -C "${ROOT_DIR}/test/src/microbench/memory" \
    CUDA_HOME="${CUDA_HOME}" ARCH="${ARCH}" PTX_PROFILE="${PTX_PROFILE}" \
    interleave >"${OUT_DIR}/logs/build.log" 2>&1
fi

python3 "${SCRIPT_DIR}/generate_l2_bucket_pair_plan.py" \
  --out "${PAIR_PLAN}" \
  --pages "${PAGES}" \
  --sample-lines "${SAMPLE_LINES}" \
  --pairs-per-bit "${PAIRS_PER_BIT}" \
  --random-pairs "${RANDOM_PAIRS}" \
  --max-pairs "${MAX_PAIRS}" \
  --bit-first "${BIT_FIRST}" \
  --bit-last "${BIT_LAST}" \
  --seed "${SEED}" 2>&1 | tee "${OUT_DIR}/logs/plan.log"

if [[ "${PLAN_ONLY}" == "1" ]]; then
  echo "out_dir=${OUT_DIR}"
  echo "pair_plan=${PAIR_PLAN}"
  exit 0
fi

BENCH_ARGS=(
  --op=ldg
  --pattern=fixed_pair
  --alloc=vmm
  --blocks="${BLOCKS}"
  --threads="${THREADS}"
  --iters="${ITERS}"
  --warmup="${WARMUP}"
  --data-bytes="${DATA_BYTES}"
  --tile-bytes="${TILE_BYTES}"
  --smem-bytes="${SMEM_BYTES}"
  --pair-csv="${PAIR_PLAN}"
  --batch-csv="${BATCH_CSV}"
)

if [[ "${RUN_NCU}" == "1" ]]; then
  "${NCU}" --target-processes all --cache-control all --clock-control none \
    --metrics "${NCU_METRICS}" --export "${REP}" --force-overwrite \
    "${BIN}" "${BENCH_ARGS[@]}" >"${OUT_DIR}/logs/ncu.log" 2>&1
  "${NCU}" --import "${REP}" --csv --page raw --print-units base \
    >"${RAW_CSV}" 2>>"${OUT_DIR}/logs/ncu.log"
  python3 "${SCRIPT_DIR}/analyze_l2_bucket_pair_ncu.py" \
    --pairs "${BATCH_CSV}" \
    --ncu-raw "${RAW_CSV}" \
    --joined-out "${JOINED_CSV}" | tee "${SUMMARY_TXT}"
else
  "${BIN}" "${BENCH_ARGS[@]}" >"${OUT_DIR}/logs/native.log" 2>&1
fi

echo "out_dir=${OUT_DIR}"
echo "pair_plan=${PAIR_PLAN}"
echo "batch_csv=${BATCH_CSV}"
if [[ "${RUN_NCU}" == "1" ]]; then
  echo "ncu_report=${REP}"
  echo "raw_csv=${RAW_CSV}"
  echo "joined_csv=${JOINED_CSV}"
  echo "summary=${SUMMARY_TXT}"
fi
