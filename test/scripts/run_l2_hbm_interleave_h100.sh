#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

BIN="${BIN:-${ROOT_DIR}/test/build/bin/microbench/memory/l2_hbm_interleave_bench_cuda128_sm_90a}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/test/run/L2_HBM_INTERLEAVE_H100_$(date +%Y%m%d_%H%M%S)}"
CASE_SET="${CASE_SET:-core}"
RUN_NATIVE="${RUN_NATIVE:-1}"
RUN_NCU="${RUN_NCU:-1}"
NCU="${NCU:-ncu}"
NCU_MODE="${NCU_MODE:-metrics}"

BLOCKS="${BLOCKS:-528}"
THREADS="${THREADS:-256}"
ITERS="${ITERS:-64}"
WARMUP="${WARMUP:-0}"
DATA_BYTES="${DATA_BYTES:-512M}"
TILE_BYTES="${TILE_BYTES:-48K}"
SMEM_BYTES="${SMEM_BYTES:-32768}"

NCU_METRICS="${NCU_METRICS:-sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_read.sum.per_second,dram__sectors_op_read.sum,dram__sectors_op_read.avg,dram__sectors_op_read.min,dram__sectors_op_read.max,fbpa__dram_read_bytes.sum,fbpa__dram_read_bytes.sum.per_second,fbpa__dram_read_sectors.sum,fbpa__dram_read_sectors.avg,fbpa__dram_read_sectors.min,fbpa__dram_read_sectors.max,lts__t_sector_hit_rate.pct,lts__t_sectors.sum,lts__t_sectors.avg,lts__t_sectors.min,lts__t_sectors.max,lts__t_sectors_srcunit_tex.sum,lts__t_sectors_srcunit_tex.avg,lts__t_sectors_srcunit_tex.min,lts__t_sectors_srcunit_tex.max,lts__t_sectors_srcunit_tex_lookup_hit.sum,lts__t_sectors_srcunit_tex_lookup_hit.avg,lts__t_sectors_srcunit_tex_lookup_hit.min,lts__t_sectors_srcunit_tex_lookup_hit.max,lts__t_sectors_srcunit_tex_lookup_miss.sum,lts__t_sectors_srcunit_tex_lookup_miss.avg,lts__t_sectors_srcunit_tex_lookup_miss.min,lts__t_sectors_srcunit_tex_lookup_miss.max}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--print-cases|--dry-run]

Environment:
  BIN          Benchmark binary. Default: ${BIN}
  OUT_DIR      Output directory. Default: timestamp under test/run.
  CASE_SET     core | stride | base | pair | cp | all. Default: core.
  RUN_NATIVE   Run native clock64/event pass. Default: 1.
  RUN_NCU      Run ncu. Default: 1.
  NCU_MODE     metrics | full. Default: metrics.
  BLOCKS       Default blocks. Default: ${BLOCKS}.
  THREADS      Default threads. Default: ${THREADS}.
  ITERS        Default measured iterations. Default: ${ITERS}.
  DATA_BYTES   Default allocation size. Default: ${DATA_BYTES}.
  TILE_BYTES   Default bytes touched per CTA iteration. Default: ${TILE_BYTES}.

Build for H100:
  make -C ${ROOT_DIR}/test/src/microbench/memory \\
    CUDA_HOME=/usr/local/cuda-12.8 ARCH=sm_90a PTX_PROFILE=compute_90a interleave
EOF
}

PRINT_CASES=0
DRY_RUN=0
for arg in "$@"; do
  case "${arg}" in
    --help|-h)
      usage
      exit 0
      ;;
    --print-cases)
      PRINT_CASES=1
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      usage >&2
      exit 1
      ;;
  esac
done

common_args() {
  printf -- "--blocks=%s --threads=%s --iters=%s --warmup=%s --data-bytes=%s --tile-bytes=%s --smem-bytes=%s --event" \
    "${BLOCKS}" "${THREADS}" "${ITERS}" "${WARMUP}" "${DATA_BYTES}" "${TILE_BYTES}" "${SMEM_BYTES}"
}

declare -a CASE_IDS=()
declare -a CASE_ARGS=()

add_case() {
  CASE_IDS+=("$1")
  shift
  CASE_ARGS+=("$*")
}

add_core_cases() {
  local common
  common="$(common_args)"
  add_case "ldg_stream_contig" "--op=ldg --pattern=stream ${common}"
  add_case "cp_async_stream_contig" "--op=cp_async --pattern=stream ${common}"
  add_case "ldg_stride_32" "--op=ldg --pattern=stride --stride-bytes=32 ${common}"
  add_case "ldg_stride_128" "--op=ldg --pattern=stride --stride-bytes=128 ${common}"
  add_case "ldg_stride_512" "--op=ldg --pattern=stride --stride-bytes=512 ${common}"
  add_case "ldg_stride_4096" "--op=ldg --pattern=stride --stride-bytes=4096 ${common}"
  add_case "ldg_stride_32768" "--op=ldg --pattern=stride --stride-bytes=32768 ${common}"
  add_case "ldg_hot_reuse_32m" "--op=ldg --pattern=stream --blocks=${BLOCKS} --threads=${THREADS} --iters=128 --warmup=16 --data-bytes=32M --tile-bytes=32K --smem-bytes=${SMEM_BYTES} --event"
}

add_stride_cases() {
  local common
  common="$(common_args)"
  for stride in 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576; do
    add_case "ldg_stride_${stride}" "--op=ldg --pattern=stride --stride-bytes=${stride} ${common}"
  done
}

add_base_cases() {
  local common
  common="$(common_args)"
  for offset in 0 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536; do
    add_case "ldg_base_${offset}" "--op=ldg --pattern=stream --base-offset=${offset} ${common}"
  done
}

add_pair_cases() {
  local common
  common="$(common_args)"
  for delta in 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144; do
    add_case "ldg_pair_delta_${delta}" "--op=ldg --pattern=pair --stride-bytes=4096 --pair-delta-bytes=${delta} ${common}"
  done
}

add_cp_cases() {
  local common
  common="$(common_args)"
  for stride in 128 512 4096 32768 262144; do
    add_case "cp_async_stride_${stride}" "--op=cp_async --pattern=stride --stride-bytes=${stride} ${common}"
  done
  for offset in 0 128 512 4096 32768 65536; do
    add_case "cp_async_base_${offset}" "--op=cp_async --pattern=stream --base-offset=${offset} ${common}"
  done
}

case "${CASE_SET}" in
  core)
    add_core_cases
    ;;
  stride)
    add_stride_cases
    ;;
  base)
    add_base_cases
    ;;
  pair)
    add_pair_cases
    ;;
  cp)
    add_cp_cases
    ;;
  all)
    add_core_cases
    add_stride_cases
    add_base_cases
    add_pair_cases
    add_cp_cases
    ;;
  *)
    echo "Unknown CASE_SET=${CASE_SET}" >&2
    exit 1
    ;;
esac

if [[ "${PRINT_CASES}" == "1" ]]; then
  for i in "${!CASE_IDS[@]}"; do
    printf "%02d,%s,%s\n" "${i}" "${CASE_IDS[$i]}" "${CASE_ARGS[$i]}"
  done
  exit 0
fi

mkdir -p "${OUT_DIR}/native" "${OUT_DIR}/ncu" "${OUT_DIR}/logs" \
         "${OUT_DIR}/csv" "${OUT_DIR}/provenance"

{
  echo "date=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "root=${ROOT_DIR}"
  echo "bin=${BIN}"
  echo "case_set=${CASE_SET}"
  echo "run_native=${RUN_NATIVE}"
  echo "run_ncu=${RUN_NCU}"
  echo "ncu_mode=${NCU_MODE}"
  echo "ncu_metrics=${NCU_METRICS}"
} >"${OUT_DIR}/provenance/run_env.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q >"${OUT_DIR}/provenance/nvidia_smi_q.txt" || true
fi
"${NCU}" --version >"${OUT_DIR}/provenance/ncu_version.txt" 2>&1 || true

if [[ ! -x "${BIN}" ]]; then
  echo "Benchmark binary not found or not executable: ${BIN}" >&2
  exit 1
fi

status_csv="${OUT_DIR}/status.csv"
echo "case_id,native_status,ncu_status,rep,args" >"${status_csv}"

for i in "${!CASE_IDS[@]}"; do
  case_id="${CASE_IDS[$i]}"
  args="${CASE_ARGS[$i]}"
  native_log="${OUT_DIR}/native/${case_id}.log"
  sample_csv="${OUT_DIR}/csv/${case_id}.samples.csv"
  ncu_log="${OUT_DIR}/logs/${case_id}.ncu.log"
  rep="${OUT_DIR}/ncu/${case_id}.ncu-rep"
  raw_csv="${OUT_DIR}/ncu/${case_id}.raw.csv"
  raw_instances_csv="${OUT_DIR}/ncu/${case_id}.raw.instances.csv"
  details_csv="${OUT_DIR}/ncu/${case_id}.details.csv"

  read -r -a arg_array <<<"${args}"
  native_status=0
  ncu_status=0

  echo "=== ${case_id} ==="
  echo "args: ${args}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "native: ${BIN} ${args} --csv=${sample_csv}"
    echo "ncu: ${NCU} --target-processes all --export ${rep} ..."
    continue
  fi

  if [[ "${RUN_NATIVE}" == "1" ]]; then
    if ! "${BIN}" "${arg_array[@]}" --csv="${sample_csv}" >"${native_log}" 2>&1; then
      native_status=1
      cat "${native_log}" >&2 || true
    fi
  fi

  if [[ "${RUN_NCU}" == "1" ]]; then
    if [[ "${NCU_MODE}" == "full" ]]; then
      ncu_profile_args=(--set full)
    else
      ncu_profile_args=(--metrics "${NCU_METRICS}")
    fi
    if ! "${NCU}" --target-processes all --cache-control all --clock-control none \
      "${ncu_profile_args[@]}" --export "${rep}" --force-overwrite \
      "${BIN}" "${arg_array[@]}" >"${ncu_log}" 2>&1; then
      ncu_status=1
      cat "${ncu_log}" >&2 || true
    elif [[ -f "${rep}" ]]; then
      "${NCU}" --import "${rep}" --csv --page raw --print-units base \
        >"${raw_csv}" 2>>"${ncu_log}" || true
      "${NCU}" --import "${rep}" --csv --page raw --print-units base \
        --print-metric-instances values \
        >"${raw_instances_csv}" 2>>"${ncu_log}" || true
      "${NCU}" --import "${rep}" --csv --page details \
        >"${details_csv}" 2>>"${ncu_log}" || true
    fi
  fi

  echo "${case_id},${native_status},${ncu_status},${rep},\"${args}\"" >>"${status_csv}"
done

python3 "${SCRIPT_DIR}/summarize_l2_hbm_interleave_ncu.py" "${OUT_DIR}/ncu" \
  --out "${OUT_DIR}/summary_instances.csv" || true

echo "out_dir=${OUT_DIR}"
echo "status=${status_csv}"
echo "summary=${OUT_DIR}/summary_instances.csv"
