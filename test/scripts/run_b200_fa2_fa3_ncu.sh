#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${TEST_DIR}/.." && pwd)"

if [[ -z "${PREBUILT_ROOT:-}" && -d "${SCRIPT_DIR}/../bin" ]]; then
  PREBUILT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

if [[ -n "${PREBUILT_ROOT:-}" ]]; then
  BIN_ROOT="${BIN_ROOT:-${PREBUILT_ROOT}/bin}"
  DEFAULT_OUT_BASE="${PREBUILT_ROOT}/../results"
else
  BIN_ROOT="${BIN_ROOT:-${TEST_DIR}/build/bin/hopper}"
  DEFAULT_OUT_BASE="${ROOT_DIR}/../b200_native_collect/results"
fi

OUT_DIR="${OUT_DIR:-${DEFAULT_OUT_BASE}/B200_FA2_FA3_NCU_$(date +%Y%m%d_%H%M%S)}"
RUN_FAMILIES="${RUN_FAMILIES:-fa2 fa3}"
RUN_GROUPS="${RUN_GROUPS:-smoke small}"
SELECT_CASES="${SELECT_CASES:-}"
RUN_NATIVE="${RUN_NATIVE:-1}"
RUN_NCU="${RUN_NCU:-1}"
RUN_NCU_ON_NATIVE_FAIL="${RUN_NCU_ON_NATIVE_FAIL:-1}"
RESUME="${RESUME:-1}"
FORCE="${FORCE:-0}"
NCU_SET="${NCU_SET:-full}"
NCU_METRICS="${NCU_METRICS:-}"
NCU_KERNEL_NAME="${NCU_KERNEL_NAME:-}"
NCU_EXTRA_ARGS="${NCU_EXTRA_ARGS:-}"

usage() {
  cat <<'EOF'
Usage:
  run_b200_fa2_fa3_ncu.sh [--print-cases|--dry-run]

Environment:
  PREBUILT_ROOT              Bundle produced by prepare_b200_fa2_fa3_prebuilt.sh.
  BIN_ROOT                   Binary directory. Defaults to $PREBUILT_ROOT/bin.
  OUT_DIR                    Result directory. Defaults under ../b200_native_collect
                             or $PREBUILT_ROOT/../results.
  RUN_FAMILIES               "fa2 fa3", "fa2", or "fa3". Default "fa2 fa3".
  RUN_GROUPS                 "smoke small medium large" or subset. Default
                             "smoke small".
  SELECT_CASES               Space-separated case ids or case names.
  RUN_NATIVE                 Run native sanity before NCU. Default 1.
  RUN_NCU                    Run Nsight Compute. Default 1.
  RUN_NCU_ON_NATIVE_FAIL     Still run NCU after native failure. Default 1.
  NCU_SET                    NCU section set. Default full.
  NCU_METRICS                Optional comma-separated metric list. Overrides set.
  NCU_KERNEL_NAME            Optional --kernel-name filter.
  NCU_EXTRA_ARGS             Extra words appended to ncu command.
  RESUME                     Skip done cases. Default 1.
  FORCE                      Rerun done cases. Default 0.

The script intentionally writes B200 results outside micro26 directories.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

DRY_RUN=0
PRINT_CASES=0
case "${1:-}" in
  --dry-run)
    DRY_RUN=1
    ;;
  --print-cases)
    PRINT_CASES=1
    ;;
esac

if [[ -n "${PREBUILT_ROOT:-}" && -d "${PREBUILT_ROOT}/lib" ]]; then
  export LD_LIBRARY_PATH="${PREBUILT_ROOT}/lib:${LD_LIBRARY_PATH:-}"
fi

CASES=(
  "fa2|smoke|h32d64_full|H32D64FullB2S128|2|128|32|64|full|run_fa2_smoke_h32d64_full_tests|Fa2PrefillFp16SmokeTest.H32D64FullB2S128"
  "fa2|smoke|h32d64_causal|H32D64CausalB2S128|2|128|32|64|causal|run_fa2_smoke_h32d64_causal_tests|Fa2PrefillFp16SmokeTest.H32D64CausalB2S128"
  "fa2|smoke|h16d128_full|H16D128FullB2S128|2|128|16|128|full|run_fa2_smoke_h16d128_full_tests|Fa2PrefillFp16SmokeTest.H16D128FullB2S128"
  "fa2|smoke|h16d128_causal|H16D128CausalB2S128|2|128|16|128|causal|run_fa2_smoke_h16d128_causal_tests|Fa2PrefillFp16SmokeTest.H16D128CausalB2S128"
  "fa2|small|h32d64_full|H32D64FullB32S256|32|256|32|64|full|run_fa2_small_h32d64_full_tests|Fa2PrefillFp16SmallTest.H32D64FullB32S256"
  "fa2|small|h32d64_causal|H32D64CausalB32S256|32|256|32|64|causal|run_fa2_small_h32d64_causal_tests|Fa2PrefillFp16SmallTest.H32D64CausalB32S256"
  "fa2|small|h16d128_full|H16D128FullB32S256|32|256|16|128|full|run_fa2_small_h16d128_full_tests|Fa2PrefillFp16SmallTest.H16D128FullB32S256"
  "fa2|small|h16d128_causal|H16D128CausalB32S256|32|256|16|128|causal|run_fa2_small_h16d128_causal_tests|Fa2PrefillFp16SmallTest.H16D128CausalB32S256"
  "fa2|medium|h32d64_full|H32D64FullB16S512|16|512|32|64|full|run_fa2_medium_h32d64_full_tests|Fa2PrefillFp16MediumTest.H32D64FullB16S512"
  "fa2|medium|h32d64_causal|H32D64CausalB16S512|16|512|32|64|causal|run_fa2_medium_h32d64_causal_tests|Fa2PrefillFp16MediumTest.H32D64CausalB16S512"
  "fa2|medium|h16d128_full|H16D128FullB16S512|16|512|16|128|full|run_fa2_medium_h16d128_full_tests|Fa2PrefillFp16MediumTest.H16D128FullB16S512"
  "fa2|medium|h16d128_causal|H16D128CausalB16S512|16|512|16|128|causal|run_fa2_medium_h16d128_causal_tests|Fa2PrefillFp16MediumTest.H16D128CausalB16S512"
  "fa2|large|h32d64_full|H32D64FullB64S512|64|512|32|64|full|run_fa2_large_h32d64_full_tests|Fa2PrefillFp16IntegrationTest.H32D64FullB64S512"
  "fa2|large|h32d64_full|H32D64FullB32S1024|32|1024|32|64|full|run_fa2_large_h32d64_full_tests|Fa2PrefillFp16IntegrationTest.H32D64FullB32S1024"
  "fa2|large|h32d64_full|H32D64FullB16S2048|16|2048|32|64|full|run_fa2_large_h32d64_full_tests|Fa2PrefillFp16IntegrationTest.H32D64FullB16S2048"
  "fa2|large|h32d64_full|H32D64FullB8S4096|8|4096|32|64|full|run_fa2_large_h32d64_full_tests|Fa2PrefillFp16IntegrationTest.H32D64FullB8S4096"
  "fa2|large|h32d64_full|H32D64FullB4S8192|4|8192|32|64|full|run_fa2_large_h32d64_full_tests|Fa2PrefillFp16IntegrationTest.H32D64FullB4S8192"
  "fa2|large|h32d64_causal|H32D64CausalB64S512|64|512|32|64|causal|run_fa2_large_h32d64_causal_tests|Fa2PrefillFp16IntegrationTest.H32D64CausalB64S512"
  "fa2|large|h32d64_causal|H32D64CausalB32S1024|32|1024|32|64|causal|run_fa2_large_h32d64_causal_tests|Fa2PrefillFp16IntegrationTest.H32D64CausalB32S1024"
  "fa2|large|h32d64_causal|H32D64CausalB16S2048|16|2048|32|64|causal|run_fa2_large_h32d64_causal_tests|Fa2PrefillFp16IntegrationTest.H32D64CausalB16S2048"
  "fa2|large|h32d64_causal|H32D64CausalB8S4096|8|4096|32|64|causal|run_fa2_large_h32d64_causal_tests|Fa2PrefillFp16IntegrationTest.H32D64CausalB8S4096"
  "fa2|large|h32d64_causal|H32D64CausalB4S8192|4|8192|32|64|causal|run_fa2_large_h32d64_causal_tests|Fa2PrefillFp16IntegrationTest.H32D64CausalB4S8192"
  "fa2|large|h16d128_full|H16D128FullB64S512|64|512|16|128|full|run_fa2_large_h16d128_full_tests|Fa2PrefillFp16IntegrationTest.H16D128FullB64S512"
  "fa2|large|h16d128_full|H16D128FullB32S1024|32|1024|16|128|full|run_fa2_large_h16d128_full_tests|Fa2PrefillFp16IntegrationTest.H16D128FullB32S1024"
  "fa2|large|h16d128_full|H16D128FullB16S2048|16|2048|16|128|full|run_fa2_large_h16d128_full_tests|Fa2PrefillFp16IntegrationTest.H16D128FullB16S2048"
  "fa2|large|h16d128_full|H16D128FullB8S4096|8|4096|16|128|full|run_fa2_large_h16d128_full_tests|Fa2PrefillFp16IntegrationTest.H16D128FullB8S4096"
  "fa2|large|h16d128_full|H16D128FullB4S8192|4|8192|16|128|full|run_fa2_large_h16d128_full_tests|Fa2PrefillFp16IntegrationTest.H16D128FullB4S8192"
  "fa2|large|h16d128_causal|H16D128CausalB64S512|64|512|16|128|causal|run_fa2_large_h16d128_causal_tests|Fa2PrefillFp16IntegrationTest.H16D128CausalB64S512"
  "fa2|large|h16d128_causal|H16D128CausalB32S1024|32|1024|16|128|causal|run_fa2_large_h16d128_causal_tests|Fa2PrefillFp16IntegrationTest.H16D128CausalB32S1024"
  "fa2|large|h16d128_causal|H16D128CausalB16S2048|16|2048|16|128|causal|run_fa2_large_h16d128_causal_tests|Fa2PrefillFp16IntegrationTest.H16D128CausalB16S2048"
  "fa2|large|h16d128_causal|H16D128CausalB8S4096|8|4096|16|128|causal|run_fa2_large_h16d128_causal_tests|Fa2PrefillFp16IntegrationTest.H16D128CausalB8S4096"
  "fa2|large|h16d128_causal|H16D128CausalB4S8192|4|8192|16|128|causal|run_fa2_large_h16d128_causal_tests|Fa2PrefillFp16IntegrationTest.H16D128CausalB4S8192"
  "fa3|smoke|h32d64_full|H32D64FullB2S128|2|128|32|64|full|run_fa3_extended_tests|Fa3PrefillFp16SmokeTest.H32D64FullB2S128"
  "fa3|smoke|h32d64_causal|H32D64CausalB2S128|2|128|32|64|causal|run_fa3_extended_tests|Fa3PrefillFp16SmokeTest.H32D64CausalB2S128"
  "fa3|smoke|h16d128_full|H16D128FullB2S128|2|128|16|128|full|run_fa3_extended_tests|Fa3PrefillFp16SmokeTest.H16D128FullB2S128"
  "fa3|smoke|h16d128_causal|H16D128CausalB2S128|2|128|16|128|causal|run_fa3_extended_tests|Fa3PrefillFp16SmokeTest.H16D128CausalB2S128"
  "fa3|small|h32d64_full|H32D64FullB32S256|32|256|32|64|full|run_fa3_extended_tests|Fa3PrefillFp16SmallTest.H32D64FullB32S256"
  "fa3|small|h32d64_causal|H32D64CausalB32S256|32|256|32|64|causal|run_fa3_extended_tests|Fa3PrefillFp16SmallTest.H32D64CausalB32S256"
  "fa3|small|h16d128_full|H16D128FullB32S256|32|256|16|128|full|run_fa3_extended_tests|Fa3PrefillFp16SmallTest.H16D128FullB32S256"
  "fa3|small|h16d128_causal|H16D128CausalB32S256|32|256|16|128|causal|run_fa3_extended_tests|Fa3PrefillFp16SmallTest.H16D128CausalB32S256"
  "fa3|medium|h32d64_full|H32D64FullB16S512|16|512|32|64|full|run_fa3_extended_tests|Fa3PrefillFp16MediumTest.H32D64FullB16S512"
  "fa3|medium|h32d64_causal|H32D64CausalB16S512|16|512|32|64|causal|run_fa3_extended_tests|Fa3PrefillFp16MediumTest.H32D64CausalB16S512"
  "fa3|medium|h16d128_full|H16D128FullB16S512|16|512|16|128|full|run_fa3_extended_tests|Fa3PrefillFp16MediumTest.H16D128FullB16S512"
  "fa3|medium|h16d128_causal|H16D128CausalB16S512|16|512|16|128|causal|run_fa3_extended_tests|Fa3PrefillFp16MediumTest.H16D128CausalB16S512"
  "fa3|large|h32d64_full|H32D64FullB64S512|64|512|32|64|full|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H32D64FullB64S512"
  "fa3|large|h32d64_full|H32D64FullB32S1024|32|1024|32|64|full|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H32D64FullB32S1024"
  "fa3|large|h32d64_full|H32D64FullB16S2048|16|2048|32|64|full|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H32D64FullB16S2048"
  "fa3|large|h32d64_full|H32D64FullB8S4096|8|4096|32|64|full|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H32D64FullB8S4096"
  "fa3|large|h32d64_full|H32D64FullB4S8192|4|8192|32|64|full|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H32D64FullB4S8192"
  "fa3|large|h32d64_causal|H32D64CausalB64S512|64|512|32|64|causal|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H32D64CausalB64S512"
  "fa3|large|h32d64_causal|H32D64CausalB32S1024|32|1024|32|64|causal|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H32D64CausalB32S1024"
  "fa3|large|h32d64_causal|H32D64CausalB16S2048|16|2048|32|64|causal|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H32D64CausalB16S2048"
  "fa3|large|h32d64_causal|H32D64CausalB8S4096|8|4096|32|64|causal|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H32D64CausalB8S4096"
  "fa3|large|h32d64_causal|H32D64CausalB4S8192|4|8192|32|64|causal|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H32D64CausalB4S8192"
  "fa3|large|h16d128_full|H16D128FullB64S512|64|512|16|128|full|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H16D128FullB64S512"
  "fa3|large|h16d128_full|H16D128FullB32S1024|32|1024|16|128|full|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H16D128FullB32S1024"
  "fa3|large|h16d128_full|H16D128FullB16S2048|16|2048|16|128|full|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H16D128FullB16S2048"
  "fa3|large|h16d128_full|H16D128FullB8S4096|8|4096|16|128|full|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H16D128FullB8S4096"
  "fa3|large|h16d128_full|H16D128FullB4S8192|4|8192|16|128|full|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H16D128FullB4S8192"
  "fa3|large|h16d128_causal|H16D128CausalB64S512|64|512|16|128|causal|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H16D128CausalB64S512"
  "fa3|large|h16d128_causal|H16D128CausalB32S1024|32|1024|16|128|causal|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H16D128CausalB32S1024"
  "fa3|large|h16d128_causal|H16D128CausalB16S2048|16|2048|16|128|causal|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H16D128CausalB16S2048"
  "fa3|large|h16d128_causal|H16D128CausalB8S4096|8|4096|16|128|causal|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H16D128CausalB8S4096"
  "fa3|large|h16d128_causal|H16D128CausalB4S8192|4|8192|16|128|causal|run_fa3_extended_tests|Fa3PrefillFp16IntegrationTest.H16D128CausalB4S8192"
)

selected_token() {
  local selected="$1"
  local token="$2"
  [[ " ${selected} " == *" all "* || " ${selected} " == *" ${token} "* ]]
}

case_selected() {
  local case_id="$1"
  local case_name="$2"
  if [[ -z "${SELECT_CASES}" ]]; then
    return 0
  fi
  [[ " ${SELECT_CASES} " == *" ${case_id} "* ||
     " ${SELECT_CASES} " == *" ${case_name} "* ]]
}

csv_header() {
  echo "case_id,family,group,variant,case,batch,seqlen,heads,head_dim,mode,binary,gtest_filter"
}

print_cases() {
  csv_header
  local row family group variant case_name batch seqlen heads head_dim mode binary filter
  for row in "${CASES[@]}"; do
    IFS='|' read -r family group variant case_name batch seqlen heads head_dim mode binary filter <<<"${row}"
    selected_token "${RUN_FAMILIES}" "${family}" || continue
    selected_token "${RUN_GROUPS}" "${group}" || continue
    case_id="${family}_${case_name}_fwd"
    case_selected "${case_id}" "${case_name}" || continue
    echo "${case_id},${family},${group},${variant},${case_name},${batch},${seqlen},${heads},${head_dim},${mode},${binary},${filter}"
  done
}

if [[ "${PRINT_CASES}" == "1" ]]; then
  print_cases
  exit 0
fi

mkdir -p "${OUT_DIR}/native" "${OUT_DIR}/ncu" "${OUT_DIR}/logs" \
  "${OUT_DIR}/provenance" "${OUT_DIR}/status"
print_cases >"${OUT_DIR}/case_manifest.csv"

{
  echo "root=${ROOT_DIR}"
  echo "test_dir=${TEST_DIR}"
  echo "prebuilt_root=${PREBUILT_ROOT:-}"
  echo "bin_root=${BIN_ROOT}"
  echo "out_dir=${OUT_DIR}"
  echo "run_families=${RUN_FAMILIES}"
  echo "run_groups=${RUN_GROUPS}"
  echo "select_cases=${SELECT_CASES}"
  echo "run_native=${RUN_NATIVE}"
  echo "run_ncu=${RUN_NCU}"
  echo "run_ncu_on_native_fail=${RUN_NCU_ON_NATIVE_FAIL}"
  echo "ncu_set=${NCU_SET}"
  echo "ncu_metrics=${NCU_METRICS}"
  echo "ncu_kernel_name=${NCU_KERNEL_NAME}"
  echo "ncu_extra_args=${NCU_EXTRA_ARGS}"
  echo "ld_library_path=${LD_LIBRARY_PATH:-}"
  echo "date=$(date -Is)"
} | tee "${OUT_DIR}/provenance/run_env.txt"

nvcc --version | tee "${OUT_DIR}/provenance/nvcc_version.txt" || true
ncu --version | tee "${OUT_DIR}/provenance/ncu_version.txt" || true
nvidia-smi | tee "${OUT_DIR}/provenance/nvidia_smi.txt" || true
nvidia-smi --query-gpu=name,driver_version,compute_cap,clocks.sm,clocks.max.sm,clocks.mem,memory.total,pstate,power.limit \
  --format=csv,noheader | tee "${OUT_DIR}/provenance/gpu_query.txt" || true

if command -v cuobjdump >/dev/null 2>&1; then
  shopt -s nullglob
  for bin in "${BIN_ROOT}"/run_fa2_*_tests "${BIN_ROOT}"/run_fa3_extended_tests; do
    base="$(basename "${bin}")"
    {
      echo "== ${base} =="
      cuobjdump --list-elf "${bin}" || true
      cuobjdump --list-ptx "${bin}" || true
    } >"${OUT_DIR}/provenance/${base}.cuobjdump.txt"
  done
  shopt -u nullglob
fi

ncu_profile_args=(--target-processes all)
if [[ -n "${NCU_METRICS}" ]]; then
  ncu_profile_args+=(--metrics "${NCU_METRICS}")
else
  ncu_profile_args+=(--set "${NCU_SET}")
fi
if [[ -n "${NCU_KERNEL_NAME}" ]]; then
  ncu_profile_args+=(--kernel-name "${NCU_KERNEL_NAME}")
fi
if [[ -n "${NCU_EXTRA_ARGS}" ]]; then
  # Intentional simple splitting for command-line style extra flags.
  read -r -a extra_args <<<"${NCU_EXTRA_ARGS}"
  ncu_profile_args+=("${extra_args[@]}")
fi

status_csv="${OUT_DIR}/status/status.csv"
echo "case_id,family,group,variant,case,batch,seqlen,heads,head_dim,mode,native_status,ncu_status,seconds,rep" >"${status_csv}"

overall_status=0
row_index=0
total_rows="${#CASES[@]}"
for row in "${CASES[@]}"; do
  row_index=$((row_index + 1))
  IFS='|' read -r family group variant case_name batch seqlen heads head_dim mode binary filter <<<"${row}"
  selected_token "${RUN_FAMILIES}" "${family}" || continue
  selected_token "${RUN_GROUPS}" "${group}" || continue
  case_id="${family}_${case_name}_fwd"
  case_selected "${case_id}" "${case_name}" || continue

  bin="${BIN_ROOT}/${binary}"
  native_log="${OUT_DIR}/native/${case_id}.log"
  ncu_log="${OUT_DIR}/logs/${case_id}.ncu.log"
  rep="${OUT_DIR}/ncu/${case_id}.ncu-rep"
  raw_csv="${OUT_DIR}/ncu/${case_id}.raw.csv"
  details_csv="${OUT_DIR}/ncu/${case_id}.details.csv"
  done_marker="${OUT_DIR}/status/${case_id}.done"

  if [[ "${RESUME}" == "1" && "${FORCE}" != "1" && -f "${done_marker}" ]]; then
    echo "[${row_index}/${total_rows}] skip done ${case_id}"
    echo "${case_id},${family},${group},${variant},${case_name},${batch},${seqlen},${heads},${head_dim},${mode},skip,skip,0,${rep}" >>"${status_csv}"
    continue
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[${row_index}/${total_rows}] ${case_id}"
    echo "  native: ${bin} --gtest_filter=${filter}"
    echo "  ncu: ncu ${ncu_profile_args[*]} --export ${rep} --force-overwrite ${bin} --gtest_filter=${filter}"
    continue
  fi

  if [[ ! -x "${bin}" ]]; then
    echo "missing executable: ${bin}" | tee "${OUT_DIR}/status/${case_id}.missing"
    echo "${case_id},${family},${group},${variant},${case_name},${batch},${seqlen},${heads},${head_dim},${mode},missing,missing,0,${rep}" >>"${status_csv}"
    overall_status=1
    continue
  fi

  echo "[${row_index}/${total_rows}] ${case_id} bin=${binary} filter=${filter}"
  start_sec="$(date +%s)"
  native_status=0
  ncu_status=0

  if [[ "${RUN_NATIVE}" == "1" ]]; then
    echo "=== native ${case_id} ===" | tee "${native_log}"
    if ! FA2_RUN_32KI=1 "${bin}" --gtest_filter="${filter}" 2>&1 | tee -a "${native_log}"; then
      native_status=1
      overall_status=1
    fi
  fi

  if [[ "${RUN_NCU}" == "1" &&
        ("${native_status}" == "0" || "${RUN_NCU_ON_NATIVE_FAIL}" == "1") ]]; then
    echo "=== ncu ${case_id} ===" | tee "${ncu_log}"
    if ! FA2_RUN_32KI=1 ncu \
      "${ncu_profile_args[@]}" \
      --export "${rep}" \
      --force-overwrite \
      "${bin}" --gtest_filter="${filter}" \
      2>&1 | tee -a "${ncu_log}"; then
      ncu_status=1
      overall_status=1
    fi

    if [[ -f "${rep}" ]]; then
      ncu --import "${rep}" --csv --page raw >"${raw_csv}" 2>>"${ncu_log}" || true
      ncu --import "${rep}" --csv --page details >"${details_csv}" 2>>"${ncu_log}" || true
    fi
  elif [[ "${native_status}" != "0" ]]; then
    ncu_status=1
  fi

  end_sec="$(date +%s)"
  elapsed=$((end_sec - start_sec))
  echo "${case_id},${family},${group},${variant},${case_name},${batch},${seqlen},${heads},${head_dim},${mode},${native_status},${ncu_status},${elapsed},${rep}" >>"${status_csv}"
  if [[ "${native_status}" == "0" && "${ncu_status}" == "0" ]]; then
    touch "${done_marker}"
  fi
done

echo "${OUT_DIR}" | tee "${OUT_DIR}/result_dir.txt"
exit "${overall_status}"
