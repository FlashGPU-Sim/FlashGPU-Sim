#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${TEST_DIR}/.." && pwd)"

export CUDA_INSTALL_PATH="${CUDA_INSTALL_PATH:-/usr/local/cuda-12.8}"
export CUDA_HOME="${CUDA_HOME:-${CUDA_INSTALL_PATH}}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_INSTALL_PATH}}"
export PATH="${CUDA_INSTALL_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_INSTALL_PATH}/lib64:${LD_LIBRARY_PATH:-}"

if [[ -z "${PREBUILT_ROOT:-}" && -d "${SCRIPT_DIR}/bin" ]]; then
  PREBUILT_ROOT="${SCRIPT_DIR}"
fi

if [[ -n "${PREBUILT_ROOT:-}" ]]; then
  BIN_ROOT="${PREBUILT_ROOT}/bin"
else
  BIN_ROOT="${TEST_DIR}/build/bin/hopper"
fi
if [[ -n "${PREBUILT_ROOT:-}" && -d "${PREBUILT_ROOT}/lib64" ]]; then
  export LD_LIBRARY_PATH="${PREBUILT_ROOT}/lib64:${LD_LIBRARY_PATH}"
fi

OUT_DIR="${OUT_DIR:-${TEST_DIR}/run/H100_FA2_FULL_NCU_$(date +%Y%m%d_%H%M%S)}"
NCU_SET="${NCU_SET:-full}"
NCU_METRICS="${NCU_METRICS:-}"
NCU_KERNEL_NAME="${NCU_KERNEL_NAME:-regex:.*flash_fwd_kernel.*}"
RUN_NATIVE="${RUN_NATIVE:-1}"
RUN_NCU="${RUN_NCU:-1}"
RESUME="${RESUME:-1}"
FORCE="${FORCE:-0}"
SELECT_CASES="${SELECT_CASES:-}"

CASES=(
  "smoke|h32d64_full|H32D64FullB2S128|2|128|32|64|full|Fa2PrefillFp16SmokeTest.H32D64FullB2S128|run_fa2_smoke_h32d64_full_tests|0"
  "smoke|h16d128_full|H16D128FullB2S128|2|128|16|128|full|Fa2PrefillFp16SmokeTest.H16D128FullB2S128|run_fa2_smoke_h16d128_full_tests|0"
  "smoke|h32d64_causal|H32D64CausalB2S128|2|128|32|64|causal|Fa2PrefillFp16SmokeTest.H32D64CausalB2S128|run_fa2_smoke_h32d64_causal_tests|0"
  "smoke|h16d128_causal|H16D128CausalB2S128|2|128|16|128|causal|Fa2PrefillFp16SmokeTest.H16D128CausalB2S128|run_fa2_smoke_h16d128_causal_tests|0"

  "small|h32d64_full|H32D64FullB32S256|32|256|32|64|full|Fa2PrefillFp16SmallTest.H32D64FullB32S256|run_fa2_small_h32d64_full_tests|0"
  "small|h16d128_full|H16D128FullB32S256|32|256|16|128|full|Fa2PrefillFp16SmallTest.H16D128FullB32S256|run_fa2_small_h16d128_full_tests|0"
  "small|h32d64_causal|H32D64CausalB32S256|32|256|32|64|causal|Fa2PrefillFp16SmallTest.H32D64CausalB32S256|run_fa2_small_h32d64_causal_tests|0"
  "small|h16d128_causal|H16D128CausalB32S256|32|256|16|128|causal|Fa2PrefillFp16SmallTest.H16D128CausalB32S256|run_fa2_small_h16d128_causal_tests|0"

  "medium|h32d64_full|H32D64FullB16S512|16|512|32|64|full|Fa2PrefillFp16MediumTest.H32D64FullB16S512|run_fa2_medium_h32d64_full_tests|0"
  "medium|h16d128_full|H16D128FullB16S512|16|512|16|128|full|Fa2PrefillFp16MediumTest.H16D128FullB16S512|run_fa2_medium_h16d128_full_tests|0"
  "medium|h32d64_causal|H32D64CausalB16S512|16|512|32|64|causal|Fa2PrefillFp16MediumTest.H32D64CausalB16S512|run_fa2_medium_h32d64_causal_tests|0"
  "medium|h16d128_causal|H16D128CausalB16S512|16|512|16|128|causal|Fa2PrefillFp16MediumTest.H16D128CausalB16S512|run_fa2_medium_h16d128_causal_tests|0"

  "large|h32d64_full|H32D64FullB64S512|64|512|32|64|full|Fa2PrefillFp16IntegrationTest.H32D64FullB64S512|run_fa2_large_h32d64_full_tests|1"
  "large|h32d64_full|H32D64FullB32S1024|32|1024|32|64|full|Fa2PrefillFp16IntegrationTest.H32D64FullB32S1024|run_fa2_large_h32d64_full_tests|1"
  "large|h32d64_full|H32D64FullB16S2048|16|2048|32|64|full|Fa2PrefillFp16IntegrationTest.H32D64FullB16S2048|run_fa2_large_h32d64_full_tests|1"
  "large|h32d64_full|H32D64FullB8S4096|8|4096|32|64|full|Fa2PrefillFp16IntegrationTest.H32D64FullB8S4096|run_fa2_large_h32d64_full_tests|1"
  "large|h32d64_full|H32D64FullB4S8192|4|8192|32|64|full|Fa2PrefillFp16IntegrationTest.H32D64FullB4S8192|run_fa2_large_h32d64_full_tests|1"

  "large|h16d128_full|H16D128FullB64S512|64|512|16|128|full|Fa2PrefillFp16IntegrationTest.H16D128FullB64S512|run_fa2_large_h16d128_full_tests|1"
  "large|h16d128_full|H16D128FullB32S1024|32|1024|16|128|full|Fa2PrefillFp16IntegrationTest.H16D128FullB32S1024|run_fa2_large_h16d128_full_tests|1"
  "large|h16d128_full|H16D128FullB16S2048|16|2048|16|128|full|Fa2PrefillFp16IntegrationTest.H16D128FullB16S2048|run_fa2_large_h16d128_full_tests|1"
  "large|h16d128_full|H16D128FullB8S4096|8|4096|16|128|full|Fa2PrefillFp16IntegrationTest.H16D128FullB8S4096|run_fa2_large_h16d128_full_tests|1"
  "large|h16d128_full|H16D128FullB4S8192|4|8192|16|128|full|Fa2PrefillFp16IntegrationTest.H16D128FullB4S8192|run_fa2_large_h16d128_full_tests|1"

  "large|h32d64_causal|H32D64CausalB64S512|64|512|32|64|causal|Fa2PrefillFp16IntegrationTest.H32D64CausalB64S512|run_fa2_large_h32d64_causal_tests|1"
  "large|h32d64_causal|H32D64CausalB32S1024|32|1024|32|64|causal|Fa2PrefillFp16IntegrationTest.H32D64CausalB32S1024|run_fa2_large_h32d64_causal_tests|1"
  "large|h32d64_causal|H32D64CausalB16S2048|16|2048|32|64|causal|Fa2PrefillFp16IntegrationTest.H32D64CausalB16S2048|run_fa2_large_h32d64_causal_tests|1"
  "large|h32d64_causal|H32D64CausalB8S4096|8|4096|32|64|causal|Fa2PrefillFp16IntegrationTest.H32D64CausalB8S4096|run_fa2_large_h32d64_causal_tests|1"
  "large|h32d64_causal|H32D64CausalB4S8192|4|8192|32|64|causal|Fa2PrefillFp16IntegrationTest.H32D64CausalB4S8192|run_fa2_large_h32d64_causal_tests|1"

  "large|h16d128_causal|H16D128CausalB64S512|64|512|16|128|causal|Fa2PrefillFp16IntegrationTest.H16D128CausalB64S512|run_fa2_large_h16d128_causal_tests|1"
  "large|h16d128_causal|H16D128CausalB32S1024|32|1024|16|128|causal|Fa2PrefillFp16IntegrationTest.H16D128CausalB32S1024|run_fa2_large_h16d128_causal_tests|1"
  "large|h16d128_causal|H16D128CausalB16S2048|16|2048|16|128|causal|Fa2PrefillFp16IntegrationTest.H16D128CausalB16S2048|run_fa2_large_h16d128_causal_tests|1"
  "large|h16d128_causal|H16D128CausalB8S4096|8|4096|16|128|causal|Fa2PrefillFp16IntegrationTest.H16D128CausalB8S4096|run_fa2_large_h16d128_causal_tests|1"
  "large|h16d128_causal|H16D128CausalB4S8192|4|8192|16|128|causal|Fa2PrefillFp16IntegrationTest.H16D128CausalB4S8192|run_fa2_large_h16d128_causal_tests|1"
)

csv_header() {
  echo "case_id,group,variant,case,batch,seqlen,heads,head_dim,mode,gtest_filter,binary,opt_in_32ki"
}

print_cases() {
  csv_header
  local row
  for row in "${CASES[@]}"; do
    IFS='|' read -r group variant case_name batch seqlen heads head_dim mode gtest_filter binary opt_in <<<"${row}"
    echo "${case_name}_fwd,${group},${variant},${case_name},${batch},${seqlen},${heads},${head_dim},${mode},${gtest_filter},${binary},${opt_in}"
  done
}

usage() {
  cat <<'EOF'
Usage:
  run_fa2_full_ncu_h100.sh [--print-cases|--dry-run]

Environment:
  PREBUILT_ROOT     Directory containing bin/run_fa2_*_tests. If unset, use repo test/build/bin/hopper.
  CUDA_INSTALL_PATH CUDA root, default /usr/local/cuda-12.8.
  OUT_DIR           Output directory.
  NCU_SET           Nsight Compute section set, default full.
  NCU_METRICS       Optional metrics list. Overrides NCU_SET.
  NCU_KERNEL_NAME   Kernel-name filter, default regex:.*flash_fwd_kernel.*
  SELECT_CASES      Space-separated case ids or case names to run.
  RUN_NATIVE        Run native sanity before ncu, default 1.
  RUN_NCU           Run ncu, default 1.
  RESUME            Skip cases with done marker, default 1.
  FORCE             Rerun even if done, default 0.
EOF
}

case_selected() {
  local case_id="$1"
  local case_name="$2"
  if [[ -z "${SELECT_CASES}" ]]; then
    return 0
  fi
  [[ " ${SELECT_CASES} " == *" ${case_id} "* || " ${SELECT_CASES} " == *" ${case_name} "* ]]
}

if [[ "${1:-}" == "--print-cases" ]]; then
  print_cases
  exit 0
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

mkdir -p "${OUT_DIR}/native" "${OUT_DIR}/ncu" "${OUT_DIR}/logs" \
  "${OUT_DIR}/provenance" "${OUT_DIR}/status"

print_cases >"${OUT_DIR}/case_manifest.csv"

{
  echo "root=${ROOT_DIR}"
  echo "test_dir=${TEST_DIR}"
  echo "prebuilt_root=${PREBUILT_ROOT:-}"
  echo "bin_root=${BIN_ROOT}"
  echo "cuda=${CUDA_INSTALL_PATH}"
  echo "ncu_set=${NCU_SET}"
  echo "ncu_metrics=${NCU_METRICS}"
  echo "ncu_kernel_name=${NCU_KERNEL_NAME}"
  echo "run_native=${RUN_NATIVE}"
  echo "run_ncu=${RUN_NCU}"
  echo "resume=${RESUME}"
  echo "force=${FORCE}"
  echo "select_cases=${SELECT_CASES}"
  echo "out_dir=${OUT_DIR}"
  echo "date=$(date -Is)"
} | tee "${OUT_DIR}/provenance/run_env.txt"

nvcc --version | tee "${OUT_DIR}/provenance/nvcc_version.txt" || true
ncu --version | tee "${OUT_DIR}/provenance/ncu_version.txt" || true
nvidia-smi | tee "${OUT_DIR}/provenance/nvidia_smi.txt" || true
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,memory.total,pstate,power.limit \
  --format=csv,noheader | tee "${OUT_DIR}/provenance/gpu_query.txt" || true

if [[ -n "${NCU_METRICS}" ]]; then
  NCU_PROFILE_ARGS=(--metrics "${NCU_METRICS}")
else
  NCU_PROFILE_ARGS=(--set "${NCU_SET}")
fi
if [[ -n "${NCU_KERNEL_NAME}" ]]; then
  NCU_PROFILE_ARGS+=(--kernel-name "${NCU_KERNEL_NAME}")
fi

status_csv="${OUT_DIR}/status/status.csv"
echo "case_id,group,variant,case,batch,seqlen,heads,head_dim,mode,native_status,ncu_status,seconds,rep" >"${status_csv}"

overall_status=0
row_index=0
total_cases="${#CASES[@]}"
for row in "${CASES[@]}"; do
  row_index=$((row_index + 1))
  IFS='|' read -r group variant case_name batch seqlen heads head_dim mode gtest_filter binary opt_in <<<"${row}"
  case_id="${case_name}_fwd"
  bin="${BIN_ROOT}/${binary}"
  native_log="${OUT_DIR}/native/${case_id}.log"
  ncu_log="${OUT_DIR}/logs/${case_id}.ncu.log"
  rep="${OUT_DIR}/ncu/${case_id}.ncu-rep"
  raw_csv="${OUT_DIR}/ncu/${case_id}.raw.csv"
  details_csv="${OUT_DIR}/ncu/${case_id}.details.csv"
  done_marker="${OUT_DIR}/status/${case_id}.done"

  if ! case_selected "${case_id}" "${case_name}"; then
    continue
  fi

  if [[ "${RESUME}" == "1" && "${FORCE}" != "1" && -f "${done_marker}" ]]; then
    echo "[${row_index}/${total_cases}] skip done ${case_id}"
    echo "${case_id},${group},${variant},${case_name},${batch},${seqlen},${heads},${head_dim},${mode},skip,skip,0,${rep}" >>"${status_csv}"
    continue
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[${row_index}/${total_cases}] ${case_id} bin=${binary} filter=${gtest_filter}"
    echo "  native: FA2_RUN_32KI=1 ${bin} --gtest_filter=${gtest_filter}"
    echo "  ncu: FA2_RUN_32KI=1 ncu --target-processes all ${NCU_PROFILE_ARGS[*]} --export ${rep} --force-overwrite ${bin} --gtest_filter=${gtest_filter}"
    continue
  fi

  if [[ ! -x "${bin}" ]]; then
    echo "missing executable: ${bin}" | tee "${OUT_DIR}/status/${case_id}.missing"
    echo "${case_id},${group},${variant},${case_name},${batch},${seqlen},${heads},${head_dim},${mode},missing,missing,0,${rep}" >>"${status_csv}"
    overall_status=1
    continue
  fi

  echo "[${row_index}/${total_cases}] ${case_id} bin=${binary} filter=${gtest_filter}"

  start_sec="$(date +%s)"
  native_status=0
  ncu_status=0

  if [[ "${RUN_NATIVE}" == "1" ]]; then
    echo "=== native ${case_id} ===" | tee "${native_log}"
    if ! FA2_RUN_32KI=1 "${bin}" --gtest_filter="${gtest_filter}" 2>&1 | tee -a "${native_log}"; then
      native_status=1
      overall_status=1
    fi
  fi

  if [[ "${RUN_NCU}" == "1" && "${native_status}" == "0" ]]; then
    echo "=== ncu ${case_id} ===" | tee "${ncu_log}"
    if ! FA2_RUN_32KI=1 ncu \
      --target-processes all \
      "${NCU_PROFILE_ARGS[@]}" \
      --export "${rep}" \
      --force-overwrite \
      "${bin}" --gtest_filter="${gtest_filter}" \
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
  echo "${case_id},${group},${variant},${case_name},${batch},${seqlen},${heads},${head_dim},${mode},${native_status},${ncu_status},${elapsed},${rep}" >>"${status_csv}"
  if [[ "${native_status}" == "0" && "${ncu_status}" == "0" ]]; then
    touch "${done_marker}"
  fi
done

echo "${OUT_DIR}" | tee "${OUT_DIR}/result_dir.txt"
exit "${overall_status}"
