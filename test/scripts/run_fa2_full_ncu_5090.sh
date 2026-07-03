#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${TEST_DIR}/.." && pwd)"

export CUDA_INSTALL_PATH="${CUDA_INSTALL_PATH:-/usr/local/cuda}"
export CUDA_HOME="${CUDA_HOME:-${CUDA_INSTALL_PATH}}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_INSTALL_PATH}}"
export PATH="${CUDA_INSTALL_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_INSTALL_PATH}/lib64:${LD_LIBRARY_PATH:-}"

if [[ -z "${PREBUILT_ROOT:-}" && -d "${SCRIPT_DIR}/bin" ]]; then
  PREBUILT_ROOT="${SCRIPT_DIR}"
fi

if [[ -n "${PREBUILT_ROOT:-}" ]]; then
  BIN_ROOT="${BIN_ROOT:-${PREBUILT_ROOT}/bin}"
else
  BIN_ROOT="${BIN_ROOT:-${TEST_DIR}/build_sm120a/bin/hopper}"
fi
if [[ -n "${PREBUILT_ROOT:-}" && -d "${PREBUILT_ROOT}/lib64" ]]; then
  export LD_LIBRARY_PATH="${PREBUILT_ROOT}/lib64:${LD_LIBRARY_PATH}"
fi

OUT_DIR="${OUT_DIR:-${TEST_DIR}/run/RTX5090_FA2_NCU_$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"

NCU_SET="${NCU_SET:-full}"
NCU_METRICS="${NCU_METRICS:-}"
NCU_KERNEL_NAME="${NCU_KERNEL_NAME:-regex:.*flash_fwd_kernel.*}"
NCU_CLOCK_CONTROL="${NCU_CLOCK_CONTROL:-none}"
NCU_TARGET_PROCESSES="${NCU_TARGET_PROCESSES:-all}"
RUN_NATIVE="${RUN_NATIVE:-1}"
RUN_NCU="${RUN_NCU:-1}"
RESUME="${RESUME:-1}"
FORCE="${FORCE:-0}"
SELECT_CASES="${SELECT_CASES:-}"

LOCK_GPU_CLOCKS="${LOCK_GPU_CLOCKS:-1}"
RESTORE_GPU_CLOCKS="${RESTORE_GPU_CLOCKS:-1}"
RESTORE_GPU_PM="${RESTORE_GPU_PM:-1}"
SUDO_GPU_CLOCK="${SUDO_GPU_CLOCK:-1}"
GPU_CLOCK_BIN="${GPU_CLOCK_BIN:-gpu-clock}"
GPU_CLOCK_CORE="${GPU_CLOCK_CORE:-1837}"
GPU_CLOCK_MEM="${GPU_CLOCK_MEM:-14001}"
GPU_CLOCK_PM="${GPU_CLOCK_PM:-1}"

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
  run_fa2_full_ncu_5090.sh [--print-cases|--dry-run]

Environment:
  PREBUILT_ROOT        Directory containing bin/run_fa2_*_tests.
  BIN_ROOT             Directory containing run_fa2_*_tests. Default test/build_sm120a/bin/hopper.
  CUDA_INSTALL_PATH    CUDA root, default /usr/local/cuda.
  OUT_DIR              Output directory.
  GPU_ID               Physical GPU index for nvidia-smi/gpu-clock, default 0.
  CUDA_VISIBLE_DEVICES CUDA device mask for the test binary, default GPU_ID.
  NCU_SET              Nsight Compute section set, default full.
  NCU_METRICS          Optional metrics list. Overrides NCU_SET.
  NCU_KERNEL_NAME      Kernel-name filter, default regex:.*flash_fwd_kernel.*
  NCU_CLOCK_CONTROL    Nsight Compute clock control, default none.
  SELECT_CASES         Space-separated case ids or case names to run.
  RUN_NATIVE           Run native sanity before ncu, default 1.
  RUN_NCU              Run ncu, default 1.
  RESUME               Skip cases with done marker, default 1.
  FORCE                Rerun even if done, default 0.

Clock control:
  LOCK_GPU_CLOCKS      Use sudo gpu-clock before running, default 1.
  GPU_CLOCK_CORE       Graphics clock passed to gpu-clock -lgc, default 1837.
  GPU_CLOCK_MEM        Memory clock passed to gpu-clock -lmc, default 14001.
  GPU_CLOCK_PM         Persistence mode passed to gpu-clock -pm, default 1.
  RESTORE_GPU_CLOCKS   Reset graphics/memory clocks on exit, default 1.
  RESTORE_GPU_PM       Restore initial persistence mode on exit, default 1.
  SUDO_GPU_CLOCK       Run gpu-clock through sudo -n, default 1.
  GPU_CLOCK_BIN        gpu-clock command name/path, default gpu-clock.

Example:
  OUT_DIR=test/run/RTX5090_FA2_NCU_SMALL_D128_$(date +%Y%m%d_%H%M%S) \
  SELECT_CASES=H16D128FullB32S256_fwd \
  ./test/scripts/run_fa2_full_ncu_5090.sh
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

gpu_clock() {
  if [[ "${SUDO_GPU_CLOCK}" == "1" ]]; then
    sudo -n "${GPU_CLOCK_BIN}" -i "${GPU_ID}" "$@"
  else
    "${GPU_CLOCK_BIN}" -i "${GPU_ID}" "$@"
  fi
}

query_gpu_state() {
  local out_file="$1"
  nvidia-smi -i "${GPU_ID}" \
    --query-gpu=name,driver_version,persistence_mode,pstate,clocks.gr,clocks.sm,clocks.mem,clocks.max.gr,clocks.max.mem,power.draw,power.limit,temperature.gpu \
    --format=csv >"${out_file}" 2>&1 || true
}

initial_pm=""
clock_locked=0
pm_touched=0

cleanup() {
  local rc=$?
  trap - EXIT INT TERM

  if [[ "${clock_locked}" == "1" && "${RESTORE_GPU_CLOCKS}" == "1" ]]; then
    {
      echo "=== reset gpu clocks $(date -Is) ==="
      gpu_clock -rgc -rmc
    } >>"${OUT_DIR}/logs/gpu_clock.log" 2>&1 || true
  fi

  if [[ "${pm_touched}" == "1" && "${RESTORE_GPU_PM}" == "1" ]]; then
    local pm_restore="0"
    if [[ "${initial_pm}" == "Enabled" ]]; then
      pm_restore="1"
    fi
    {
      echo "=== restore persistence mode ${pm_restore} $(date -Is) ==="
      gpu_clock -pm "${pm_restore}"
    } >>"${OUT_DIR}/logs/gpu_clock.log" 2>&1 || true
  fi

  if [[ -d "${OUT_DIR}/provenance" ]]; then
    query_gpu_state "${OUT_DIR}/provenance/gpu_query_after_cleanup.txt"
  fi
  exit "${rc}"
}

lock_gpu_clocks() {
  command -v "${GPU_CLOCK_BIN}" >/dev/null
  initial_pm="$(nvidia-smi -i "${GPU_ID}" --query-gpu=persistence_mode --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]' || true)"

  {
    echo "=== lock gpu clocks $(date -Is) ==="
    echo "gpu_id=${GPU_ID}"
    echo "core=${GPU_CLOCK_CORE}"
    echo "mem=${GPU_CLOCK_MEM}"
    echo "pm=${GPU_CLOCK_PM}"
    echo "initial_persistence_mode=${initial_pm}"
  } | tee -a "${OUT_DIR}/logs/gpu_clock.log"

  if [[ -n "${GPU_CLOCK_PM}" ]]; then
    gpu_clock -pm "${GPU_CLOCK_PM}" 2>&1 | tee -a "${OUT_DIR}/logs/gpu_clock.log"
    pm_touched=1
  fi

  gpu_clock -lgc "${GPU_CLOCK_CORE}" -lmc "${GPU_CLOCK_MEM}" 2>&1 | tee -a "${OUT_DIR}/logs/gpu_clock.log"
  clock_locked=1
  query_gpu_state "${OUT_DIR}/provenance/gpu_query_after_lock.txt"
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
  echo "gpu_id=${GPU_ID}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "ncu_set=${NCU_SET}"
  echo "ncu_metrics=${NCU_METRICS}"
  echo "ncu_kernel_name=${NCU_KERNEL_NAME}"
  echo "ncu_clock_control=${NCU_CLOCK_CONTROL}"
  echo "ncu_target_processes=${NCU_TARGET_PROCESSES}"
  echo "run_native=${RUN_NATIVE}"
  echo "run_ncu=${RUN_NCU}"
  echo "resume=${RESUME}"
  echo "force=${FORCE}"
  echo "select_cases=${SELECT_CASES}"
  echo "lock_gpu_clocks=${LOCK_GPU_CLOCKS}"
  echo "gpu_clock_bin=${GPU_CLOCK_BIN}"
  echo "gpu_clock_core=${GPU_CLOCK_CORE}"
  echo "gpu_clock_mem=${GPU_CLOCK_MEM}"
  echo "gpu_clock_pm=${GPU_CLOCK_PM}"
  echo "restore_gpu_clocks=${RESTORE_GPU_CLOCKS}"
  echo "restore_gpu_pm=${RESTORE_GPU_PM}"
  echo "sudo_gpu_clock=${SUDO_GPU_CLOCK}"
  echo "out_dir=${OUT_DIR}"
  echo "date=$(date -Is)"
} | tee "${OUT_DIR}/provenance/run_env.txt"

nvcc --version | tee "${OUT_DIR}/provenance/nvcc_version.txt" || true
ncu --version | tee "${OUT_DIR}/provenance/ncu_version.txt" || true
nvidia-smi | tee "${OUT_DIR}/provenance/nvidia_smi.txt" || true
query_gpu_state "${OUT_DIR}/provenance/gpu_query_before_lock.txt"
nvidia-smi -i "${GPU_ID}" --query-supported-clocks=mem,gr --format=csv \
  >"${OUT_DIR}/provenance/gpu_supported_clocks.txt" 2>&1 || true

if [[ -n "${NCU_METRICS}" ]]; then
  NCU_PROFILE_ARGS=(--metrics "${NCU_METRICS}")
else
  NCU_PROFILE_ARGS=(--set "${NCU_SET}")
fi
if [[ -n "${NCU_KERNEL_NAME}" ]]; then
  NCU_PROFILE_ARGS+=(--kernel-name "${NCU_KERNEL_NAME}")
fi
if [[ -n "${NCU_CLOCK_CONTROL}" ]]; then
  NCU_PROFILE_ARGS+=(--clock-control "${NCU_CLOCK_CONTROL}")
fi

if [[ "${DRY_RUN}" != "1" && "${LOCK_GPU_CLOCKS}" == "1" ]]; then
  trap cleanup EXIT INT TERM
  lock_gpu_clocks
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
    echo "  native: FA2_RUN_32KI=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${bin} --gtest_filter=${gtest_filter}"
    echo "  ncu: FA2_RUN_32KI=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ncu --target-processes ${NCU_TARGET_PROCESSES} ${NCU_PROFILE_ARGS[*]} --export ${rep} --force-overwrite ${bin} --gtest_filter=${gtest_filter}"
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
      --target-processes "${NCU_TARGET_PROCESSES}" \
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
