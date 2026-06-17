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

OUT_DIR="${OUT_DIR:-${TEST_DIR}/run/H100_FA3_SYNC_NOPROFILE_REGTIMELINE_$(date +%Y%m%d_%H%M%S)}"
FA3_CASES="${FA3_CASES:-H1D128B1S4096}"
JOBS="${JOBS:-8}"
NCU_SET="${NCU_SET:-full}"
NCU_METRICS="${NCU_METRICS:-}"

mkdir -p "${OUT_DIR}/clock" "${OUT_DIR}/ncu" "${OUT_DIR}/logs" "${OUT_DIR}/provenance"

{
  echo "root=${ROOT_DIR}"
  echo "cuda=${CUDA_INSTALL_PATH}"
  echo "cases=${FA3_CASES}"
  echo "ncu_set=${NCU_SET}"
  echo "ncu_metrics=${NCU_METRICS}"
  echo "out_dir=${OUT_DIR}"
} | tee "${OUT_DIR}/provenance/run_env.txt"
nvcc --version | tee "${OUT_DIR}/provenance/nvcc_version.txt"
ncu --version | tee "${OUT_DIR}/provenance/ncu_version.txt"
nvidia-smi | tee "${OUT_DIR}/provenance/nvidia_smi.txt"

make -C "${TEST_DIR}" -j"${JOBS}" \
  HOPPER_CUDA_ARCH=sm_90a \
  build/bin/hopper/run_fa3_sensitivity_qk_pv_only_no_tma_noprofile_tests \
  build/bin/hopper/run_fa3_sensitivity_sync_only_no_tma_noprofile_tests \
  build/bin/hopper/run_fa3_sensitivity_qk_pv_only_no_tma_reg_timeline_tests \
  2>&1 | tee "${OUT_DIR}/logs/build.log"

if [[ -n "${NCU_METRICS}" ]]; then
  NCU_PROFILE_ARGS=(--metrics "${NCU_METRICS}")
else
  NCU_PROFILE_ARGS=(--set "${NCU_SET}")
fi

run_plain() {
  local variant="$1"
  local bin="$2"
  local out_csv="${OUT_DIR}/clock/${variant}.csv"
  local run_log="${OUT_DIR}/logs/${variant}.run.log"
  local rep="${OUT_DIR}/ncu/${variant}.ncu-rep"
  local ncu_csv="${OUT_DIR}/ncu/${variant}.csv"
  local ncu_log="${OUT_DIR}/logs/${variant}.ncu.log"

  echo "=== run ${variant} cases=${FA3_CASES} ===" | tee "${run_log}"
  FA3_H1D128_PROFILE_CASE_LIST="${FA3_CASES}" \
  FA3_H1D128_PROFILE_OUT="${out_csv}" \
    "${bin}" 2>&1 | tee -a "${run_log}"

  echo "=== ncu ${variant} cases=${FA3_CASES} ===" | tee "${ncu_log}"
  FA3_H1D128_PROFILE_CASE_LIST="${FA3_CASES}" \
  FA3_H1D128_PROFILE_OUT="${OUT_DIR}/clock/${variant}_ncu.csv" \
    ncu \
      --target-processes all \
      "${NCU_PROFILE_ARGS[@]}" \
      --export "${rep}" \
      --force-overwrite \
      "${bin}" \
        2>&1 | tee -a "${ncu_log}"
  ncu --import "${rep}" --csv --page raw > "${ncu_csv}" 2>>"${ncu_log}" || true
}

run_reg_timeline() {
  local variant="$1"
  local bin="$2"
  local clock_csv="${OUT_DIR}/clock/${variant}.csv"
  local iter_csv="${OUT_DIR}/clock/${variant}_iter.csv"
  local timeline_csv="${OUT_DIR}/clock/${variant}_timeline.csv"
  local reg_timeline_csv="${OUT_DIR}/clock/${variant}_reg_timeline.csv"
  local run_log="${OUT_DIR}/logs/${variant}.run.log"
  local rep="${OUT_DIR}/ncu/${variant}.ncu-rep"
  local ncu_csv="${OUT_DIR}/ncu/${variant}.csv"
  local ncu_log="${OUT_DIR}/logs/${variant}.ncu.log"

  echo "=== reg timeline ${variant} cases=${FA3_CASES} ===" | tee "${run_log}"
  FA3_H1D128_PROFILE_CASE_LIST="${FA3_CASES}" \
  FA3_H1D128_PROFILE_OUT="${clock_csv}" \
  FA3_H1D128_PROFILE_ITER_OUT="${iter_csv}" \
  FA3_H1D128_PROFILE_TIMELINE_OUT="${timeline_csv}" \
  FA3_H1D128_PROFILE_REG_TIMELINE_OUT="${reg_timeline_csv}" \
    "${bin}" 2>&1 | tee -a "${run_log}"

  echo "=== ncu reg timeline ${variant} cases=${FA3_CASES} ===" | tee "${ncu_log}"
  FA3_H1D128_PROFILE_CASE_LIST="${FA3_CASES}" \
  FA3_H1D128_PROFILE_OUT="${OUT_DIR}/clock/${variant}_ncu.csv" \
  FA3_H1D128_PROFILE_ITER_OUT="${OUT_DIR}/clock/${variant}_ncu_iter.csv" \
  FA3_H1D128_PROFILE_TIMELINE_OUT="${OUT_DIR}/clock/${variant}_ncu_timeline.csv" \
  FA3_H1D128_PROFILE_REG_TIMELINE_OUT="${OUT_DIR}/clock/${variant}_ncu_reg_timeline.csv" \
    ncu \
      --target-processes all \
      "${NCU_PROFILE_ARGS[@]}" \
      --export "${rep}" \
      --force-overwrite \
      "${bin}" \
        2>&1 | tee -a "${ncu_log}"
  ncu --import "${rep}" --csv --page raw > "${ncu_csv}" 2>>"${ncu_log}" || true
}

run_plain \
  qk_pv_only_no_tma_noprofile \
  "${TEST_DIR}/build/bin/hopper/run_fa3_sensitivity_qk_pv_only_no_tma_noprofile_tests"
run_plain \
  sync_only_no_tma_noprofile \
  "${TEST_DIR}/build/bin/hopper/run_fa3_sensitivity_sync_only_no_tma_noprofile_tests"
run_reg_timeline \
  qk_pv_only_no_tma_reg_timeline \
  "${TEST_DIR}/build/bin/hopper/run_fa3_sensitivity_qk_pv_only_no_tma_reg_timeline_tests"

echo "${OUT_DIR}" | tee "${OUT_DIR}/result_dir.txt"
