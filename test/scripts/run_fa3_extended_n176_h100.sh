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

OUT_DIR="${OUT_DIR:-${TEST_DIR}/run/H100_FA3_N176_EXTENDED_$(date +%Y%m%d_%H%M%S)}"
FA3_CASES="${FA3_CASES:-H1D128FullB1S4096}"
JOBS="${JOBS:-8}"
NCU_SET="${NCU_SET:-full}"
NCU_METRICS="${NCU_METRICS:-}"

mkdir -p "${OUT_DIR}/clock" "${OUT_DIR}/ncu" "${OUT_DIR}/logs" \
  "${OUT_DIR}/provenance" "${OUT_DIR}/sass"

PROFILE_BIN="${TEST_DIR}/build/bin/hopper/run_fa3_qk_pv_only_no_tma_extended_tests"
NOPROFILE_BIN="${TEST_DIR}/build/bin/hopper/run_fa3_qk_pv_only_no_tma_extended_noprofile_tests"
GTEST_FILTER="Fa3H1D128ProfileTest.SelectedD128FullCases"

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
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,memory.total \
  --format=csv,noheader | tee "${OUT_DIR}/provenance/gpu_query.txt"

make -C "${TEST_DIR}" -j"${JOBS}" \
  HOPPER_CUDA_ARCH=sm_90a \
  CUDA_ARCH=sm_90a \
  build/bin/hopper/run_fa3_qk_pv_only_no_tma_extended_tests \
  build/bin/hopper/run_fa3_qk_pv_only_no_tma_extended_noprofile_tests \
  2>&1 | tee "${OUT_DIR}/logs/build.log"

dump_sass() {
  local label="$1"
  local bin="$2"
  local sass="${OUT_DIR}/sass/${label}.sass"
  local summary="${OUT_DIR}/sass/${label}.hgmma_summary.txt"
  cuobjdump --dump-sass "${bin}" > "${sass}" 2>"${OUT_DIR}/logs/${label}.cuobjdump.log" || true
  {
    echo "binary=${bin}"
    echo "hgmma_count=$(grep -c 'HGMMA' "${sass}" || true)"
    grep 'HGMMA' "${sass}" | sed -E 's/.*(HGMMA[^;]*).*/\1/' | sort | uniq -c
  } | tee "${summary}"
}

dump_sass "extended_clock64" "${PROFILE_BIN}"
dump_sass "extended_noprofile" "${NOPROFILE_BIN}"

if [[ -n "${NCU_METRICS}" ]]; then
  NCU_PROFILE_ARGS=(--metrics "${NCU_METRICS}")
else
  NCU_PROFILE_ARGS=(--set "${NCU_SET}")
fi

echo "=== no-profile run cases=${FA3_CASES} ===" | tee "${OUT_DIR}/logs/extended_noprofile.run.log"
FA3_H1D128_PROFILE_CASE_LIST="${FA3_CASES}" \
FA3_H1D128_PROFILE_OUT="${OUT_DIR}/clock/extended_noprofile.csv" \
  "${NOPROFILE_BIN}" --gtest_filter="${GTEST_FILTER}" \
    2>&1 | tee -a "${OUT_DIR}/logs/extended_noprofile.run.log"

echo "=== clock64 run cases=${FA3_CASES} ===" | tee "${OUT_DIR}/logs/extended_clock64.run.log"
FA3_H1D128_PROFILE_CASE_LIST="${FA3_CASES}" \
FA3_H1D128_PROFILE_OUT="${OUT_DIR}/clock/extended_clock64.csv" \
FA3_H1D128_PROFILE_ITER_OUT="${OUT_DIR}/clock/extended_clock64_iter.csv" \
FA3_H1D128_PROFILE_TIMELINE_OUT="${OUT_DIR}/clock/extended_clock64_timeline.csv" \
FA3_H1D128_PROFILE_REG_TIMELINE_OUT="${OUT_DIR}/clock/extended_clock64_reg_timeline.csv" \
  "${PROFILE_BIN}" --gtest_filter="${GTEST_FILTER}" \
    2>&1 | tee -a "${OUT_DIR}/logs/extended_clock64.run.log"

echo "=== ncu no-profile cases=${FA3_CASES} ===" | tee "${OUT_DIR}/logs/extended_noprofile.ncu.log"
FA3_H1D128_PROFILE_CASE_LIST="${FA3_CASES}" \
FA3_H1D128_PROFILE_OUT="${OUT_DIR}/clock/extended_noprofile_ncu.csv" \
  ncu \
    --target-processes all \
    "${NCU_PROFILE_ARGS[@]}" \
    --export "${OUT_DIR}/ncu/extended_noprofile.ncu-rep" \
    --force-overwrite \
    "${NOPROFILE_BIN}" --gtest_filter="${GTEST_FILTER}" \
      2>&1 | tee -a "${OUT_DIR}/logs/extended_noprofile.ncu.log"
ncu --import "${OUT_DIR}/ncu/extended_noprofile.ncu-rep" --csv --page raw \
  > "${OUT_DIR}/ncu/extended_noprofile.csv" \
  2>>"${OUT_DIR}/logs/extended_noprofile.ncu.log" || true

echo "${OUT_DIR}" | tee "${OUT_DIR}/result_dir.txt"
