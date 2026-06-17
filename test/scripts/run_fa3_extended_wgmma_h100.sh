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

OUT_DIR="${OUT_DIR:-${TEST_DIR}/run/H100_FA3_EXT_WGMMA_$(date +%Y%m%d_%H%M%S)}"
FA3_CASES="${FA3_CASES:-H1D128B1S4096}"
JOBS="${JOBS:-8}"
NCU_SET="${NCU_SET:-full}"
NCU_METRICS="${NCU_METRICS:-}"
WGMMA_N16_CHAIN_CASES="${WGMMA_N16_CHAIN_CASES:-chain11_k1 chain11_k2 chain11_k4 chain11_k8}"
WGMMA_N16_CHAIN_ROUNDS="${WGMMA_N16_CHAIN_ROUNDS:-16}"
WGMMA_N16_CHAIN_BLOCKS="${WGMMA_N16_CHAIN_BLOCKS:-}"

mkdir -p "${OUT_DIR}/clock" "${OUT_DIR}/ncu" "${OUT_DIR}/logs" "${OUT_DIR}/provenance"

{
  echo "root=${ROOT_DIR}"
  echo "cuda=${CUDA_INSTALL_PATH}"
  echo "fa3_cases=${FA3_CASES}"
  echo "wgmma_n16_chain_cases=${WGMMA_N16_CHAIN_CASES}"
  echo "wgmma_n16_chain_rounds=${WGMMA_N16_CHAIN_ROUNDS}"
  echo "wgmma_n16_chain_blocks=${WGMMA_N16_CHAIN_BLOCKS}"
  echo "ncu_set=${NCU_SET}"
  echo "ncu_metrics=${NCU_METRICS}"
  echo "out_dir=${OUT_DIR}"
} | tee "${OUT_DIR}/provenance/run_env.txt"
nvcc --version | tee "${OUT_DIR}/provenance/nvcc_version.txt"
ncu --version | tee "${OUT_DIR}/provenance/ncu_version.txt"
nvidia-smi | tee "${OUT_DIR}/provenance/nvidia_smi.txt"

make -C "${TEST_DIR}" -j"${JOBS}" \
  HOPPER_CUDA_ARCH=sm_90a \
  CUDA_ARCH=sm_90a \
  build/bin/hopper/run_fa3_sensitivity_qk_pv_only_no_tma_extended_tests \
  build/bin/wgmma/wgmma_n16_chain_bench \
  2>&1 | tee "${OUT_DIR}/logs/build.log"

if [[ -n "${NCU_METRICS}" ]]; then
  NCU_PROFILE_ARGS=(--metrics "${NCU_METRICS}")
else
  NCU_PROFILE_ARGS=(--set "${NCU_SET}")
fi

run_fa3_extended() {
  local variant="qk_pv_only_no_tma_extended"
  local bin="${TEST_DIR}/build/bin/hopper/run_fa3_sensitivity_qk_pv_only_no_tma_extended_tests"
  local clock_csv="${OUT_DIR}/clock/${variant}.csv"
  local iter_csv="${OUT_DIR}/clock/${variant}_iter.csv"
  local timeline_csv="${OUT_DIR}/clock/${variant}_timeline.csv"
  local reg_timeline_csv="${OUT_DIR}/clock/${variant}_reg_timeline.csv"
  local run_log="${OUT_DIR}/logs/${variant}.run.log"
  local rep="${OUT_DIR}/ncu/${variant}.ncu-rep"
  local ncu_csv="${OUT_DIR}/ncu/${variant}.csv"
  local ncu_log="${OUT_DIR}/logs/${variant}.ncu.log"

  echo "=== clock64 ${variant} cases=${FA3_CASES} ===" | tee "${run_log}"
  FA3_H1D128_PROFILE_CASE_LIST="${FA3_CASES}" \
  FA3_H1D128_PROFILE_OUT="${clock_csv}" \
  FA3_H1D128_PROFILE_ITER_OUT="${iter_csv}" \
  FA3_H1D128_PROFILE_TIMELINE_OUT="${timeline_csv}" \
  FA3_H1D128_PROFILE_REG_TIMELINE_OUT="${reg_timeline_csv}" \
    "${bin}" --gtest_filter=Fa3H1D128ProfileTest.SelectedD128FullCases \
      2>&1 | tee -a "${run_log}"

  echo "=== ncu ${variant} cases=${FA3_CASES} ===" | tee "${ncu_log}"
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
      "${bin}" --gtest_filter=Fa3H1D128ProfileTest.SelectedD128FullCases \
        2>&1 | tee -a "${ncu_log}"
  ncu --import "${rep}" --csv --page raw > "${ncu_csv}" 2>>"${ncu_log}" || true
}

run_n16_chain_case() {
  local selected="$1"
  local variant="wgmma_n16_${selected}"
  local bin="${TEST_DIR}/build/bin/wgmma/wgmma_n16_chain_bench"
  local prefix="${OUT_DIR}/clock/${variant}"
  local run_log="${OUT_DIR}/logs/${variant}.run.log"
  local rep="${OUT_DIR}/ncu/${variant}.ncu-rep"
  local ncu_csv="${OUT_DIR}/ncu/${variant}.csv"
  local ncu_log="${OUT_DIR}/logs/${variant}.ncu.log"
  local blocks_env=()
  if [[ -n "${WGMMA_N16_CHAIN_BLOCKS}" ]]; then
    blocks_env=(WGMMA_N16_CHAIN_BLOCKS="${WGMMA_N16_CHAIN_BLOCKS}")
  fi

  echo "=== clock64 ${variant} ===" | tee "${run_log}"
  env "${blocks_env[@]}" \
    WGMMA_N16_CHAIN_SELECTED="${selected}" \
    WGMMA_N16_CHAIN_ROUNDS="${WGMMA_N16_CHAIN_ROUNDS}" \
    WGMMA_N16_CHAIN_OUT_PREFIX="${prefix}" \
    "${bin}" --gtest_filter=WgmmaN16ChainBench.Selected \
      2>&1 | tee -a "${run_log}"

  echo "=== ncu ${variant} ===" | tee "${ncu_log}"
  env "${blocks_env[@]}" \
    WGMMA_N16_CHAIN_SELECTED="${selected}" \
    WGMMA_N16_CHAIN_ROUNDS="${WGMMA_N16_CHAIN_ROUNDS}" \
    WGMMA_N16_CHAIN_OUT_PREFIX="${prefix}_ncu" \
    ncu \
      --target-processes all \
      "${NCU_PROFILE_ARGS[@]}" \
      --export "${rep}" \
      --force-overwrite \
      "${bin}" --gtest_filter=WgmmaN16ChainBench.Selected \
        2>&1 | tee -a "${ncu_log}"
  ncu --import "${rep}" --csv --page raw > "${ncu_csv}" 2>>"${ncu_log}" || true
}

run_fa3_extended
for selected in ${WGMMA_N16_CHAIN_CASES}; do
  run_n16_chain_case "${selected}"
done

echo "${OUT_DIR}" | tee "${OUT_DIR}/result_dir.txt"
