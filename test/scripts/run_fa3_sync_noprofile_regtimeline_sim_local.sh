#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/wzr/flashgpu-sim-blackwell/flashgpu-sim}"
OUT="${OUT:-${ROOT}/test/run/FA3_SYNC_NOPROFILE_REGTIMELINE_SIM_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-SM90_H100_1500MHZ_HBM80_L2S160_MSHR512_L2NOC1700}"
CASE="${FA3_CASES:-H1D128FullB1S4096}"
CONFIG_SRC="${ROOT}/configs/${CONFIG}"

export CUDA_INSTALL_PATH="${CUDA_INSTALL_PATH:-/usr/local/cuda-12.8}"
export CUDA_VERSION_NUMBER=12080
export GPGPUSIM_ROOT="${ROOT}"
export GPGPUSIM_CONFIG="${GPGPUSIM_CONFIG:-gcc-13.3.0/cuda-12080/release}"
export GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN=1
export GPGPUSIM_POWER_MODEL="${ROOT}/src/accelwattch/"
export PTXAS_CUDA_INSTALL_PATH="${CUDA_INSTALL_PATH}"
export PTX_SIM_USE_PTX_FILE=1.ptx
export PTX_SIM_KERNELFILE=_1.ptx
export CUOBJDUMP_SIM_FILE=jj
export QTINC=/usr/include
export PATH="${ROOT}/bin:${CUDA_INSTALL_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROOT}/lib/${GPGPUSIM_CONFIG}:${CUDA_INSTALL_PATH}/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "${OUT}"

run_one() {
  local variant="$1"
  local bin="$2"
  local dir="${OUT}/${variant}"
  mkdir -p "${dir}"
  cp "${CONFIG_SRC}/gpgpusim.config" "${dir}/"
  cp "${CONFIG_SRC}/config_ampere_islip.icnt" "${dir}/" 2>/dev/null || true
  {
    echo "variant=${variant}"
    echo "case=${CASE}"
    echo "bin=${ROOT}/test/build/bin/hopper/${bin}"
    echo "config=${CONFIG}"
    echo "cuda=${CUDA_INSTALL_PATH}"
    "${CUDA_INSTALL_PATH}/bin/nvcc" --version
  } >"${dir}/run_env.txt"
  (
    cd "${dir}"
    FA3_H1D128_PROFILE_CASE_LIST="${CASE}" \
    FA3_H1D128_PROFILE_OUT=profile.csv \
    FA3_H1D128_PROFILE_ITER_OUT=iter.csv \
    FA3_H1D128_PROFILE_TIMELINE_OUT=timeline.csv \
    FA3_H1D128_PROFILE_REG_TIMELINE_OUT=reg_timeline.csv \
      "${ROOT}/test/build/bin/hopper/${bin}" \
      --gtest_filter=Fa3H1D128ProfileTest.SelectedD128FullCases \
      >run.log 2>&1
  )
  rg "gpu_sim_cycle =|gpu_tot_sim_cycle|PASSED|FAILED|ERROR|Assertion" \
    "${dir}/run.log" >"${dir}/sanity.txt" || true
}

run_one qk_pv_only_no_tma_noprofile \
  run_fa3_qk_pv_only_no_tma_noprofile_tests
run_one sync_only_no_tma_noprofile \
  run_fa3_sync_only_no_tma_noprofile_tests
run_one qk_pv_only_no_tma_reg_timeline \
  run_fa3_qk_pv_only_no_tma_reg_timeline_tests

echo "${OUT}" >"${OUT}/result_dir.txt"
cat "${OUT}"/*/sanity.txt >"${OUT}/summary.txt" 2>/dev/null || true
