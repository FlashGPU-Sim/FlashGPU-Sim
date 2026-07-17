#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASE="${FA3_CASES:-H1D128FullB1S4096}"
NCU="${NCU:-/usr/local/cuda-13.0/bin/ncu}"

mkdir -p "${ROOT}/clock" "${ROOT}/ncu" "${ROOT}/logs" "${ROOT}/provenance"

export LD_LIBRARY_PATH="${ROOT}/lib64:/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"

run_plain() {
  local variant="$1"
  local bin="${ROOT}/bin/$2"

  FA3_H1D128_PROFILE_CASE_LIST="${CASE}" \
  FA3_H1D128_PROFILE_OUT="${ROOT}/clock/${variant}.csv" \
    "${bin}" >"${ROOT}/logs/${variant}.run.log" 2>&1

  FA3_H1D128_PROFILE_CASE_LIST="${CASE}" \
  FA3_H1D128_PROFILE_OUT="${ROOT}/clock/${variant}_ncu.csv" \
    "${NCU}" \
      --target-processes all \
      --set full \
      --export "${ROOT}/ncu/${variant}.ncu-rep" \
      --force-overwrite \
      "${bin}" >"${ROOT}/logs/${variant}.ncu.log" 2>&1

  "${NCU}" --import "${ROOT}/ncu/${variant}.ncu-rep" --csv --page raw \
    >"${ROOT}/ncu/${variant}.csv" 2>>"${ROOT}/logs/${variant}.ncu.log" || true
}

run_reg_timeline() {
  local variant="$1"
  local bin="${ROOT}/bin/$2"

  FA3_H1D128_PROFILE_CASE_LIST="${CASE}" \
  FA3_H1D128_PROFILE_OUT="${ROOT}/clock/${variant}.csv" \
  FA3_H1D128_PROFILE_ITER_OUT="${ROOT}/clock/${variant}_iter.csv" \
  FA3_H1D128_PROFILE_TIMELINE_OUT="${ROOT}/clock/${variant}_timeline.csv" \
  FA3_H1D128_PROFILE_REG_TIMELINE_OUT="${ROOT}/clock/${variant}_reg_timeline.csv" \
    "${bin}" >"${ROOT}/logs/${variant}.run.log" 2>&1

  FA3_H1D128_PROFILE_CASE_LIST="${CASE}" \
  FA3_H1D128_PROFILE_OUT="${ROOT}/clock/${variant}_ncu.csv" \
  FA3_H1D128_PROFILE_ITER_OUT="${ROOT}/clock/${variant}_ncu_iter.csv" \
  FA3_H1D128_PROFILE_TIMELINE_OUT="${ROOT}/clock/${variant}_ncu_timeline.csv" \
  FA3_H1D128_PROFILE_REG_TIMELINE_OUT="${ROOT}/clock/${variant}_ncu_reg_timeline.csv" \
    "${NCU}" \
      --target-processes all \
      --set full \
      --export "${ROOT}/ncu/${variant}.ncu-rep" \
      --force-overwrite \
      "${bin}" >"${ROOT}/logs/${variant}.ncu.log" 2>&1

  "${NCU}" --import "${ROOT}/ncu/${variant}.ncu-rep" --csv --page raw \
    >"${ROOT}/ncu/${variant}.csv" 2>>"${ROOT}/logs/${variant}.ncu.log" || true
}

{
  echo "root=${ROOT}"
  echo "case=${CASE}"
  echo "ncu=${NCU}"
  echo "ld_library_path=${LD_LIBRARY_PATH}"
  cat "${ROOT}/provenance/local_build_nvcc_version.txt" 2>/dev/null || true
  "${NCU}" --version
  nvidia-smi
} >"${ROOT}/provenance/run_env.txt"

run_plain \
  qk_pv_only_no_tma_noprofile \
  run_fa3_qk_pv_only_no_tma_noprofile_tests
run_plain \
  sync_only_no_tma_noprofile \
  run_fa3_sync_only_no_tma_noprofile_tests
run_reg_timeline \
  qk_pv_only_no_tma_reg_timeline \
  run_fa3_qk_pv_only_no_tma_reg_timeline_tests
