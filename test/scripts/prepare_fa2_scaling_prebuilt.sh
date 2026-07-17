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

JOBS="${JOBS:-8}"
OUT_DIR="${OUT_DIR:-${TEST_DIR}/run/FA2_SCALING_PREBUILT_CUDA128_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
PREBUILT_ROOT="${OUT_DIR}/prebuilt"
BIN_DIR="${PREBUILT_ROOT}/bin"
LIB_DIR="${PREBUILT_ROOT}/lib64"
PROV_DIR="${PREBUILT_ROOT}/provenance"
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${BIN_DIR}" "${LIB_DIR}" "${PROV_DIR}" "${LOG_DIR}"

{
  echo "root=${ROOT_DIR}"
  echo "test_dir=${TEST_DIR}"
  echo "cuda=${CUDA_INSTALL_PATH}"
  echo "jobs=${JOBS}"
  echo "out_dir=${OUT_DIR}"
  echo "prebuilt_root=${PREBUILT_ROOT}"
  echo "date=$(date -Is)"
} | tee "${PROV_DIR}/prepare_env.txt"

nvcc --version | tee "${PROV_DIR}/nvcc_version.txt"

make -C "${TEST_DIR}" -j"${JOBS}" \
  CUDA_INSTALL_PATH="${CUDA_INSTALL_PATH}" \
  CUDA_HOME="${CUDA_HOME}" \
  CUDA_PATH="${CUDA_PATH}" \
  HOPPER_CUDA_ARCH=sm_90a \
  CUDA_ARCH=sm_90a \
  fa2-scaling \
  2>&1 | tee "${LOG_DIR}/build_hopper_fa2_scaling.log"

bins=(
  run_fa2_scaling_baseline_tests
  run_fa2_scaling_nothing_tests
  run_fa2_scaling_only_cp_async_tests
  run_fa2_scaling_only_softmax_tests
  run_fa2_scaling_only_mma_tests
  run_fa2_scaling_softmax_mma_tests
)

for bin_name in "${bins[@]}"; do
  src="${TEST_DIR}/build/bin/hopper/${bin_name}"
  if [[ ! -x "${src}" ]]; then
    echo "missing built binary: ${src}" >&2
    exit 1
  fi
  cp -f "${src}" "${BIN_DIR}/${bin_name}"
done

cp -P "${CUDA_INSTALL_PATH}"/lib64/libcudart.so* "${LIB_DIR}/"

cp -f "${SCRIPT_DIR}/run_fa2_scaling_h100.sh" \
  "${PREBUILT_ROOT}/run_fa2_scaling_h100.sh"
chmod +x "${PREBUILT_ROOT}/run_fa2_scaling_h100.sh"

cat >"${PREBUILT_ROOT}/run_remote.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
PREBUILT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PREBUILT_ROOT
export CUDA_INSTALL_PATH="${CUDA_INSTALL_PATH:-/usr/local/cuda-12.8}"
export OUT_DIR="${OUT_DIR:-${PREBUILT_ROOT}/../H100_FA2_SCALING_$(date +%Y%m%d_%H%M%S)}"
exec "${PREBUILT_ROOT}/run_fa2_scaling_h100.sh" "$@"
EOF
chmod +x "${PREBUILT_ROOT}/run_remote.sh"

PREBUILT_ROOT="${PREBUILT_ROOT}" \
  "${PREBUILT_ROOT}/run_fa2_scaling_h100.sh" \
  --print-cases >"${PREBUILT_ROOT}/case_manifest.csv"

(
  cd "${BIN_DIR}"
  sha256sum "${bins[@]}" >"${PROV_DIR}/sha256sums.txt"
)

{
  for bin_name in "${bins[@]}"; do
    echo "===== ${bin_name} ====="
    ldd "${BIN_DIR}/${bin_name}" || true
    echo
  done
} >"${PROV_DIR}/ldd.txt"

{
  for bin_name in "${bins[@]}"; do
    echo "===== ${bin_name} ====="
    "${BIN_DIR}/${bin_name}" --gtest_list_tests || true
    echo
  done
} >"${PROV_DIR}/gtest_list_tests.txt"

{
  echo "binary_count=${#bins[@]}"
  echo "case_count=$(($(wc -l < "${PREBUILT_ROOT}/case_manifest.csv") - 1))"
  echo "collector=${PREBUILT_ROOT}/run_fa2_scaling_h100.sh"
  echo "remote_wrapper=${PREBUILT_ROOT}/run_remote.sh"
  echo "manifest=${PREBUILT_ROOT}/case_manifest.csv"
  echo "runtime_libs=${LIB_DIR}"
} | tee "${PREBUILT_ROOT}/README.txt"

tarball="${OUT_DIR}/fa2_scaling_prebuilt_cuda128.tar.gz"
tar -C "${OUT_DIR}" -czf "${tarball}" prebuilt

echo "PREBUILT_ROOT=${PREBUILT_ROOT}"
echo "TARBALL=${tarball}"
