#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../../../setup_environment" ]]; then
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
  TEST_DIR="${ROOT_DIR}/tests"
else
  TEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
  ROOT_DIR="$(cd "${TEST_DIR}/.." && pwd)"
fi

ARCH="${ARCH:-sm_100a}"
BUILD_FA2="${BUILD_FA2:-1}"
BUILD_FA3="${BUILD_FA3:-1}"
BUILD_GROUPS="${BUILD_GROUPS:-smoke small}"
MAKE_JOBS="${MAKE_JOBS:-1}"
FORCE_REBUILD="${FORCE_REBUILD:-1}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/../b200_native_collect}"
PREBUILT_ROOT="${PREBUILT_ROOT:-${OUT_ROOT}/prebuilt_cuda13_${ARCH}_$(date +%Y%m%d_%H%M%S)}"

usage() {
  cat <<'EOF'
Usage:
  prepare_b200_fa2_fa3_prebuilt.sh

Environment:
  ARCH             CUDA architecture, default sm_100a.
  FA4_CUDA_ROOT    CUDA root from the FA4 venv. Auto-detected if unset.
  FA4_PYTHON       Python from the FA4 venv. Auto-detected if unset.
  BUILD_FA2        Build FA2 split binaries, default 1.
  BUILD_FA3        Build FA3 extended binary, default 1.
  BUILD_GROUPS     FA2 groups to build: smoke small medium large, default
                   "smoke small".
  MAKE_JOBS        make -j value, default 1.
  FORCE_REBUILD    Pass -B to make so old Hopper binaries cannot be reused.
                   Default 1.
  OUT_ROOT         Output root, default ../b200_native_collect.
  PREBUILT_ROOT    Exact prebuilt output directory.
  CUDA_DRIVER_STUB_DIR
                   Directory containing libcuda.so stub. Auto-detected.

The output bundle contains bin/, lib/, scripts/, and provenance/. It is meant
to be copied to a B200 host and consumed by run_b200_fa2_fa3_ncu.sh.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

detect_fa4_python() {
  if [[ -n "${FA4_PYTHON:-}" ]]; then
    echo "${FA4_PYTHON}"
  elif [[ -x "${ROOT_DIR}/../fa4-env-cu133/bin/python" ]]; then
    echo "${ROOT_DIR}/../fa4-env-cu133/bin/python"
  else
    echo python3
  fi
}

detect_fa4_cuda_root() {
  if [[ -n "${FA4_CUDA_ROOT:-}" ]]; then
    echo "${FA4_CUDA_ROOT}"
    return
  fi
  local python_bin
  python_bin="$(detect_fa4_python)"
  "${python_bin}" <<'PY'
import sysconfig
from pathlib import Path

root = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13"
if (root / "bin" / "nvcc").exists() and (root / "lib" / "libcudart.so.13").exists():
    print(root)
else:
    raise SystemExit(1)
PY
}

detect_driver_stub_dir() {
  if [[ -n "${CUDA_DRIVER_STUB_DIR:-}" ]]; then
    echo "${CUDA_DRIVER_STUB_DIR}"
    return
  fi
  local candidate
  for candidate in \
      "${CUDA_HOME:-}/lib64/stubs" \
      "${CUDA_PATH:-}/lib64/stubs" \
      "/usr/local/cuda/lib64/stubs" \
      "/usr/local/cuda-13.0/lib64/stubs" \
      "/usr/local/cuda-12.8/lib64/stubs"; do
    if [[ -n "${candidate}" && -f "${candidate}/libcuda.so" ]]; then
      echo "${candidate}"
      return
    fi
  done
  return 1
}

append_fa2_group_targets() {
  local group="$1"
  local variant
  for variant in h32d64_full h32d64_causal h16d128_full h16d128_causal; do
    BUILD_TARGETS+=("build/bin/hopper/run_fa2_${group}_${variant}_tests")
  done
}

FA4_CUDA_ROOT="$(detect_fa4_cuda_root)"
FA4_CUDA_ROOT="$(cd "${FA4_CUDA_ROOT}" && pwd)"
NVCC="${NVCC:-${FA4_CUDA_ROOT}/bin/nvcc}"
CUOBJDUMP="${CUOBJDUMP:-${FA4_CUDA_ROOT}/bin/cuobjdump}"
RUNTIME_LIB="${FA4_CUDA_ROOT}/lib/libcudart.so.13"
if [[ ! -x "${NVCC}" ]]; then
  echo "nvcc not found: ${NVCC}" >&2
  exit 2
fi
if [[ ! -f "${RUNTIME_LIB}" ]]; then
  echo "CUDA runtime not found: ${RUNTIME_LIB}" >&2
  exit 2
fi

DRIVER_STUB_DIR="$(detect_driver_stub_dir || true)"
if [[ -z "${DRIVER_STUB_DIR}" ]]; then
  echo "warning: libcuda stub not found; falling back to -lcuda" >&2
fi

mkdir -p "${PREBUILT_ROOT}/bin" "${PREBUILT_ROOT}/lib" \
  "${PREBUILT_ROOT}/scripts" "${PREBUILT_ROOT}/provenance"
cp -f "${RUNTIME_LIB}" "${PREBUILT_ROOT}/lib/"

BUILD_TARGETS=()
if [[ "${BUILD_FA2}" == "1" ]]; then
  for group in ${BUILD_GROUPS}; do
    case "${group}" in
      smoke|small|medium|large)
        append_fa2_group_targets "${group}"
        ;;
      *)
        echo "unknown FA2 group in BUILD_GROUPS: ${group}" >&2
        exit 2
        ;;
    esac
  done
fi
if [[ "${BUILD_FA3}" == "1" ]]; then
  BUILD_TARGETS+=("build/bin/hopper/run_fa3_extended_tests")
fi

if [[ "${#BUILD_TARGETS[@]}" -eq 0 ]]; then
  echo "nothing to build; set BUILD_FA2 or BUILD_FA3" >&2
  exit 2
fi

{
  echo "root=${ROOT_DIR}"
  echo "test_dir=${TEST_DIR}"
  echo "prebuilt_root=${PREBUILT_ROOT}"
  echo "arch=${ARCH}"
  echo "fa4_cuda_root=${FA4_CUDA_ROOT}"
  echo "nvcc=${NVCC}"
  echo "cuobjdump=${CUOBJDUMP}"
  echo "driver_stub_dir=${DRIVER_STUB_DIR}"
  echo "build_fa2=${BUILD_FA2}"
  echo "build_fa3=${BUILD_FA3}"
  echo "build_groups=${BUILD_GROUPS}"
  echo "make_jobs=${MAKE_JOBS}"
  echo "force_rebuild=${FORCE_REBUILD}"
  echo "date=$(date -Is)"
  git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null | sed 's/^/git_branch=/'
  git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null | sed 's/^/git_commit=/'
  git -C "${ROOT_DIR}" status --short 2>/dev/null | sed 's/^/git_status=/'
} | tee "${PREBUILT_ROOT}/provenance/build_env.txt"
"${NVCC}" --version | tee "${PREBUILT_ROOT}/provenance/nvcc_version.txt"
printf "%s\n" "${BUILD_TARGETS[@]}" >"${PREBUILT_ROOT}/provenance/build_targets.txt"

driver_link_args=()
if [[ -n "${DRIVER_STUB_DIR}" ]]; then
  driver_link_args+=("-L${DRIVER_STUB_DIR}" "-lcuda")
else
  driver_link_args+=("-lcuda")
fi

cuda_libs=(
  "-L${PREBUILT_ROOT}/lib"
  "-Wl,-rpath,\$\$ORIGIN/../lib"
  "-l:libcudart.so.13"
  "${driver_link_args[@]}"
  "-static-libstdc++"
  "-static-libgcc"
)

(
  cd "${TEST_DIR}"
  make_args=()
  if [[ "${FORCE_REBUILD}" == "1" ]]; then
    make_args+=("-B")
  fi
  make -j"${MAKE_JOBS}" "${make_args[@]}" "${BUILD_TARGETS[@]}" \
    CUDA_INSTALL_PATH="${FA4_CUDA_ROOT}" \
    CUDA_HOME="${FA4_CUDA_ROOT}" \
    CUDA_PATH="${FA4_CUDA_ROOT}" \
    NVCC="${NVCC}" \
    CUDA_ARCH="${ARCH}" \
    HOPPER_CUDA_ARCH="${ARCH}" \
    CUDA_LIBS="${cuda_libs[*]}"
)

manifest="${PREBUILT_ROOT}/manifest.csv"
echo "binary,path,sha256,elf,ptx" >"${manifest}"
for target in "${BUILD_TARGETS[@]}"; do
  src="${TEST_DIR}/${target}"
  if [[ ! -x "${src}" ]]; then
    echo "expected binary missing after build: ${src}" >&2
    exit 1
  fi
  name="$(basename "${src}")"
  dst="${PREBUILT_ROOT}/bin/${name}"
  cp -f "${src}" "${dst}"
  chmod +x "${dst}"
  if command -v patchelf >/dev/null 2>&1; then
    patchelf --set-rpath '$ORIGIN/../lib' "${dst}" || true
  fi
  sha="$(sha256sum "${dst}" | awk '{print $1}')"
  elf="$("${CUOBJDUMP}" --list-elf "${dst}" 2>/dev/null | tr '\n' ';' | sed 's/;$//')"
  ptx="$("${CUOBJDUMP}" --list-ptx "${dst}" 2>/dev/null | tr '\n' ';' | sed 's/;$//')"
  if [[ "${elf}" != *".${ARCH}.cubin"* ]]; then
    echo "binary ${name} does not contain ${ARCH} cubin" >&2
    echo "${elf}" >&2
    exit 1
  fi
  printf '%s,%s,%s,"%s","%s"\n' "${name}" "${dst}" "${sha}" "${elf}" "${ptx}" >>"${manifest}"
  {
    echo "== ${name} =="
    "${CUOBJDUMP}" --list-elf "${dst}" || true
    "${CUOBJDUMP}" --list-ptx "${dst}" || true
  } >"${PREBUILT_ROOT}/provenance/${name}.cuobjdump.txt"
done

if command -v ldd >/dev/null 2>&1; then
  for dst in "${PREBUILT_ROOT}"/bin/*; do
    ldd "${dst}" 2>/dev/null |
      awk '/libstdc\+\+\.so|libgcc_s\.so/ { print $3 }' |
      while IFS= read -r dep; do
        if [[ -f "${dep}" ]]; then
          dep_dst="${PREBUILT_ROOT}/lib/$(basename "${dep}")"
          if [[ ! -e "${dep_dst}" ]]; then
            cp "${dep}" "${dep_dst}"
          fi
        fi
      done
  done
fi

cp -f "${SCRIPT_DIR}/run_b200_fa2_fa3_ncu.sh" "${PREBUILT_ROOT}/scripts/"
cp -f "${SCRIPT_DIR}/prepare_b200_fa2_fa3_prebuilt.sh" "${PREBUILT_ROOT}/scripts/"

cat >"${PREBUILT_ROOT}/README.md" <<EOF
# B200 FA2/FA3 native prebuilt

- Built at: $(date -Is)
- Arch: \`${ARCH}\`
- CUDA: \`${FA4_CUDA_ROOT}\`
- Default NCU script: \`scripts/run_b200_fa2_fa3_ncu.sh\`
- Results default to \`../results/B200_FA2_FA3_NCU_<timestamp>\` relative to this bundle.

FA2 is compiled as split gtest binaries. FA3 is compiled as the extended
Hopper-origin gtest binary under \`${ARCH}\`; check provenance because the
SM100A forward kernel may be an SM90 guard/stub even when compilation succeeds.
EOF

echo "${PREBUILT_ROOT}" | tee "${PREBUILT_ROOT}/prebuilt_root.txt"
