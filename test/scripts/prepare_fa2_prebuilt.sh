#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TEST_DIR}/.." && pwd)"

CUDA_ROOT="${CUDA_INSTALL_PATH:-${CUDA_HOME:-}}"
OUT_DIR="${TEST_DIR}/run/fa2-prebuilt"
JOBS="${JOBS:-4}"
DEVICE="h100"
CUDA_ARCH=""
BUILD_DIR=""
declare -a REQUESTED_GROUPS=()
declare -a SELECTORS=()

usage() {
  cat <<'EOF'
Usage: prepare_fa2_prebuilt.sh [OPTIONS]

Build and package one or more registry-owned FA2 experiment groups.

Options:
  --device NAME         Build for h100 (sm_90a) or rtx5090 (sm_120a).
                       Defaults to h100.
  --group NAME[:MODE]  Group to package; repeatable. Defaults to full.
                       full = smoke + small + medium + large
                       all  = full + breakdown + scaling + concurrency
                       A mode-less breakdown/scaling/concurrency selects all modes.
  --cuda-root PATH     CUDA Toolkit root. Defaults to CUDA_INSTALL_PATH or CUDA_HOME.
  --out-dir PATH       Package output. Defaults to test/run/fa2-prebuilt.
  --jobs N             Build parallelism. Defaults to JOBS or 4.
  -h, --help           Show this help.

Examples:
  prepare_fa2_prebuilt.sh --device h100 --cuda-root /usr/local/cuda-12.8 --group full
  prepare_fa2_prebuilt.sh --device rtx5090 --cuda-root /usr/local/cuda-12.8 --group full
  prepare_fa2_prebuilt.sh --group breakdown --group scaling
  prepare_fa2_prebuilt.sh --group breakdown:only_mma
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      [[ $# -ge 2 ]] || { echo "--device requires a value" >&2; exit 2; }
      DEVICE="$2"
      shift 2
      ;;
    --group)
      [[ $# -ge 2 ]] || { echo "--group requires a value" >&2; exit 2; }
      REQUESTED_GROUPS+=("$2")
      shift 2
      ;;
    --cuda-root)
      [[ $# -ge 2 ]] || { echo "--cuda-root requires a value" >&2; exit 2; }
      CUDA_ROOT="$2"
      shift 2
      ;;
    --out-dir)
      [[ $# -ge 2 ]] || { echo "--out-dir requires a value" >&2; exit 2; }
      OUT_DIR="$2"
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || { echo "--jobs requires a value" >&2; exit 2; }
      JOBS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$DEVICE" in
  h100) CUDA_ARCH=sm_90a ;;
  rtx5090) CUDA_ARCH=sm_120a ;;
  *)
    echo "--device must be h100 or rtx5090" >&2
    exit 2
    ;;
esac
BUILD_DIR="build_prebuilt_${CUDA_ARCH}"

[[ -n "$CUDA_ROOT" ]] || {
  echo "Set --cuda-root, CUDA_INSTALL_PATH, or CUDA_HOME" >&2
  exit 2
}
[[ -x "$CUDA_ROOT/bin/nvcc" ]] || {
  echo "CUDA compiler not found: $CUDA_ROOT/bin/nvcc" >&2
  exit 2
}
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || {
  echo "--jobs must be a positive integer" >&2
  exit 2
}

add_selector() {
  local candidate="$1"
  local existing=""
  for existing in "${SELECTORS[@]}"; do
    [[ "$existing" == "$candidate" ]] && return
  done
  SELECTORS+=("$candidate")
}

expand_group() {
  local spec="$1"
  case "$spec" in
    full)
      add_selector smoke
      add_selector small
      add_selector medium
      add_selector large
      ;;
    all)
      expand_group full
      add_selector breakdown:all
      add_selector scaling:all
      add_selector concurrency:all
      ;;
    breakdown|scaling|concurrency)
      add_selector "${spec}:all"
      ;;
    smoke|small|medium|large|breakdown:*|scaling:*|concurrency:*)
      add_selector "$spec"
      ;;
    *)
      echo "Unknown FA2 group selector: $spec" >&2
      exit 2
      ;;
  esac
}

if [[ ${#REQUESTED_GROUPS[@]} -eq 0 ]]; then
  REQUESTED_GROUPS=(full)
fi
for group in "${REQUESTED_GROUPS[@]}"; do
  expand_group "$group"
done

registry_metadata() {
  local selector="$1"
  local group="${selector%%:*}"
  local mode=""
  local suite="analysis"
  local target="fa2"
  if [[ "$selector" == *:* ]]; then
    mode="${selector#*:}"
  fi
  if [[ "$group" == smoke ]]; then
    suite="test"
    target="sm90"
    group="fa2-smoke"
  fi

  make -s --no-print-directory -C "$TEST_DIR" print-target-group-metadata \
    SUITE="$suite" SUITE_TARGET="$target" TARGET_GROUP="$group" \
    TARGET_MODE="$mode"
}

binary_group_binaries() {
  make -s --no-print-directory -C "$TEST_DIR" print-binary-group \
    BUILD_DIR="$BUILD_DIR" BINARY_GROUP="$1"
}

mode_for_binary() {
  local group="$1"
  local binary_name="$2"
  local prefix="run_fa2_${group}_"
  if [[ "$group" == breakdown || "$group" == scaling || "$group" == concurrency ]]; then
    local mode="${binary_name#"$prefix"}"
    printf '%s\n' "${mode%_tests}"
  else
    printf 'default\n'
  fi
}

emit_gtest_names() {
  local binary="$1"
  env -u GTEST_OUTPUT "$binary" --gtest_color=no --gtest_list_tests |
    awk '
      /^[^[:space:]]/ {
        line=$0
        sub(/[[:space:]]*#.*/, "", line)
        sub(/[[:space:]]+$/, "", line)
        if (line ~ /\.$/) {
          sub(/\.$/, "", line)
          suite=line
        }
        next
      }
      /^[[:space:]]/ {
        line=$0
        sub(/[[:space:]]*#.*/, "", line)
        sub(/^[[:space:]]+/, "", line)
        sub(/[[:space:]]+$/, "", line)
        if (suite != "" && line != "") print suite "." line
      }
    '
}

OUT_DIR="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"
PREBUILT_ROOT="$OUT_DIR"
BIN_DIR="$PREBUILT_ROOT/bin"
LIB_DIR="$PREBUILT_ROOT/lib64"
MANIFEST_DIR="$PREBUILT_ROOT/manifest"
PROV_DIR="$PREBUILT_ROOT/provenance"
LOG_DIR="$PREBUILT_ROOT/logs"
mkdir -p "$BIN_DIR" "$LIB_DIR" "$MANIFEST_DIR" "$PROV_DIR" "$LOG_DIR"

RAW_MANIFEST="$MANIFEST_DIR/fa2_cases.raw.tsv"
FINAL_MANIFEST="$MANIFEST_DIR/fa2_cases.tsv"
: >"$RAW_MANIFEST"
printf 'selector\tgroup\tmode\tbinary\ttest\n' >"$FINAL_MANIFEST"

declare -A BUILT_GROUPS=()
declare -A COPIED_BINARIES=()

for selector in "${SELECTORS[@]}"; do
  metadata="$(registry_metadata "$selector")" || {
    echo "Unable to resolve registry selector: $selector" >&2
    exit 2
  }
  IFS='|' read -r build_group binary_group executor default_filter case_list <<<"$metadata"
  [[ -n "$build_group" && -n "$binary_group" ]] || {
    echo "Incomplete registry metadata for: $selector" >&2
    exit 2
  }

  if [[ -z "${BUILT_GROUPS[$build_group]:-}" ]]; then
    echo "Building FA2 selector=$selector target=$build_group"
    make -C "$TEST_DIR" "-j$JOBS" \
      CUDA_INSTALL_PATH="$CUDA_ROOT" CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT" \
      BUILD_DIR="$BUILD_DIR" \
      HOPPER_CUDA_ARCH="$CUDA_ARCH" CUDA_ARCH="$CUDA_ARCH" "$build_group" \
      2>&1 | tee "$LOG_DIR/build_${build_group}.log"
    BUILT_GROUPS[$build_group]=1
  fi

  group="${selector%%:*}"
  while IFS= read -r binary_rel; do
    [[ -n "$binary_rel" ]] || continue
    binary_src="$TEST_DIR/$binary_rel"
    binary_name="$(basename "$binary_rel")"
    [[ -x "$binary_src" ]] || {
      echo "Built binary is missing or not executable: $binary_src" >&2
      exit 1
    }
    if [[ -z "${COPIED_BINARIES[$binary_name]:-}" ]]; then
      cp -f "$binary_src" "$BIN_DIR/$binary_name"
      COPIED_BINARIES[$binary_name]=1
    fi

    actual_mode="$(mode_for_binary "$group" "$binary_name")"
    while IFS= read -r test_name; do
      [[ -n "$test_name" ]] || continue
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "$selector" "$group" "$actual_mode" "$binary_name" "$test_name" \
        >>"$RAW_MANIFEST"
    done < <(emit_gtest_names "$binary_src")
  done < <(binary_group_binaries "$binary_group")
done

LC_ALL=C sort -u "$RAW_MANIFEST" >>"$FINAL_MANIFEST"
rm -f "$RAW_MANIFEST"
CASE_COUNT="$(($(wc -l <"$FINAL_MANIFEST") - 1))"
[[ "$CASE_COUNT" -gt 0 ]] || {
  echo "No GTest cases were discovered for selectors: ${SELECTORS[*]}" >&2
  exit 1
}

cuda_lib_dir=""
for candidate in "$CUDA_ROOT/lib64" "$CUDA_ROOT/targets/x86_64-linux/lib"; do
  if compgen -G "$candidate/libcudart.so*" >/dev/null; then
    cuda_lib_dir="$candidate"
    break
  fi
done
[[ -n "$cuda_lib_dir" ]] || {
  echo "Unable to find libcudart.so under $CUDA_ROOT" >&2
  exit 1
}
cp -a "$cuda_lib_dir"/libcudart.so* "$LIB_DIR/"

cp -f "$SCRIPT_DIR/run_fa2_ncu.sh" "$PREBUILT_ROOT/run_fa2_ncu.sh"
chmod +x "$PREBUILT_ROOT/run_fa2_ncu.sh"

{
  echo "created_at=$(date -Is)"
  echo "repository=$REPO_ROOT"
  echo "commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "cuda_root=$CUDA_ROOT"
  echo "device=$DEVICE"
  echo "cuda_arch=$CUDA_ARCH"
  echo "build_dir=$TEST_DIR/$BUILD_DIR"
  echo "jobs=$JOBS"
  echo "selectors=${SELECTORS[*]}"
  echo "manifest=$FINAL_MANIFEST"
} >"$PROV_DIR/package.txt"
"$CUDA_ROOT/bin/nvcc" --version >"$PROV_DIR/nvcc_version.txt"
(
  cd "$BIN_DIR"
  sha256sum ./* >"$PROV_DIR/sha256sums.txt"
)

cat >"$PREBUILT_ROOT/README.txt" <<EOF
FA2 prebuilt package
====================

Selectors: ${SELECTORS[*]}
Device:    ${DEVICE}
CUDA arch: ${CUDA_ARCH}
Manifest:  manifest/fa2_cases.tsv

Inspect without a GPU:
  ./run_fa2_ncu.sh --device ${DEVICE} --print-cases
  ./run_fa2_ncu.sh --device ${DEVICE} --group breakdown --dry-run

Collect on hardware:
  ./run_fa2_ncu.sh --device ${DEVICE} --group full
  ./run_fa2_ncu.sh --device ${DEVICE} --group scaling
EOF

ARCHIVE="${PREBUILT_ROOT}.tar.gz"
tar -C "$(dirname "$PREBUILT_ROOT")" -czf "$ARCHIVE" "$(basename "$PREBUILT_ROOT")"

echo "prebuilt_root=$PREBUILT_ROOT"
echo "archive=$ARCHIVE"
echo "manifest=$FINAL_MANIFEST"
echo "cases=$CASE_COUNT"
