#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TEST_DIR}/.." && pwd)"

CUDA_ROOT="${CUDA_INSTALL_PATH:-${CUDA_HOME:-}}"
OUT_DIR="${TEST_DIR}/run/fa2-prebuilt"
JOBS="${JOBS:-4}"
DEVICE="h100"
ARCH_NAME=""
NVCC_TARGET=""
BUILD_DIR=""
declare -a REQUESTED_GROUPS=()
declare -a SELECTORS=()

usage() {
  cat <<'EOF'
Usage: prepare_fa2_prebuilt.sh [OPTIONS]

Build and package one or more manifest-owned FA2 experiment profiles.

Options:
  --device NAME         Use h100/sm90 or rtx5090/sm120. The selected
                       architecture manifest must include FA2. Defaults to h100.
  --group NAME[:MODE]  Group to package; repeatable. Defaults to full.
                       full = smoke + small + medium + large
                       all  = full + breakdown + scaling + concurrency
                       A mode-less breakdown/scaling/concurrency selects all modes.
  --cuda-root PATH     CUDA Toolkit root. Defaults to CUDA_INSTALL_PATH or CUDA_HOME.
  --out-dir PATH       Package output. Defaults to tests/run/fa2-prebuilt.
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
  h100) ARCH_NAME=sm90 ;;
  rtx5090) ARCH_NAME=sm120 ;;
  *)
    echo "--device must be h100 or rtx5090" >&2
    exit 2
    ;;
esac

ARCH_METADATA="$(make -s --no-print-directory -C "$TEST_DIR" \
  print-architecture-metadata ARCH="$ARCH_NAME")" || {
  echo "Unable to resolve architecture manifest: $ARCH_NAME" >&2
  exit 2
}
IFS='|' read -r DEFAULT_CONFIG NVCC_TARGET <<<"$ARCH_METADATA"
if ! make -s --no-print-directory -C "$TEST_DIR" \
    list-test-groups ARCH="$ARCH_NAME" | grep -Fxq fa2; then
  echo "FA2 is not supported by the $ARCH_NAME architecture manifest" >&2
  exit 2
fi
BUILD_DIR="build_prebuilt_${NVCC_TARGET}"

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

manifest_metadata() {
  local selector="$1"
  local group="${selector%%:*}"
  local mode=""
  if [[ "$selector" == *:* ]]; then
    mode="${selector#*:}"
  fi

  make -s --no-print-directory -C "$TEST_DIR" print-test-group-metadata \
    ARCH="$ARCH_NAME" TEST_GROUP=fa2 PROFILE="$group" MODE="$mode"
}

binary_group_binaries() {
  make -s --no-print-directory -C "$TEST_DIR" print-binary-group \
    BUILD_DIR="$BUILD_DIR" BINARY_GROUP="$1"
}

mode_for_binary() {
  local group="$1"
  local binary_name="$2"
  local prefix="${group}_"
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

declare -A BUILT_TARGETS=()
declare -A COPIED_BINARIES=()

for selector in "${SELECTORS[@]}"; do
  metadata="$(manifest_metadata "$selector")" || {
    echo "Unable to resolve manifest selector: $selector" >&2
    exit 2
  }
  IFS='|' read -r build_target binary_group executor default_filter case_list <<<"$metadata"
  [[ -n "$build_target" && -n "$binary_group" ]] || {
    echo "Incomplete manifest metadata for: $selector" >&2
    exit 2
  }

  if [[ -z "${BUILT_TARGETS[$build_target]:-}" ]]; then
    echo "Building FA2 selector=$selector target=$build_target"
    make -C "$TEST_DIR" "-j$JOBS" \
      CUDA_INSTALL_PATH="$CUDA_ROOT" CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT" \
      BUILD_DIR="$BUILD_DIR" ARCH="$ARCH_NAME" "$build_target" \
      2>&1 | tee "$LOG_DIR/build_${build_target}.log"
    BUILT_TARGETS[$build_target]=1
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
  echo "architecture=$ARCH_NAME"
  echo "nvcc_target=$NVCC_TARGET"
  echo "config=$DEFAULT_CONFIG"
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
Device:      ${DEVICE}
Architecture: ${ARCH_NAME}
NVCC target: ${NVCC_TARGET}
Manifest:    manifest/fa2_cases.tsv

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
