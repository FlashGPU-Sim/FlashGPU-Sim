#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PREBUILT_ROOT=""
OUT_DIR=""
DEVICE=""
CUDA_ROOT="${CUDA_INSTALL_PATH:-${CUDA_HOME:-}}"
NCU="${NCU:-ncu}"
NCU_SET="${NCU_SET:-full}"
NCU_METRICS="${NCU_METRICS:-}"
NCU_KERNEL_NAME="${NCU_KERNEL_NAME:-regex:.*flash_fwd_kernel.*}"
NCU_TARGET_PROCESSES="${NCU_TARGET_PROCESSES:-all}"
NCU_CLOCK_CONTROL="${NCU_CLOCK_CONTROL:-}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-}"
RUN_NATIVE=1
RUN_NCU=1
RESUME="${RESUME:-1}"
FORCE="${FORCE:-0}"
DRY_RUN=0
PRINT_CASES=0
declare -a REQUESTED_GROUPS=()
declare -a SELECTORS=()
declare -a CASE_FILTERS=()

usage() {
  cat <<'EOF'
Usage: run_fa2_ncu.sh --device h100|rtx5090 [OPTIONS]

Run native and NCU collection from a package made by prepare_fa2_prebuilt.sh.

Options:
  --device NAME         h100 or rtx5090 (required).
  --prebuilt-root PATH  Package root. Auto-detected when this script is inside it.
  --group NAME[:MODE]   Group to run; repeatable. Defaults to every packaged case.
                        full and all use the same aliases as the packager.
  --case TEXT           Keep tests containing TEXT; repeatable.
  --out-dir PATH        Output directory. Defaults to a timestamped sibling directory.
  --cuda-root PATH      Optional CUDA Toolkit root for ncu/runtime lookup.
  --ncu PATH            Nsight Compute executable. Defaults to NCU or ncu.
  --ncu-set NAME        NCU section set. Defaults to full.
  --ncu-metrics LIST    Metric list; overrides --ncu-set.
  --kernel-name FILTER  NCU kernel filter.
  --no-native           Skip the native sanity run.
  --no-ncu              Skip NCU collection.
  --resume              Skip cases with a successful .done marker (default).
  --no-resume           Ignore completion markers.
  --force               Rerun cases and replace existing completion markers.
  --print-cases         Print the selected manifest as CSV and exit; no GPU required.
  --dry-run             Print native/NCU commands without executing them.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      [[ $# -ge 2 ]] || { echo "--device requires a value" >&2; exit 2; }
      DEVICE="$2"
      shift 2
      ;;
    --prebuilt-root)
      [[ $# -ge 2 ]] || { echo "--prebuilt-root requires a value" >&2; exit 2; }
      PREBUILT_ROOT="$2"
      shift 2
      ;;
    --group)
      [[ $# -ge 2 ]] || { echo "--group requires a value" >&2; exit 2; }
      REQUESTED_GROUPS+=("$2")
      shift 2
      ;;
    --case)
      [[ $# -ge 2 ]] || { echo "--case requires a value" >&2; exit 2; }
      CASE_FILTERS+=("$2")
      shift 2
      ;;
    --out-dir)
      [[ $# -ge 2 ]] || { echo "--out-dir requires a value" >&2; exit 2; }
      OUT_DIR="$2"
      shift 2
      ;;
    --cuda-root)
      [[ $# -ge 2 ]] || { echo "--cuda-root requires a value" >&2; exit 2; }
      CUDA_ROOT="$2"
      shift 2
      ;;
    --ncu)
      [[ $# -ge 2 ]] || { echo "--ncu requires a value" >&2; exit 2; }
      NCU="$2"
      shift 2
      ;;
    --ncu-set)
      [[ $# -ge 2 ]] || { echo "--ncu-set requires a value" >&2; exit 2; }
      NCU_SET="$2"
      shift 2
      ;;
    --ncu-metrics)
      [[ $# -ge 2 ]] || { echo "--ncu-metrics requires a value" >&2; exit 2; }
      NCU_METRICS="$2"
      shift 2
      ;;
    --kernel-name)
      [[ $# -ge 2 ]] || { echo "--kernel-name requires a value" >&2; exit 2; }
      NCU_KERNEL_NAME="$2"
      shift 2
      ;;
    --no-native)
      RUN_NATIVE=0
      shift
      ;;
    --no-ncu)
      RUN_NCU=0
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --no-resume)
      RESUME=0
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --print-cases)
      PRINT_CASES=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

[[ "$RESUME" =~ ^[01]$ ]] || { echo "RESUME must be 0 or 1" >&2; exit 2; }
[[ "$FORCE" =~ ^[01]$ ]] || { echo "FORCE must be 0 or 1" >&2; exit 2; }

case "$DEVICE" in
  h100) ;;
  rtx5090)
    if [[ -z "$NCU_CLOCK_CONTROL" ]]; then
      NCU_CLOCK_CONTROL=none
    fi
    ;;
  *)
    echo "--device must be h100 or rtx5090" >&2
    exit 2
    ;;
esac

if [[ -z "$PREBUILT_ROOT" && -f "$SCRIPT_DIR/manifest/fa2_cases.tsv" ]]; then
  PREBUILT_ROOT="$SCRIPT_DIR"
fi
[[ -n "$PREBUILT_ROOT" ]] || {
  echo "Use --prebuilt-root with a package made by prepare_fa2_prebuilt.sh" >&2
  exit 2
}
PREBUILT_ROOT="$(cd "$PREBUILT_ROOT" && pwd)"
BIN_ROOT="$PREBUILT_ROOT/bin"
MANIFEST="$PREBUILT_ROOT/manifest/fa2_cases.tsv"
PACKAGE_METADATA="$PREBUILT_ROOT/provenance/package.txt"
[[ -f "$MANIFEST" ]] || { echo "Missing package manifest: $MANIFEST" >&2; exit 2; }
[[ -d "$BIN_ROOT" ]] || { echo "Missing package binary directory: $BIN_ROOT" >&2; exit 2; }
[[ -f "$PACKAGE_METADATA" ]] || {
  echo "Missing package metadata: $PACKAGE_METADATA" >&2
  exit 2
}
PACKAGE_DEVICE="$(awk -F= '$1 == "device" { print $2; exit }' "$PACKAGE_METADATA")"
PACKAGE_ARCH="$(awk -F= '$1 == "cuda_arch" { print $2; exit }' "$PACKAGE_METADATA")"
[[ -n "$PACKAGE_DEVICE" && -n "$PACKAGE_ARCH" ]] || {
  echo "Package metadata does not declare device and cuda_arch" >&2
  exit 2
}
[[ "$PACKAGE_DEVICE" == "$DEVICE" ]] || {
  echo "Package targets $PACKAGE_DEVICE ($PACKAGE_ARCH), not requested device $DEVICE" >&2
  exit 2
}

if [[ -n "$CUDA_ROOT" ]]; then
  export PATH="$CUDA_ROOT/bin:$PATH"
  for cuda_lib_dir in "$CUDA_ROOT/lib64" "$CUDA_ROOT/targets/x86_64-linux/lib"; do
    if [[ -d "$cuda_lib_dir" ]]; then
      export LD_LIBRARY_PATH="$cuda_lib_dir:${LD_LIBRARY_PATH:-}"
      break
    fi
  done
fi
if [[ -d "$PREBUILT_ROOT/lib64" ]]; then
  export LD_LIBRARY_PATH="$PREBUILT_ROOT/lib64:${LD_LIBRARY_PATH:-}"
fi

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

for group in "${REQUESTED_GROUPS[@]}"; do
  expand_group "$group"
done

selector_matches() {
  local group="$1"
  local mode="$2"
  local selector=""
  [[ ${#SELECTORS[@]} -eq 0 ]] && return 0
  for selector in "${SELECTORS[@]}"; do
    if [[ "$selector" == *:* ]]; then
      wanted_group="${selector%%:*}"
      wanted_mode="${selector#*:}"
      if [[ "$group" == "$wanted_group" && ( "$wanted_mode" == all || "$mode" == "$wanted_mode" ) ]]; then
        return 0
      fi
    elif [[ "$group" == "$selector" ]]; then
      return 0
    fi
  done
  return 1
}

case_matches() {
  local test_name="$1"
  local needle=""
  [[ ${#CASE_FILTERS[@]} -eq 0 ]] && return 0
  for needle in "${CASE_FILTERS[@]}"; do
    [[ "$test_name" == *"$needle"* ]] && return 0
  done
  return 1
}

declare -a ROW_GROUPS=()
declare -a ROW_MODES=()
declare -a ROW_BINARIES=()
declare -a ROW_TESTS=()
declare -A SEEN_ROWS=()

{
  IFS= read -r _
  while IFS=$'\t' read -r selector group mode binary_name test_name; do
    [[ -n "$group" && -n "$binary_name" && -n "$test_name" ]] || continue
    selector_matches "$group" "$mode" || continue
    case_matches "$test_name" || continue
    key="$group|$mode|$binary_name|$test_name"
    [[ -z "${SEEN_ROWS[$key]:-}" ]] || continue
    SEEN_ROWS[$key]=1
    ROW_GROUPS+=("$group")
    ROW_MODES+=("$mode")
    ROW_BINARIES+=("$binary_name")
    ROW_TESTS+=("$test_name")
  done
} <"$MANIFEST"

[[ ${#ROW_TESTS[@]} -gt 0 ]] || {
  echo "No packaged FA2 cases matched the requested group/case filters" >&2
  exit 1
}

if [[ "$PRINT_CASES" -eq 1 ]]; then
  echo "group,mode,binary,test"
  for i in "${!ROW_TESTS[@]}"; do
    printf '%s,%s,%s,%s\n' \
      "${ROW_GROUPS[$i]}" "${ROW_MODES[$i]}" \
      "${ROW_BINARIES[$i]}" "${ROW_TESTS[$i]}"
  done
  exit 0
fi

if [[ "$DRY_RUN" -eq 0 && "$RUN_NCU" -eq 1 ]]; then
  command -v "$NCU" >/dev/null 2>&1 || {
    echo "Nsight Compute executable not found: $NCU" >&2
    exit 2
  }
fi

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$PREBUILT_ROOT/../${DEVICE^^}_FA2_NCU_$(date +%Y%m%d_%H%M%S)"
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$OUT_DIR/native" "$OUT_DIR/ncu" "$OUT_DIR/logs" \
    "$OUT_DIR/provenance" "$OUT_DIR/status"
  OUT_DIR="$(cd "$OUT_DIR" && pwd)"
fi

declare -a NCU_PROFILE_ARGS=()
if [[ -n "$NCU_METRICS" ]]; then
  NCU_PROFILE_ARGS=(--metrics "$NCU_METRICS")
else
  NCU_PROFILE_ARGS=(--set "$NCU_SET")
fi
if [[ -n "$NCU_KERNEL_NAME" ]]; then
  NCU_PROFILE_ARGS+=(--kernel-name "$NCU_KERNEL_NAME")
fi
if [[ -n "$NCU_CLOCK_CONTROL" ]]; then
  NCU_PROFILE_ARGS+=(--clock-control "$NCU_CLOCK_CONTROL")
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  {
    echo "created_at=$(date -Is)"
    echo "device=$DEVICE"
    echo "prebuilt_root=$PREBUILT_ROOT"
    echo "manifest=$MANIFEST"
    echo "package_device=$PACKAGE_DEVICE"
    echo "package_cuda_arch=$PACKAGE_ARCH"
    echo "groups=${REQUESTED_GROUPS[*]:-<all-packaged>}"
    echo "case_filters=${CASE_FILTERS[*]:-<none>}"
    echo "run_native=$RUN_NATIVE"
    echo "run_ncu=$RUN_NCU"
    echo "resume=$RESUME"
    echo "force=$FORCE"
    echo "ncu=$NCU"
    echo "ncu_set=$NCU_SET"
    echo "ncu_metrics=$NCU_METRICS"
  } >"$OUT_DIR/provenance/run.txt"
  if command -v "$NCU" >/dev/null 2>&1; then
    "$NCU" --version >"$OUT_DIR/provenance/ncu_version.txt" 2>&1 || true
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -q >"$OUT_DIR/provenance/nvidia_smi_q.txt" 2>&1 || true
  fi
  echo "group,mode,binary,test,native_status,ncu_status,seconds,report" \
    >"$OUT_DIR/status.csv"
fi

overall_status=0
for i in "${!ROW_TESTS[@]}"; do
  group="${ROW_GROUPS[$i]}"
  mode="${ROW_MODES[$i]}"
  binary_name="${ROW_BINARIES[$i]}"
  test_name="${ROW_TESTS[$i]}"
  binary="$BIN_ROOT/$binary_name"

  case_id="$(printf '%s_%s_%s' "$group" "${mode:-default}" "$test_name" |
    tr -c 'A-Za-z0-9_.-' '_')"
  native_log="$OUT_DIR/native/$case_id.log"
  ncu_log="$OUT_DIR/logs/$case_id.ncu.log"
  report="$OUT_DIR/ncu/$case_id.ncu-rep"
  raw_csv="$OUT_DIR/ncu/$case_id.raw.csv"
  details_csv="$OUT_DIR/ncu/$case_id.details.csv"
  done_marker="$OUT_DIR/status/$case_id.done"

  if [[ "$RESUME" -eq 1 && "$FORCE" -ne 1 && -f "$done_marker" ]]; then
    echo "=== skip done $group ${mode:-default} $test_name ==="
    if [[ "$DRY_RUN" -eq 0 ]]; then
      printf '%s,%s,%s,%s,skip,skip,0,%s\n' \
        "$group" "$mode" "$binary_name" "$test_name" "$report" \
        >>"$OUT_DIR/status.csv"
    fi
    continue
  fi

  [[ -x "$binary" ]] || { echo "Missing executable: $binary" >&2; exit 1; }
  declare -a CASE_ENV=()
  case "$group" in
    smoke|small|medium|large) CASE_ENV=(FA2_RUN_32KI=1) ;;
  esac
  if [[ -n "$CUDA_VISIBLE_DEVICES_VALUE" ]]; then
    CASE_ENV+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE")
  fi

  echo "=== $group ${mode:-default} $test_name ==="
  if [[ "$RUN_NATIVE" -eq 1 ]]; then
    echo "native: env ${CASE_ENV[*]} $binary --gtest_filter=$test_name"
  fi
  if [[ "$RUN_NCU" -eq 1 ]]; then
    echo "ncu: env ${CASE_ENV[*]} $NCU --target-processes $NCU_TARGET_PROCESSES ${NCU_PROFILE_ARGS[*]} --export $report --force-overwrite $binary --gtest_filter=$test_name"
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    continue
  fi
  if [[ "$FORCE" -eq 1 ]]; then
    rm -f "$done_marker"
  fi

  start="$(date +%s)"
  native_status=0
  ncu_status=0
  if [[ "$RUN_NATIVE" -eq 1 ]]; then
    env "${CASE_ENV[@]}" "$binary" "--gtest_filter=$test_name" \
      >"$native_log" 2>&1 || native_status=$?
  fi
  if [[ "$RUN_NCU" -eq 1 && "$native_status" -eq 0 ]]; then
    env "${CASE_ENV[@]}" "$NCU" \
      --target-processes "$NCU_TARGET_PROCESSES" \
      "${NCU_PROFILE_ARGS[@]}" \
      --export "$report" --force-overwrite \
      "$binary" "--gtest_filter=$test_name" >"$ncu_log" 2>&1 || ncu_status=$?
    if [[ -f "$report" ]]; then
      "$NCU" --import "$report" --csv --page raw >"$raw_csv" 2>>"$ncu_log" || true
      "$NCU" --import "$report" --csv --page details >"$details_csv" 2>>"$ncu_log" || true
    fi
  elif [[ "$RUN_NCU" -eq 1 ]]; then
    ncu_status=1
  fi
  elapsed="$(($(date +%s) - start))"
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$group" "$mode" "$binary_name" "$test_name" \
    "$native_status" "$ncu_status" "$elapsed" "$report" >>"$OUT_DIR/status.csv"
  if [[ "$native_status" -ne 0 || "$ncu_status" -ne 0 ]]; then
    overall_status=1
  else
    touch "$done_marker"
  fi
done

echo "out_dir=$OUT_DIR"
exit "$overall_status"
