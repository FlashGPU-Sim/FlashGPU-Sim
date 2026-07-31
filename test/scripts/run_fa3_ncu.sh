#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${TEST_DIR}/.." && pwd)"

CUDA_ROOT="${CUDA_INSTALL_PATH:-${CUDA_HOME:-}}"
OUT_DIR=""
JOBS="${JOBS:-4}"
NCU="${NCU:-ncu}"
NCU_SET="${NCU_SET:-full}"
NCU_METRICS="${NCU_METRICS:-}"
NCU_KERNEL_NAME="${NCU_KERNEL_NAME:-regex:.*flash_fwd_kernel.*}"
NCU_TARGET_PROCESSES="${NCU_TARGET_PROCESSES:-all}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-}"
RUN_NATIVE=1
RUN_NCU=1
RESUME="${RESUME:-1}"
FORCE="${FORCE:-0}"
DUMP_SASS=0
NO_BUILD=0
DRY_RUN=0
PRINT_CASES=0
MODE=""
declare -a REQUESTED_GROUPS=()
declare -a CASE_OVERRIDES=()

usage() {
  cat <<'EOF'
Usage: run_fa3_ncu.sh --group GROUP:MODE [OPTIONS]

Build and profile one or more registry-owned FA3 experiment groups on H100.

Options:
  --group GROUP:MODE   breakdown, scaling, or concurrency selector; repeatable.
                       MODE must be explicit. Example: breakdown:qk_pv_only_no_tma
  --mode MODE          Convenience mode for one mode-less --group.
  --case NAME          Override registry cases; repeatable and applied to all groups.
  --cuda-root PATH     CUDA Toolkit root. Defaults to CUDA_INSTALL_PATH or CUDA_HOME.
  --out-dir PATH       Output directory. Defaults to test/run/H100_FA3_NCU_<time>.
  --jobs N             Build parallelism. Defaults to JOBS or 4.
  --ncu PATH           Nsight Compute executable. Defaults to NCU or ncu.
  --ncu-set NAME       NCU section set. Defaults to full.
  --ncu-metrics LIST   Metric list; overrides --ncu-set.
  --kernel-name FILTER NCU kernel filter.
  --no-build           Reuse existing binaries.
  --no-native          Skip the native sanity run.
  --no-ncu             Skip NCU collection.
  --resume             Skip cases with a successful .done marker (default).
  --no-resume          Ignore completion markers.
  --force              Rerun cases and replace existing completion markers.
  --dump-sass          Save cuobjdump --dump-sass output for each binary.
  --print-cases        Print resolved registry cases as CSV and exit.
  --dry-run            Print build/run commands without executing them.
  -h, --help           Show this help.

Examples:
  run_fa3_ncu.sh --group breakdown:qk_pv_only_no_tma
  run_fa3_ncu.sh --group scaling:baseline --group concurrency:sync_only_no_tma
  run_fa3_ncu.sh --group breakdown --mode qk_pv_only_no_tma --case H1D128FullB1S4096
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group)
      [[ $# -ge 2 ]] || { echo "--group requires a value" >&2; exit 2; }
      REQUESTED_GROUPS+=("$2")
      shift 2
      ;;
    --mode)
      [[ $# -ge 2 ]] || { echo "--mode requires a value" >&2; exit 2; }
      MODE="$2"
      shift 2
      ;;
    --case)
      [[ $# -ge 2 ]] || { echo "--case requires a value" >&2; exit 2; }
      CASE_OVERRIDES+=("$2")
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
    --no-build)
      NO_BUILD=1
      shift
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
    --dump-sass)
      DUMP_SASS=1
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

[[ ${#REQUESTED_GROUPS[@]} -gt 0 ]] || {
  echo "At least one --group GROUP:MODE is required" >&2
  exit 2
}
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || {
  echo "--jobs must be a positive integer" >&2
  exit 2
}
[[ "$RESUME" =~ ^[01]$ ]] || { echo "RESUME must be 0 or 1" >&2; exit 2; }
[[ "$FORCE" =~ ^[01]$ ]] || { echo "FORCE must be 0 or 1" >&2; exit 2; }

if [[ -n "$MODE" ]]; then
  [[ ${#REQUESTED_GROUPS[@]} -eq 1 && "${REQUESTED_GROUPS[0]}" != *:* ]] || {
    echo "--mode requires exactly one mode-less --group" >&2
    exit 2
  }
  REQUESTED_GROUPS[0]="${REQUESTED_GROUPS[0]}:${MODE}"
fi

declare -a SELECTORS=()
for selector in "${REQUESTED_GROUPS[@]}"; do
  group="${selector%%:*}"
  [[ "$selector" == *:* && -n "${selector#*:}" ]] || {
    echo "FA3 selector requires an explicit mode: $selector" >&2
    exit 2
  }
  mode="${selector#*:}"
  case "$group" in
    breakdown|scaling|concurrency) ;;
    *)
      echo "Unknown FA3 experiment group: $group" >&2
      exit 2
      ;;
  esac
  [[ "$mode" != all ]] || {
    echo "FA3 mode 'all' is intentionally unsupported; list the modes you need" >&2
    exit 2
  }
  duplicate=0
  for existing in "${SELECTORS[@]}"; do
    [[ "$existing" != "$selector" ]] || duplicate=1
  done
  [[ "$duplicate" -eq 1 ]] || SELECTORS+=("$selector")
done

registry_metadata() {
  local selector="$1"
  local group="${selector%%:*}"
  local mode="${selector#*:}"
  make -s --no-print-directory -C "$TEST_DIR" print-target-group-metadata \
    SUITE=analysis SUITE_TARGET=fa3 TARGET_GROUP="$group" TARGET_MODE="$mode"
}

binary_group_binaries() {
  make -s --no-print-directory -C "$TEST_DIR" print-binary-group \
    BINARY_GROUP="$1"
}

declare -a ROW_GROUPS=()
declare -a ROW_MODES=()
declare -a ROW_BUILD_GROUPS=()
declare -a ROW_BINARIES=()
declare -a ROW_FILTERS=()
declare -a ROW_CASES=()
declare -A SEEN_ROWS=()

for selector in "${SELECTORS[@]}"; do
  metadata="$(registry_metadata "$selector")" || {
    echo "Unable to resolve registry selector: $selector" >&2
    exit 2
  }
  IFS='|' read -r build_group binary_group executor default_filter case_list <<<"$metadata"
  [[ -n "$build_group" && -n "$binary_group" && "$executor" == fa3-profile ]] || {
    echo "Selector is not an executable FA3 profile target: $selector" >&2
    exit 2
  }

  mapfile -t binaries < <(binary_group_binaries "$binary_group")
  [[ ${#binaries[@]} -eq 1 && -n "${binaries[0]}" ]] || {
    echo "Expected one binary for FA3 selector $selector" >&2
    exit 2
  }

  declare -a cases=()
  if [[ ${#CASE_OVERRIDES[@]} -gt 0 ]]; then
    cases=("${CASE_OVERRIDES[@]}")
  else
    read -r -a cases <<<"$case_list"
  fi
  [[ ${#cases[@]} -gt 0 ]] || {
    echo "No cases resolved for FA3 selector: $selector" >&2
    exit 2
  }

  group="${selector%%:*}"
  mode="${selector#*:}"
  for case_name in "${cases[@]}"; do
    key="$group|$mode|${binaries[0]}|$case_name"
    [[ -z "${SEEN_ROWS[$key]:-}" ]] || continue
    SEEN_ROWS[$key]=1
    ROW_GROUPS+=("$group")
    ROW_MODES+=("$mode")
    ROW_BUILD_GROUPS+=("$build_group")
    ROW_BINARIES+=("${binaries[0]}")
    ROW_FILTERS+=("$default_filter")
    ROW_CASES+=("$case_name")
  done
done

if [[ "$PRINT_CASES" -eq 1 ]]; then
  echo "group,mode,binary,gtest_filter,case"
  for i in "${!ROW_CASES[@]}"; do
    printf '%s,%s,%s,%s,%s\n' \
      "${ROW_GROUPS[$i]}" "${ROW_MODES[$i]}" "${ROW_BINARIES[$i]}" \
      "${ROW_FILTERS[$i]}" "${ROW_CASES[$i]}"
  done
  exit 0
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  [[ -n "$CUDA_ROOT" ]] || {
    echo "Set --cuda-root, CUDA_INSTALL_PATH, or CUDA_HOME" >&2
    exit 2
  }
  [[ -x "$CUDA_ROOT/bin/nvcc" ]] || {
    echo "CUDA compiler not found: $CUDA_ROOT/bin/nvcc" >&2
    exit 2
  }
  export PATH="$CUDA_ROOT/bin:$PATH"
  for cuda_lib_dir in "$CUDA_ROOT/lib64" "$CUDA_ROOT/targets/x86_64-linux/lib"; do
    if [[ -d "$cuda_lib_dir" ]]; then
      export LD_LIBRARY_PATH="$cuda_lib_dir:${LD_LIBRARY_PATH:-}"
      break
    fi
  done
  if [[ "$RUN_NCU" -eq 1 ]]; then
    command -v "$NCU" >/dev/null 2>&1 || {
      echo "Nsight Compute executable not found: $NCU" >&2
      exit 2
    }
  fi
fi

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$TEST_DIR/run/H100_FA3_NCU_$(date +%Y%m%d_%H%M%S)"
fi
if [[ "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$OUT_DIR/clock" "$OUT_DIR/ncu" "$OUT_DIR/logs" \
    "$OUT_DIR/provenance" "$OUT_DIR/sass" "$OUT_DIR/status"
  OUT_DIR="$(cd "$OUT_DIR" && pwd)"
fi

declare -A BUILD_GROUPS=()
declare -a UNIQUE_BUILD_GROUPS=()
for build_group in "${ROW_BUILD_GROUPS[@]}"; do
  if [[ -z "${BUILD_GROUPS[$build_group]:-}" ]]; then
    BUILD_GROUPS[$build_group]=1
    UNIQUE_BUILD_GROUPS+=("$build_group")
  fi
done

for build_group in "${UNIQUE_BUILD_GROUPS[@]}"; do
  build_cmd=(
    make -C "$TEST_DIR" "-j$JOBS"
    "CUDA_INSTALL_PATH=$CUDA_ROOT" "CUDA_HOME=$CUDA_ROOT" "CUDA_PATH=$CUDA_ROOT"
    HOPPER_CUDA_ARCH=sm_90a CUDA_ARCH=sm_90a "$build_group"
  )
  if [[ "$NO_BUILD" -eq 0 ]]; then
    printf 'build:'
    printf ' %q' "${build_cmd[@]}"
    printf '\n'
    if [[ "$DRY_RUN" -eq 0 ]]; then
      "${build_cmd[@]}" 2>&1 | tee "$OUT_DIR/logs/build_${build_group}.log"
    fi
  fi
done

declare -a NCU_PROFILE_ARGS=()
if [[ -n "$NCU_METRICS" ]]; then
  NCU_PROFILE_ARGS=(--metrics "$NCU_METRICS")
else
  NCU_PROFILE_ARGS=(--set "$NCU_SET")
fi
if [[ -n "$NCU_KERNEL_NAME" ]]; then
  NCU_PROFILE_ARGS+=(--kernel-name "$NCU_KERNEL_NAME")
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  {
    echo "created_at=$(date -Is)"
    echo "repository=$REPO_ROOT"
    echo "commit=$(git -C "$REPO_ROOT" rev-parse HEAD)"
    echo "cuda_root=$CUDA_ROOT"
    echo "selectors=${SELECTORS[*]}"
    echo "case_overrides=${CASE_OVERRIDES[*]:-<registry>}"
    echo "jobs=$JOBS"
    echo "run_native=$RUN_NATIVE"
    echo "run_ncu=$RUN_NCU"
    echo "resume=$RESUME"
    echo "force=$FORCE"
    echo "dump_sass=$DUMP_SASS"
    echo "ncu=$NCU"
    echo "ncu_set=$NCU_SET"
    echo "ncu_metrics=$NCU_METRICS"
  } >"$OUT_DIR/provenance/run.txt"
  "$CUDA_ROOT/bin/nvcc" --version >"$OUT_DIR/provenance/nvcc_version.txt"
  if [[ "$RUN_NCU" -eq 1 ]]; then
    "$NCU" --version >"$OUT_DIR/provenance/ncu_version.txt" 2>&1 || true
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -q >"$OUT_DIR/provenance/nvidia_smi_q.txt" 2>&1 || true
  fi
  echo "group,mode,binary,case,native_status,ncu_status,seconds,report" \
    >"$OUT_DIR/status.csv"
fi

declare -A DUMPED_BINARIES=()
overall_status=0
for i in "${!ROW_CASES[@]}"; do
  group="${ROW_GROUPS[$i]}"
  mode="${ROW_MODES[$i]}"
  binary_rel="${ROW_BINARIES[$i]}"
  gtest_filter="${ROW_FILTERS[$i]}"
  case_name="${ROW_CASES[$i]}"
  binary="$TEST_DIR/$binary_rel"
  case_id="$(printf '%s_%s_%s' "$group" "$mode" "$case_name" |
    tr -c 'A-Za-z0-9_.-' '_')"
  native_log="$OUT_DIR/logs/$case_id.native.log"
  ncu_log="$OUT_DIR/logs/$case_id.ncu.log"
  report="$OUT_DIR/ncu/$case_id.ncu-rep"
  done_marker="$OUT_DIR/status/$case_id.done"

  if [[ "$RESUME" -eq 1 && "$FORCE" -ne 1 && -f "$done_marker" ]]; then
    echo "skip done: $group $mode $case_name"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      printf '%s,%s,%s,%s,skip,skip,0,%s\n' \
        "$group" "$mode" "$(basename "$binary")" "$case_name" "$report" \
        >>"$OUT_DIR/status.csv"
    fi
    continue
  fi

  declare -a NATIVE_ENV=(
    "FA3_H1D128_PROFILE_CASE_LIST=$case_name"
    "FA3_H1D128_PROFILE_OUT=$OUT_DIR/clock/$case_id.csv"
    "FA3_H1D128_PROFILE_ITER_OUT=$OUT_DIR/clock/${case_id}_iter.csv"
    "FA3_H1D128_PROFILE_TIMELINE_OUT=$OUT_DIR/clock/${case_id}_timeline.csv"
    "FA3_H1D128_PROFILE_REG_TIMELINE_OUT=$OUT_DIR/clock/${case_id}_reg_timeline.csv"
  )
  declare -a NCU_ENV=(
    "FA3_H1D128_PROFILE_CASE_LIST=$case_name"
    "FA3_H1D128_PROFILE_OUT=$OUT_DIR/clock/${case_id}_ncu.csv"
    "FA3_H1D128_PROFILE_ITER_OUT=$OUT_DIR/clock/${case_id}_ncu_iter.csv"
    "FA3_H1D128_PROFILE_TIMELINE_OUT=$OUT_DIR/clock/${case_id}_ncu_timeline.csv"
    "FA3_H1D128_PROFILE_REG_TIMELINE_OUT=$OUT_DIR/clock/${case_id}_ncu_reg_timeline.csv"
  )
  if [[ -n "$CUDA_VISIBLE_DEVICES_VALUE" ]]; then
    NATIVE_ENV+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE")
    NCU_ENV+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_VALUE")
  fi

  if [[ "$RUN_NATIVE" -eq 1 ]]; then
    printf 'native: env'
    printf ' %q' "${NATIVE_ENV[@]}" "$binary" "--gtest_filter=$gtest_filter"
    printf '\n'
  fi
  if [[ "$RUN_NCU" -eq 1 ]]; then
    printf 'ncu: env'
    printf ' %q' "${NCU_ENV[@]}" "$NCU" --target-processes "$NCU_TARGET_PROCESSES" \
      "${NCU_PROFILE_ARGS[@]}" --export "$report" --force-overwrite \
      "$binary" "--gtest_filter=$gtest_filter"
    printf '\n'
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    continue
  fi
  if [[ "$FORCE" -eq 1 ]]; then
    rm -f "$done_marker"
  fi

  [[ -x "$binary" ]] || {
    echo "FA3 executable is missing: $binary" >&2
    exit 1
  }
  if [[ "$DUMP_SASS" -eq 1 && -z "${DUMPED_BINARIES[$binary_rel]:-}" ]]; then
    "$CUDA_ROOT/bin/cuobjdump" --dump-sass "$binary" \
      >"$OUT_DIR/sass/$(basename "$binary").sass"
    DUMPED_BINARIES[$binary_rel]=1
  fi

  start="$(date +%s)"
  native_status=0
  ncu_status=0
  if [[ "$RUN_NATIVE" -eq 1 ]]; then
    env "${NATIVE_ENV[@]}" "$binary" "--gtest_filter=$gtest_filter" \
      >"$native_log" 2>&1 || native_status=$?
  fi
  if [[ "$RUN_NCU" -eq 1 && "$native_status" -eq 0 ]]; then
    env "${NCU_ENV[@]}" "$NCU" \
      --target-processes "$NCU_TARGET_PROCESSES" \
      "${NCU_PROFILE_ARGS[@]}" \
      --export "$report" --force-overwrite \
      "$binary" "--gtest_filter=$gtest_filter" >"$ncu_log" 2>&1 || ncu_status=$?
    if [[ -f "$report" ]]; then
      "$NCU" --import "$report" --csv --page raw \
        >"$OUT_DIR/ncu/$case_id.raw.csv" 2>>"$ncu_log" || true
      "$NCU" --import "$report" --csv --page details \
        >"$OUT_DIR/ncu/$case_id.details.csv" 2>>"$ncu_log" || true
    fi
  elif [[ "$RUN_NCU" -eq 1 ]]; then
    ncu_status=1
  fi
  elapsed="$(($(date +%s) - start))"
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$group" "$mode" "$(basename "$binary")" "$case_name" \
    "$native_status" "$ncu_status" "$elapsed" "$report" >>"$OUT_DIR/status.csv"
  if [[ "$native_status" -ne 0 || "$ncu_status" -ne 0 ]]; then
    overall_status=1
  else
    touch "$done_marker"
  fi
done

echo "out_dir=$OUT_DIR"
exit "$overall_status"
