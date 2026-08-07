#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CONFIG="SM100_B200"
SUITE="smoke"
RUN_DIR=""
RUN_DIR_SET=0
ACTION="run"
CASE_NAME=""
LAUNCHER="generated"
ARTIFACT_DIRECTION="fwd"
FATBIN=""
PTX=""
KERNEL=""
TIMEOUT_SECONDS=0
EXPORT_TIMEOUT_SECONDS=600
DRY_RUN=0
VERBOSE=0
PIN_CPU=1
CPU_SET="${FA4_CPU_SET:-}"
CPUS_PER_JOB="${FA4_CPUS_PER_JOB:-}"
THREADS_PER_JOB="${FA4_THREADS_PER_JOB:-}"
AFFINITY_MODE="${FA4_AFFINITY_MODE:-}"
CPU_SET_EXPLICIT=0
NO_PIN_EXPLICIT=0
LIST_ONLY=0
REBUILD_LAUNCHER=0
RUN_BWD=0
LEGACY_COORDS=0
ARTIFACT_HEAD_DIM=64
ARTIFACT_HEAD_DIM_SET=0
ARTIFACT_HEAD_DIM_V=""
ARTIFACT_HEAD_DIM_V_SET=0
ARTIFACT_DTYPE="fp16"
ARTIFACT_DTYPE_SET=0
ARTIFACT_CAUSAL=1
ARTIFACT_CAUSAL_SET=0
DYNAMIC_SMEM=231424
CASES_FILE="$SCRIPT_DIR/fa4_b200_cases.csv"
FA4_PYTHON="${FA4_PYTHON:-}"
GENERATED_EXPORT_NAME="fa4_b200_launcher"
GENERATED_HEADER=""
GENERATED_OBJECT=""
GENERATED_PTX=""
GENERATED_METADATA=""
GENERATED_CUBIN=""
LOG_DIR=""
MANIFEST=""
ARCHIVE_MANIFEST=""
RUN_ID=""
ORIGINAL_ARGV=()
SELECTED_CASE_ROWS=()
CASE_RESULTS=()
RUN_COUNT=0
PASS_COUNT=0
SKIP_COUNT=0
FAIL_COUNT=0
CONFIG_SYNCED=0
RUNNER_BUILT=0
SYNCED_ICNT_FILES=()

COLOR_ENABLED=0
COLOR_RESET=""
COLOR_DIM=""
COLOR_BRIGHT_RED=""
COLOR_BRIGHT_GREEN=""
COLOR_BRIGHT_YELLOW=""
COLOR_BRIGHT_MAGENTA=""
COLOR_BRIGHT_CYAN=""
COLOR_BRIGHT_WHITE=""
if [[ -t 1 && -z "${NO_COLOR:-}" && "${TERM:-}" != "dumb" ]]; then
  COLOR_ENABLED=1
  COLOR_RESET=$'\033[0m'
  COLOR_DIM=$'\033[2m'
  COLOR_BRIGHT_RED=$'\033[91m'
  COLOR_BRIGHT_GREEN=$'\033[92m'
  COLOR_BRIGHT_YELLOW=$'\033[93m'
  COLOR_BRIGHT_MAGENTA=$'\033[95m'
  COLOR_BRIGHT_CYAN=$'\033[96m'
  COLOR_BRIGHT_WHITE=$'\033[97m'
fi

usage() {
  cat <<'EOF'
usage: tests/dev/fa4/run_fa4_b200_cases.sh [action] [options]

actions:
  run                       Export/cache, build, simulate, and check (default)
  export                    Export/cache the CuTe DSL C and device artifacts
  sim                       Rebuild the runner, simulate, and check using
                            existing artifacts

options:
  --config NAME             GPGPU-Sim config to copy into the FA4 run dir
                            (default: SM100_B200)
  --suite NAME              smoke | small | medium | large | all
                            (default: smoke)
  --case NAME               Select exactly one CSV case. Unless explicitly
                            overridden, its dtype/head dimensions/causal mode
                            also select the generated artifact specialization.
  --launcher NAME           generated | manual
                            (default: generated)
  --direction fwd|bwd       FA4 generated kernel direction (default: fwd)
  --run-bwd                 Actually launch backward artifacts. By default
                            --direction bwd exports only; BWD opaque TMA store
                            descriptors are still under bring-up.
  --run-dir DIR             FA4 artifact/run directory
                            (default for --case NAME:
                            temp/fa4-b200-runs/NAME/CONFIG;
                            otherwise: .../suite-SUITE/CONFIG)
  --fatbin PATH             FA4 fatbin path
  --ptx PATH                PTX path used to auto-detect the kernel name
  --kernel NAME             Kernel symbol name
  --timeout SECONDS         Per-case timeout (default: 0, disabled)
  --export-timeout SECONDS  CuTe DSL export timeout (default: 600)
  --artifact-head-dim N     Head dimension compiled into the artifact
                            (default: 64)
  --artifact-head-dim-v N   V/O head dimension compiled into the artifact
                            (default: artifact-head-dim)
  --artifact-dtype fp16|bf16
                            Dtype compiled into the artifact (default: fp16)
  --causal                  Use a causal generated FA4 artifact (default)
  --non-causal              Use a non-causal generated FA4 artifact
  --dynamic-smem BYTES      Dynamic shared memory bytes (default: 231424)
  --fa4-python PATH         Python interpreter with FA4 installed
                            (default: FA4_PYTHON, then
                            ../fa4-env-cu133/bin/python if present, then
                            python3 from PATH)
  --rebuild-launcher        Re-export the generated FA4 C launcher
  --legacy-coord-pointers   Use the first bring-up harness ABI behavior
  --list                    Print selected cases and exit
  --dry-run                 Print commands without running them
  --verbose                 Stream full exporter/build/simulator logs
  --cpus-per-job N          Physical cores selected for a direct run
                            (default: 4)
  --threads-per-job N       Thread-library limit (default: cpus-per-job)
  --cpu-set LIST            Manually bind the whole workload to this CPU list;
                            otherwise idle physical cores are selected
  --no-pin                  Let the whole workload run on all allowed CPUs
  -h, --help                Show this help

The default terminal output is a stage summary. Full output is kept under
RUN_DIR/logs, and RUN_DIR/run-manifest.json records the resolved workload,
software, artifacts, launch contract, checks, results, and warning categories.
Stage colors are enabled on interactive terminals; set NO_COLOR=1 to disable.

CUDA_INSTALL_PATH must point at a CUDA toolkit usable by this GPGPU-Sim build;
/usr/local/cuda-12.8 is selected automatically when available.

For PTX 9.1 / sm_100a artifacts, the runner normally finds CUDA 13 under the
selected FA4 Python environment (site-packages/nvidia/cu13) and sets
PTXAS_CUDA_INSTALL_PATH automatically.  FA4_CU13_ROOT and
PTXAS_CUDA_INSTALL_PATH are optional overrides for nonstandard layouts.
EOF
}

parse_args() {
  if [[ $# -gt 0 && "$1" != -* ]]; then
    ACTION="$1"
    shift
  fi
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        CONFIG="$2"
        shift 2
        ;;
      --suite)
        SUITE="$2"
        shift 2
        ;;
      --case)
        CASE_NAME="$2"
        shift 2
        ;;
      --launcher)
        LAUNCHER="$2"
        shift 2
        ;;
      --direction)
        ARTIFACT_DIRECTION="$2"
        shift 2
        ;;
      --run-dir)
        RUN_DIR="$2"
        RUN_DIR_SET=1
        shift 2
        ;;
      --fatbin)
        FATBIN="$2"
        shift 2
        ;;
      --ptx)
        PTX="$2"
        shift 2
        ;;
      --kernel)
        KERNEL="$2"
        shift 2
        ;;
      --timeout)
        TIMEOUT_SECONDS="$2"
        shift 2
        ;;
      --export-timeout)
        EXPORT_TIMEOUT_SECONDS="$2"
        shift 2
        ;;
      --artifact-head-dim)
        ARTIFACT_HEAD_DIM="$2"
        ARTIFACT_HEAD_DIM_SET=1
        shift 2
        ;;
      --artifact-head-dim-v)
        ARTIFACT_HEAD_DIM_V="$2"
        ARTIFACT_HEAD_DIM_V_SET=1
        shift 2
        ;;
      --artifact-dtype)
        ARTIFACT_DTYPE="$2"
        ARTIFACT_DTYPE_SET=1
        shift 2
        ;;
      --causal)
        ARTIFACT_CAUSAL=1
        ARTIFACT_CAUSAL_SET=1
        shift
        ;;
      --non-causal)
        ARTIFACT_CAUSAL=0
        ARTIFACT_CAUSAL_SET=1
        shift
        ;;
      --dynamic-smem)
        DYNAMIC_SMEM="$2"
        shift 2
        ;;
      --fa4-python)
        FA4_PYTHON="$2"
        shift 2
        ;;
      --rebuild-launcher)
        REBUILD_LAUNCHER=1
        shift
        ;;
      --run-bwd)
        RUN_BWD=1
        shift
        ;;
      --legacy-coord-pointers)
        LEGACY_COORDS=1
        shift
        ;;
      --list)
        LIST_ONLY=1
        shift
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      --verbose)
        VERBOSE=1
        shift
        ;;
      --cpu-set)
        CPU_SET="$2"
        CPU_SET_EXPLICIT=1
        PIN_CPU=1
        shift 2
        ;;
      --cpus-per-job)
        CPUS_PER_JOB="$2"
        shift 2
        ;;
      --threads-per-job)
        THREADS_PER_JOB="$2"
        shift 2
        ;;
      --no-pin)
        PIN_CPU=0
        NO_PIN_EXPLICIT=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "unknown option: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
  done
  case "$ACTION" in
    run|export|sim)
      ;;
    *)
      echo "unknown action: $ACTION" >&2
      usage >&2
      exit 2
      ;;
  esac
  if [[ "$REBUILD_LAUNCHER" -eq 1 && "$ACTION" == "sim" ]]; then
    echo "--rebuild-launcher is only valid with run or export" >&2
    exit 2
  fi
  case "$LAUNCHER" in
    generated|manual)
      ;;
    *)
      echo "unknown launcher: $LAUNCHER" >&2
      exit 2
      ;;
  esac
  case "$ARTIFACT_DIRECTION" in
    fwd|bwd)
      ;;
    *)
      echo "unknown direction: $ARTIFACT_DIRECTION" >&2
      exit 2
      ;;
  esac
  case "$ARTIFACT_DTYPE" in
    fp16|bf16)
      ;;
    *)
      echo "unknown artifact dtype: $ARTIFACT_DTYPE" >&2
      exit 2
      ;;
  esac
  if [[ "$CPU_SET_EXPLICIT" -eq 1 && "$NO_PIN_EXPLICIT" -eq 1 ]]; then
    echo "--cpu-set and --no-pin are mutually exclusive" >&2
    exit 2
  fi
  if [[ -n "$CPUS_PER_JOB" && ! "$CPUS_PER_JOB" =~ ^[1-9][0-9]*$ ]]; then
    echo "--cpus-per-job must be a positive integer" >&2
    exit 2
  fi
  if [[ -n "$THREADS_PER_JOB" && ! "$THREADS_PER_JOB" =~ ^[1-9][0-9]*$ ]]; then
    echo "--threads-per-job must be a positive integer" >&2
    exit 2
  fi
  if [[ ! "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
    echo "--timeout must be a non-negative integer" >&2
    exit 2
  fi
  if [[ ! "$EXPORT_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "--export-timeout must be a positive integer" >&2
    exit 2
  fi
}

export_thread_limits() {
  export OMP_NUM_THREADS="$THREADS_PER_JOB"
  export OPENBLAS_NUM_THREADS="$THREADS_PER_JOB"
  export MKL_NUM_THREADS="$THREADS_PER_JOB"
  export NUMEXPR_NUM_THREADS="$THREADS_PER_JOB"
}

resolve_cpu_affinity() {
  # A queue worker applies taskset to this entire script and communicates the
  # resolved resources through RUN_QUEUE_*.  Do not create a nested policy.
  if [[ -n "${RUN_QUEUE_JOB_ID:-}" ]]; then
    AFFINITY_MODE="${RUN_QUEUE_AFFINITY_MODE:-unpinned}"
    CPU_SET="${RUN_QUEUE_CPU_SET:-}"
    CPUS_PER_JOB="${RUN_QUEUE_CPUS_PER_JOB:-${CPUS_PER_JOB:-4}}"
    THREADS_PER_JOB="${RUN_QUEUE_THREADS_PER_JOB:-${THREADS_PER_JOB:-$CPUS_PER_JOB}}"
    if [[ "$AFFINITY_MODE" == "unpinned" ]]; then
      PIN_CPU=0
      CPU_SET=""
    elif [[ -z "$CPU_SET" ]]; then
      echo "queue affinity mode '$AFFINITY_MODE' did not provide RUN_QUEUE_CPU_SET" >&2
      exit 2
    else
      PIN_CPU=1
    fi
    export_thread_limits
    return
  fi

  if [[ "$PIN_CPU" -eq 0 ]]; then
    AFFINITY_MODE="unpinned"
    CPU_SET=""
    CPUS_PER_JOB="${CPUS_PER_JOB:-4}"
    THREADS_PER_JOB="${THREADS_PER_JOB:-$CPUS_PER_JOB}"
    export_thread_limits
    return
  fi

  if [[ "${FA4_AFFINITY_APPLIED:-0}" == "1" ]]; then
    AFFINITY_MODE="${AFFINITY_MODE:-auto}"
    CPUS_PER_JOB="${CPUS_PER_JOB:-4}"
    THREADS_PER_JOB="${THREADS_PER_JOB:-$CPUS_PER_JOB}"
    export_thread_limits
    return
  fi

  if ! command -v taskset >/dev/null 2>&1; then
    echo "taskset is required for CPU affinity; use --no-pin to disable it" >&2
    exit 2
  fi

  local selected_width
  if [[ -n "$CPU_SET" ]]; then
    AFFINITY_MODE="manual"
    if ! selected_width="$(taskset -c "$CPU_SET" nproc 2>/dev/null)"; then
      echo "invalid or unavailable --cpu-set: $CPU_SET" >&2
      exit 2
    fi
    if [[ -z "$CPUS_PER_JOB" ]]; then
      CPUS_PER_JOB="$selected_width"
    elif [[ "$CPUS_PER_JOB" -ne "$selected_width" ]]; then
      echo "--cpus-per-job=$CPUS_PER_JOB does not match --cpu-set width $selected_width" >&2
      exit 2
    fi
  else
    AFFINITY_MODE="auto"
    CPUS_PER_JOB="${CPUS_PER_JOB:-4}"
    local affinity_helper="$ROOT_DIR/tests/scripts/cpu_affinity.py"
    if [[ ! -f "$affinity_helper" ]]; then
      echo "CPU affinity helper not found: $affinity_helper" >&2
      exit 2
    fi
    if ! CPU_SET="$(python3 "$affinity_helper" --workers 1 --cpus-per-job "$CPUS_PER_JOB")"; then
      echo "failed to select $CPUS_PER_JOB idle physical CPUs" >&2
      exit 2
    fi
    if [[ -z "$CPU_SET" ]]; then
      echo "CPU affinity helper returned an empty CPU set" >&2
      exit 2
    fi
  fi

  THREADS_PER_JOB="${THREADS_PER_JOB:-$CPUS_PER_JOB}"
  export_thread_limits
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return
  fi

  export FA4_AFFINITY_APPLIED=1
  export FA4_AFFINITY_MODE="$AFFINITY_MODE"
  export FA4_CPU_SET="$CPU_SET"
  export FA4_CPUS_PER_JOB="$CPUS_PER_JOB"
  export FA4_THREADS_PER_JOB="$THREADS_PER_JOB"
  exec taskset -c "$CPU_SET" "$SCRIPT_DIR/run_fa4_b200_cases.sh" "${ORIGINAL_ARGV[@]:1}"
}

ceil_div() {
  local value="$1"
  local divisor="$2"
  echo $(((value + divisor - 1) / divisor))
}

case_table() {
  if [[ ! -f "$CASES_FILE" ]]; then
    echo "case table not found: $CASES_FILE" >&2
    exit 2
  fi
  if [[ -n "$CASE_NAME" ]]; then
    awk -F, -v name="$CASE_NAME" '
      /^[[:space:]]*(#|$)/ { next }
      $2 == name {
        first = 1
        for (i = 2; i <= NF; ++i) {
          printf "%s%s", first ? "" : ",", $i
          first = 0
        }
        printf "\n"
      }
    ' "$CASES_FILE"
    return
  fi
  case "$SUITE" in
    smoke|small|medium|large)
      awk -F, -v suite="$SUITE" '
        /^[[:space:]]*(#|$)/ { next }
        $1 == suite {
          first = 1
          for (i = 2; i <= NF; ++i) {
            printf "%s%s", first ? "" : ",", $i
            first = 0
          }
          printf "\n"
        }
      ' "$CASES_FILE"
      ;;
    all)
      awk -F, '
        /^[[:space:]]*(#|$)/ { next }
        {
          first = 1
          for (i = 2; i <= NF; ++i) {
            printf "%s%s", first ? "" : ",", $i
            first = 0
          }
          printf "\n"
        }
      ' "$CASES_FILE"
      ;;
    *)
      echo "unknown suite: $SUITE" >&2
      exit 2
      ;;
  esac
}

load_selected_cases() {
  local rows
  rows="$(case_table)"
  SELECTED_CASE_ROWS=()
  while IFS= read -r row; do
    [[ -n "$row" ]] || continue
    SELECTED_CASE_ROWS+=("$row")
  done <<<"$rows"
  if [[ "${#SELECTED_CASE_ROWS[@]}" -eq 0 ]]; then
    if [[ -n "$CASE_NAME" ]]; then
      echo "FA4 case not found: $CASE_NAME" >&2
    else
      echo "no FA4 cases selected for suite: $SUITE" >&2
    fi
    exit 2
  fi
}

resolve_artifact_specialization() {
  if [[ -n "$CASE_NAME" ]]; then
    local name batch seqlen_q seqlen_k heads head_dim head_dim_v dtype causal
    IFS=',' read -r name batch seqlen_q seqlen_k heads head_dim head_dim_v dtype causal \
      <<<"${SELECTED_CASE_ROWS[0]}"
    if [[ "$ARTIFACT_HEAD_DIM_SET" -eq 0 ]]; then
      ARTIFACT_HEAD_DIM="$head_dim"
    fi
    if [[ "$ARTIFACT_HEAD_DIM_V_SET" -eq 0 ]]; then
      ARTIFACT_HEAD_DIM_V="$head_dim_v"
    fi
    if [[ "$ARTIFACT_DTYPE_SET" -eq 0 ]]; then
      ARTIFACT_DTYPE="$dtype"
    fi
    if [[ "$ARTIFACT_CAUSAL_SET" -eq 0 ]]; then
      case "$causal" in
        true|1|yes)
          ARTIFACT_CAUSAL=1
          ;;
        false|0|no)
          ARTIFACT_CAUSAL=0
          ;;
        *)
          echo "invalid causal field for $name: $causal" >&2
          exit 2
          ;;
      esac
    fi
  elif [[ "$ARTIFACT_HEAD_DIM_V_SET" -eq 0 ]]; then
    ARTIFACT_HEAD_DIM_V="$ARTIFACT_HEAD_DIM"
  fi
}

resolve_run_dir() {
  if [[ "$RUN_DIR_SET" -eq 1 ]]; then
    return
  fi
  local selection="suite-$SUITE"
  if [[ -n "$CASE_NAME" ]]; then
    selection="$CASE_NAME"
  fi
  RUN_DIR="$ROOT_DIR/temp/fa4-b200-runs/$selection/$CONFIG"
}

stage() {
  local label="$1"
  shift
  local message="$*"
  if [[ "$COLOR_ENABLED" -eq 0 ]]; then
    printf '[%-7s] %s\n' "$label" "$message"
    return
  fi

  local label_color="$COLOR_BRIGHT_CYAN"
  case "$label" in
    PLAN|MANIFEST|LOG)
      label_color="$COLOR_DIM"
      ;;
    EXPORT|ENV|CONFIG|BUILD)
      label_color="$COLOR_BRIGHT_CYAN"
      ;;
    SIM)
      label_color="$COLOR_BRIGHT_MAGENTA"
      ;;
    BWD)
      label_color="$COLOR_BRIGHT_YELLOW"
      ;;
    SUMMARY)
      label_color="$COLOR_BRIGHT_WHITE"
      ;;
  esac

  local state="${message%% *}"
  local rest="${message#"$state"}"
  local state_color=""
  case "$state" in
    PASS|CACHED|REUSE)
      state_color="$COLOR_BRIGHT_GREEN"
      ;;
    FAIL|ERROR)
      state_color="$COLOR_BRIGHT_RED"
      ;;
    SKIP|STALE|STOP)
      state_color="$COLOR_BRIGHT_YELLOW"
      ;;
    START)
      state_color="$COLOR_BRIGHT_CYAN"
      ;;
    PLAN)
      state_color="$COLOR_DIM"
      ;;
  esac

  if [[ -n "$state_color" ]]; then
    printf '%s[%-7s]%s %s%s%s%s\n' \
      "$label_color" "$label" "$COLOR_RESET" \
      "$state_color" "$state" "$COLOR_RESET" "$rest"
  else
    printf '%s[%-7s]%s %s\n' \
      "$label_color" "$label" "$COLOR_RESET" "$message"
  fi
}

print_command() {
  if [[ "$COLOR_ENABLED" -eq 1 ]]; then
    printf '          %scommand:%s' "$COLOR_DIM" "$COLOR_RESET"
  else
    printf '          command:'
  fi
  printf ' %q' "$@"
  printf '\n'
}

warning_count() {
  local log="$1"
  if [[ ! -f "$log" ]]; then
    echo 0
    return
  fi
  awk '
    {
      lower = tolower($0)
      if ($0 !~ /^[[:space:]]*-/ &&
          (lower ~ /warning:/ || lower ~ /warning --/ || lower ~ /deprecated/)) {
        ++count
      }
    }
    END { print count + 0 }
  ' "$log"
}

last_sim_stat() {
  local log="$1"
  local key="$2"
  awk -F= -v key="$key" '
    $1 ~ ("^" key "[[:space:]]*$") { value=$2 }
    END { gsub(/[[:space:]]/, "", value); print value }
  ' "$log"
}

run_logged() {
  local label="$1"
  local log="$2"
  local description="$3"
  shift 3
  stage "$label" "START $description"
  if [[ "$VERBOSE" -eq 1 || "$DRY_RUN" -eq 1 ]]; then
    print_command "$@"
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    stage "$label" "PLAN  log=$log"
    return 0
  fi

  mkdir -p "$(dirname "$log")"
  local status
  set +e
  if [[ "$VERBOSE" -eq 1 ]]; then
    "$@" 2>&1 | tee "$log"
    status="${PIPESTATUS[0]}"
  else
    "$@" >"$log" 2>&1
    status=$?
  fi
  set -e
  if [[ "$status" -ne 0 ]]; then
    stage "$label" "FAIL  exit=$status log=$log"
    if [[ "$VERBOSE" -eq 0 ]]; then
      echo "---------- last 40 log lines ----------" >&2
      tail -n 40 "$log" >&2 || true
      echo "---------------------------------------" >&2
    fi
    return "$status"
  fi
  stage "$label" "PASS  warnings=$(warning_count "$log") log=$log"
}

clean_export_ld_library_path() {
  local entries=()
  local entry
  local sim_root="${GPGPUSIM_ROOT:-$ROOT_DIR}"
  IFS=':' read -r -a entries <<<"${LD_LIBRARY_PATH:-}"
  local kept=()
  declare -A seen=()
  for entry in "${entries[@]}"; do
    [[ -n "$entry" ]] || continue
    if [[ "$entry" == "$ROOT_DIR"/lib/* || "$entry" == "$sim_root"/lib/* ]]; then
      continue
    fi
    if [[ -n "${seen[$entry]:-}" ]]; then
      continue
    fi
    seen["$entry"]=1
    kept+=("$entry")
  done
  local joined=""
  for entry in "${kept[@]}"; do
    if [[ -n "$joined" ]]; then
      joined+=":"
    fi
    joined+="$entry"
  done
  printf '%s\n' "$joined"
}

setup_sim_env() {
  if [[ -z "${CUDA_INSTALL_PATH:-}" && -d /usr/local/cuda-12.8 ]]; then
    export CUDA_INSTALL_PATH=/usr/local/cuda-12.8
  fi
  local fa4_cu13_root="${FA4_CU13_ROOT:-}"
  if [[ -z "$fa4_cu13_root" ]]; then
    fa4_cu13_root="$(default_fa4_cu13_root || true)"
  fi
  if [[ (-z "${PTXAS_CUDA_INSTALL_PATH:-}" ||
         "${PTXAS_CUDA_INSTALL_PATH:-}" == "${CUDA_INSTALL_PATH:-}") &&
        -x "$fa4_cu13_root/bin/ptxas" ]]; then
    export PTXAS_CUDA_INSTALL_PATH="$fa4_cu13_root"
  fi
  if [[ -z "${CUDA_INSTALL_PATH:-}" ]]; then
    echo "CUDA_INSTALL_PATH is not set" >&2
    exit 2
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    export GPGPUSIM_ROOT="${GPGPUSIM_ROOT:-$ROOT_DIR}"
    export GPGPUSIM_CONFIG="${GPGPUSIM_CONFIG:-<set-by-setup_environment>}"
    stage "ENV" "PLAN  source $ROOT_DIR/setup_environment"
    return
  fi
  if [[ -z "${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" ]]; then
    # shellcheck disable=SC1091
    local setup_log="$LOG_DIR/environment.log"
    local status
    stage "ENV" "START source setup_environment"
    set +u
    set +e
    source "$ROOT_DIR/setup_environment" >"$setup_log" 2>&1
    status=$?
    set -e
    set -u
    if [[ "$status" -ne 0 ]]; then
      stage "ENV" "FAIL  exit=$status log=$setup_log"
      tail -n 40 "$setup_log" >&2 || true
      return "$status"
    fi
    if [[ "$VERBOSE" -eq 1 ]]; then
      cat "$setup_log"
    fi
    stage "ENV" "PASS  GPGPUSIM_CONFIG=$GPGPUSIM_CONFIG log=$setup_log"
  else
    stage "ENV" "REUSE GPGPUSIM_CONFIG=${GPGPUSIM_CONFIG:-unknown}"
  fi
}

sync_config() {
  local config_dir="$ROOT_DIR/configs/$CONFIG"
  if [[ ! -f "$config_dir/gpgpusim.config" ]]; then
    echo "config not found: $CONFIG" >&2
    exit 2
  fi
  stage "CONFIG" "START $CONFIG -> $RUN_DIR"
  mkdir -p "$RUN_DIR"
  if [[ "$VERBOSE" -eq 1 || "$DRY_RUN" -eq 1 ]]; then
    print_command cp "$config_dir/gpgpusim.config" "$RUN_DIR/gpgpusim.config"
  fi
  if [[ "$DRY_RUN" -eq 0 ]]; then
    cp "$config_dir/gpgpusim.config" "$RUN_DIR/gpgpusim.config"
  fi
  shopt -s nullglob
  local icnt_files=("$config_dir"/*.icnt)
  if [[ "${#icnt_files[@]}" -gt 0 ]]; then
    if [[ "$VERBOSE" -eq 1 || "$DRY_RUN" -eq 1 ]]; then
      print_command cp "${icnt_files[@]}" "$RUN_DIR/"
    fi
    if [[ "$DRY_RUN" -eq 0 ]]; then
      cp "${icnt_files[@]}" "$RUN_DIR/"
      local icnt_file
      for icnt_file in "${icnt_files[@]}"; do
        SYNCED_ICNT_FILES+=("$RUN_DIR/$(basename "$icnt_file")")
      done
    fi
  fi
  shopt -u nullglob
  if [[ "$DRY_RUN" -eq 1 ]]; then
    stage "CONFIG" "PLAN  config=$CONFIG"
  else
    CONFIG_SYNCED=1
    stage "CONFIG" "PASS  config=$RUN_DIR/gpgpusim.config"
  fi
}

detect_kernel() {
  if [[ -n "$KERNEL" ]]; then
    return
  fi
  if [[ ! -f "$PTX" ]]; then
    echo "cannot auto-detect kernel without PTX: $PTX" >&2
    exit 2
  fi
  KERNEL="$(awk '/^\.visible[[:space:]]+\.entry/ { split($3, a, "("); print a[1]; exit }' "$PTX")"
  if [[ -z "$KERNEL" ]]; then
    echo "failed to detect kernel name from $PTX" >&2
    exit 2
  fi
}

build_harness() {
  local src="$SCRIPT_DIR/fa4_b200_driver_harness.cc"
  local out="$RUN_DIR/fa4_b200_driver_harness"
  local cuda_lib="$CUDA_INSTALL_PATH/lib64"
  local cuda_stub="$CUDA_INSTALL_PATH/lib64/stubs"
  local lib_args=("-L$cuda_lib")
  if [[ -d "$cuda_stub" ]]; then
    lib_args+=("-L$cuda_stub")
  fi
  run_logged "BUILD" "$LOG_DIR/build.log" "manual host harness" \
    g++ -std=c++17 -O2 "$src" -I"$CUDA_INSTALL_PATH/include" \
      "${lib_args[@]}" -Wl,-rpath,"$cuda_lib" -lcudart -lcuda -o "$out"
}

default_fa4_python() {
  if [[ -n "$FA4_PYTHON" ]]; then
    echo "$FA4_PYTHON"
  elif [[ -x "$ROOT_DIR/../fa4-env-cu133/bin/python" ]]; then
    echo "$ROOT_DIR/../fa4-env-cu133/bin/python"
  else
    echo python3
  fi
}

default_fa4_cu13_root() {
  local python_bin
  python_bin="$(default_fa4_python)"
  if ! command -v "$python_bin" >/dev/null 2>&1 &&
      [[ ! -x "$python_bin" ]]; then
    return 1
  fi
  "$python_bin" <<'PY'
import sysconfig
from pathlib import Path

root = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13"
ptxas = root / "bin" / "ptxas"
if ptxas.exists():
    print(root)
else:
    raise SystemExit(1)
PY
}

generated_launcher_metadata_matches() {
  [[ -f "$GENERATED_METADATA" ]] || return 1
  python3 - "$GENERATED_METADATA" "$ARTIFACT_DIRECTION" "$ARTIFACT_HEAD_DIM" \
    "$ARTIFACT_HEAD_DIM_V" "$ARTIFACT_DTYPE" "$ARTIFACT_CAUSAL" <<'PY'
import json
import sys

metadata_path, direction, head_dim, head_dim_v, dtype, causal = sys.argv[1:7]
expected_causal = causal == "1"
try:
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    shape = metadata.get("shape", {})
    ok = (
        metadata.get("direction", "fwd") == direction
        and int(shape.get("head_dim", -1)) == int(head_dim)
        and int(shape.get("head_dim_v", -1)) == int(head_dim_v)
        and shape.get("dtype") == dtype
        and isinstance(shape.get("causal"), bool)
        and shape.get("causal") is expected_causal
    )
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

generated_launcher_artifacts_current() {
  local export_script="$1"
  [[ -f "$GENERATED_HEADER" &&
     -f "$GENERATED_OBJECT" &&
     -f "$GENERATED_PTX" &&
     -f "$GENERATED_METADATA" ]] || return 1
  [[ "$GENERATED_HEADER" -nt "$export_script" &&
     "$GENERATED_OBJECT" -nt "$export_script" &&
     "$GENERATED_PTX" -nt "$export_script" &&
     "$GENERATED_METADATA" -nt "$export_script" ]] || return 1
  generated_launcher_metadata_matches
}

export_generated_launcher() {
  FA4_PYTHON="$(default_fa4_python)"
  if ! command -v "$FA4_PYTHON" >/dev/null 2>&1 &&
      [[ ! -x "$FA4_PYTHON" ]]; then
    echo "FA4 python not found or not executable: $FA4_PYTHON" >&2
    exit 2
  fi
  local export_script="$SCRIPT_DIR/export_fa4_b200_launcher.py"
  local dump_dir="$RUN_DIR/${GENERATED_EXPORT_NAME}_dump"

  if [[ "$REBUILD_LAUNCHER" -eq 0 ]] &&
      generated_launcher_artifacts_current "$export_script"; then
    stage "EXPORT" "CACHED metadata=$GENERATED_METADATA"
    return
  fi
  if [[ "$REBUILD_LAUNCHER" -eq 0 && -f "$GENERATED_METADATA" ]]; then
    stage "EXPORT" "STALE cache is old or does not match the requested specialization"
  fi

  local export_args=(
    "$FA4_PYTHON" "$export_script"
    --export-dir "$RUN_DIR"
    --export-name "$GENERATED_EXPORT_NAME"
    --direction "$ARTIFACT_DIRECTION"
    --dump-dir "$dump_dir"
    --batch 1
    --seqlen-q 128
    --seqlen-k 128
    --heads 2
    --head-dim "$ARTIFACT_HEAD_DIM"
    --head-dim-v "$ARTIFACT_HEAD_DIM_V"
    --dtype "$ARTIFACT_DTYPE"
  )
  if [[ "$ARTIFACT_CAUSAL" -eq 0 ]]; then
    export_args+=(--non-causal)
  fi
  if [[ "$ARTIFACT_DIRECTION" == "bwd" ]]; then
    # Current simulator bring-up intentionally excludes cta_group::2.
    export_args+=(--disable-2cta)
  fi
  local export_ld_library_path
  export_ld_library_path="$(clean_export_ld_library_path)"
  run_logged "EXPORT" "$LOG_DIR/export.log" \
    "CuTe DSL $ARTIFACT_DIRECTION $ARTIFACT_DTYPE d=$ARTIFACT_HEAD_DIM/$ARTIFACT_HEAD_DIM_V" \
    timeout "$EXPORT_TIMEOUT_SECONDS" \
      env "LD_LIBRARY_PATH=$export_ld_library_path" "${export_args[@]}"
}

build_generated_runner() {
  local src="$SCRIPT_DIR/fa4_b200_cute_launcher_runner.cc"
  local out="$RUN_DIR/fa4_b200_cute_launcher_runner"
  local sim_lib="$GPGPUSIM_ROOT/lib/$GPGPUSIM_CONFIG"
  local cuda_lib="$CUDA_INSTALL_PATH/lib64"
  local cuda_stub="$CUDA_INSTALL_PATH/lib64/stubs"
  if [[ "$DRY_RUN" -eq 0 && ! -d "$sim_lib" ]]; then
    echo "GPGPU-Sim runtime library directory not found: $sim_lib" >&2
    exit 2
  fi
  local lib_args=("-L$sim_lib" "-L$cuda_lib")
  if [[ -d "$cuda_stub" ]]; then
    lib_args+=("-L$cuda_stub")
  fi
  local runner_def="-DFA4_B200_RUNNER_BWD=0"
  if [[ "$ARTIFACT_DIRECTION" == "bwd" ]]; then
    runner_def="-DFA4_B200_RUNNER_BWD=1"
  fi
  run_logged "BUILD" "$LOG_DIR/build.log" "generated host runner" \
    g++ -std=c++17 -O2 "$src" "$GENERATED_OBJECT" \
      "$runner_def" \
      -I"$RUN_DIR" -I"$CUDA_INSTALL_PATH/include" \
      "${lib_args[@]}" -Wl,-rpath,"$sim_lib" -Wl,-rpath,"$cuda_lib" \
      -lcudart -lcuda -ldl -pthread -o "$out"
}

run_case() {
  local name="$1"
  local batch="$2"
  local seqlen_q="$3"
  local seqlen_k="$4"
  local heads="$5"
  local head_dim="$6"
  local head_dim_v="$7"
  local dtype="$8"
  local causal="$9"

  local m_blocks
  m_blocks="$(ceil_div "$seqlen_q" 128)"
  local grid_x=$((m_blocks * heads * batch))
  local log="$LOG_DIR/cases/${name}.log"

  local case_causal
  case "$causal" in
    true|1|yes)
      case_causal=1
      ;;
    false|0|no)
      case_causal=0
      ;;
    *)
      echo "invalid causal field for $name: $causal" >&2
      exit 2
      ;;
  esac

  if [[ "$head_dim" != "$ARTIFACT_HEAD_DIM" ||
        "$head_dim_v" != "$ARTIFACT_HEAD_DIM_V" ||
        "$dtype" != "$ARTIFACT_DTYPE" ||
        "$case_causal" != "$ARTIFACT_CAUSAL" ]]; then
    local case_mode="non-causal"
    local artifact_mode="non-causal"
    if [[ "$case_causal" -eq 1 ]]; then
      case_mode="causal"
    fi
    if [[ "$ARTIFACT_CAUSAL" -eq 1 ]]; then
      artifact_mode="causal"
    fi
    local reason="case requires ${dtype} d=${head_dim}/${head_dim_v} ${case_mode};"
    reason+=" artifact is ${ARTIFACT_DTYPE}"
    reason+=" d=${ARTIFACT_HEAD_DIM}/${ARTIFACT_HEAD_DIM_V} ${artifact_mode}"
    if [[ -n "$CASE_NAME" ]]; then
      stage "SIM" "FAIL  $name: $reason"
      CASE_RESULTS+=("$name|fail|")
      ((++FAIL_COUNT))
      return 2
    fi
    stage "SIM" "SKIP  $name: $reason"
    CASE_RESULTS+=("$name|skip|")
    ((++SKIP_COUNT))
    return 0
  fi

  local cmd=()
  if [[ "$LAUNCHER" == "generated" ]]; then
    cmd=(
      "$RUN_DIR/fa4_b200_cute_launcher_runner"
      --batch "$batch"
      --seqlen-q "$seqlen_q"
      --seqlen-k "$seqlen_k"
      --heads "$heads"
      --head-dim "$head_dim"
      --head-dim-v "$head_dim_v"
      --dtype "$dtype"
    )
    if [[ "$case_causal" -eq 1 ]]; then
      cmd+=(--causal)
    else
      cmd+=(--non-causal)
    fi
  else
    cmd=(
      "$RUN_DIR/fa4_b200_driver_harness"
      "$FATBIN"
      "$KERNEL"
      --batch "$batch"
      --seqlen-q "$seqlen_q"
      --seqlen-k "$seqlen_k"
      --heads "$heads"
      --head-dim "$head_dim"
      --head-dim-v "$head_dim_v"
      --dtype "$dtype"
      --grid-x "$grid_x"
      --grid-y 1
      --grid-z 1
      --dynamic-smem "$DYNAMIC_SMEM"
    )
    if [[ "$LEGACY_COORDS" -eq 1 ]]; then
      cmd+=(--legacy-coord-pointers)
    fi
  fi
  ((++RUN_COUNT))
  stage "SIM" "START $name B=$batch Sq=$seqlen_q Sk=$seqlen_k H=$heads grid_x=$grid_x cpu=${CPU_SET:-unbound}"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    if [[ "$LAUNCHER" == "generated" ]]; then
      printf '          command: GPGPUSIM_CUDA_LIBRARY_PTX=%q' "$GENERATED_PTX"
    else
      printf '          command:'
    fi
    printf ' %q' "${cmd[@]}"
    printf '\n'
    stage "SIM" "PLAN  $name log=$log"
    CASE_RESULTS+=("$name|planned|")
    return 0
  fi

  mkdir -p "$(dirname "$log")"
  local status
  set +e
  if [[ "$VERBOSE" -eq 1 ]]; then
    if [[ "$LAUNCHER" == "generated" ]]; then
      (
        cd "$RUN_DIR" || exit 1
        export GPGPUSIM_CUDA_LIBRARY_PTX="$GENERATED_PTX"
        timeout "$TIMEOUT_SECONDS" "${cmd[@]}"
      ) 2>&1 | tee "$log"
    else
      (cd "$RUN_DIR" && timeout "$TIMEOUT_SECONDS" "${cmd[@]}") 2>&1 | tee "$log"
    fi
    status="${PIPESTATUS[0]}"
  else
    if [[ "$LAUNCHER" == "generated" ]]; then
      (
        cd "$RUN_DIR" || exit 1
        export GPGPUSIM_CUDA_LIBRARY_PTX="$GENERATED_PTX"
        timeout "$TIMEOUT_SECONDS" "${cmd[@]}"
      ) >"$log" 2>&1
    else
      (cd "$RUN_DIR" && timeout "$TIMEOUT_SECONDS" "${cmd[@]}") >"$log" 2>&1
    fi
    status=$?
  fi
  set -e

  if [[ "$status" -ne 0 ]]; then
    local failure="exit=$status"
    if [[ "$status" -eq 124 ]]; then
      failure="timeout=${TIMEOUT_SECONDS}s"
    fi
    stage "SIM" "FAIL  $name $failure log=$log"
    if [[ "$VERBOSE" -eq 0 ]]; then
      echo "---------- last 60 simulator log lines ----------" >&2
      tail -n 60 "$log" >&2 || true
      echo "-------------------------------------------------" >&2
    fi
    CASE_RESULTS+=("$name|fail|$log")
    ((++FAIL_COUNT))
    return "$status"
  fi

  local cycles instructions ipc check
  cycles="$(last_sim_stat "$log" gpu_tot_sim_cycle)"
  instructions="$(last_sim_stat "$log" gpu_tot_sim_insn)"
  ipc="$(last_sim_stat "$log" gpu_ipc)"
  if [[ "$LAUNCHER" == "manual" ]]; then
    check="input-canary"
  elif [[ "$ARTIFACT_DIRECTION" == "bwd" ]]; then
    check="backward-input-integrity"
  else
    check="full-O-numeric"
  fi
  local result_summary
  result_summary="$name cycles=${cycles:-unknown}"
  result_summary+=" instructions=${instructions:-unknown} ipc=${ipc:-unknown}"
  result_summary+=" check=$check warnings=$(warning_count "$log")"
  stage "SIM" "PASS  $result_summary"
  stage "LOG" "$name -> $log"
  CASE_RESULTS+=("$name|pass|$log")
  ((++PASS_COUNT))
}

require_generated_artifacts() {
  local missing=()
  local path
  for path in "$GENERATED_HEADER" "$GENERATED_OBJECT" "$GENERATED_PTX" "$GENERATED_METADATA"; do
    if [[ ! -f "$path" ]]; then
      missing+=("$path")
    fi
  done
  if [[ "${#missing[@]}" -gt 0 ]]; then
    echo "generated FA4 artifacts are incomplete:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    echo "run the 'export' action first, or use the default 'run' action" >&2
    return 2
  fi
  if ! generated_launcher_metadata_matches; then
    echo "generated artifact specialization does not match the requested case/options" >&2
    echo "check $GENERATED_METADATA or run the 'export' action with matching options" >&2
    return 2
  fi
}

show_plan() {
  local mode="non-causal"
  if [[ "$ARTIFACT_CAUSAL" -eq 1 ]]; then
    mode="causal"
  fi
  local names=()
  local row name rest
  for row in "${SELECTED_CASE_ROWS[@]}"; do
    IFS=',' read -r name rest <<<"$row"
    names+=("$name")
  done
  stage "PLAN" "action=$ACTION launcher=$LAUNCHER config=$CONFIG run_dir=$RUN_DIR"
  stage "PLAN" "artifact=$ARTIFACT_DIRECTION/$ARTIFACT_DTYPE d=$ARTIFACT_HEAD_DIM/$ARTIFACT_HEAD_DIM_V $mode"
  stage "PLAN" "cases=${#SELECTED_CASE_ROWS[@]} [${names[*]}]"
  stage "PLAN" "affinity=$AFFINITY_MODE cpu_set=${CPU_SET:-all-allowed} cpus_per_job=$CPUS_PER_JOB threads_per_job=$THREADS_PER_JOB"
  stage "PLAN" "run_id=$RUN_ID logs=$LOG_DIR"
}

write_run_manifest() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    stage "MANIFEST" "PLAN  $MANIFEST"
    return 0
  fi
  local tool="$SCRIPT_DIR/write_fa4_b200_run_manifest.py"
  local python_bin
  python_bin="$(command -v python3 || true)"
  if [[ -z "$python_bin" ]]; then
    echo "python3 is required to write the FA4 run manifest" >&2
    return 2
  fi
  local causal="false"
  if [[ "$ARTIFACT_CAUSAL" -eq 1 ]]; then
    causal="true"
  fi
  local runner="$RUN_DIR/fa4_b200_driver_harness"
  local manifest_args=(
    "$python_bin" "$tool"
    --output "$ARCHIVE_MANIFEST"
    --repo-root "$ROOT_DIR"
    --run-dir "$RUN_DIR"
    --log-dir "$LOG_DIR"
    --run-id "$RUN_ID"
    --phase "$ACTION"
    --config "$CONFIG"
    --suite "$SUITE"
    --launcher "$LAUNCHER"
    --direction "$ARTIFACT_DIRECTION"
    --head-dim "$ARTIFACT_HEAD_DIM"
    --head-dim-v "$ARTIFACT_HEAD_DIM_V"
    --dtype "$ARTIFACT_DTYPE"
    --causal "$causal"
    --dynamic-smem "$DYNAMIC_SMEM"
    --affinity-mode "$AFFINITY_MODE"
    --cpus-per-job "$CPUS_PER_JOB"
    --threads-per-job "$THREADS_PER_JOB"
    --cases-file "$CASES_FILE"
  )
  if [[ -n "$CASE_NAME" ]]; then
    manifest_args+=(--case "$CASE_NAME")
  fi
  if [[ -n "${FA4_PYTHON:-}" ]]; then
    manifest_args+=(--fa4-python "$FA4_PYTHON")
  fi
  if [[ -n "${CUDA_INSTALL_PATH:-}" ]]; then
    manifest_args+=(--cuda-root "$CUDA_INSTALL_PATH")
  fi
  if [[ -n "${PTXAS_CUDA_INSTALL_PATH:-}" ]]; then
    manifest_args+=(--ptxas-root "$PTXAS_CUDA_INSTALL_PATH")
  fi
  if [[ "$PIN_CPU" -eq 1 ]]; then
    manifest_args+=(--cpu-set "$CPU_SET")
  fi
  if [[ -n "${RUN_QUEUE_JOB_ID:-}" ]]; then
    manifest_args+=(--queue-job-id "$RUN_QUEUE_JOB_ID")
  fi
  if [[ -n "${RUN_QUEUE_SLOT:-}" ]]; then
    manifest_args+=(--queue-slot "$RUN_QUEUE_SLOT")
  fi
  if [[ "$LAUNCHER" == "generated" ]]; then
    runner="$RUN_DIR/fa4_b200_cute_launcher_runner"
    manifest_args+=(
      --metadata "$GENERATED_METADATA"
      --header "$GENERATED_HEADER"
      --object "$GENERATED_OBJECT"
      --ptx "$GENERATED_PTX"
      --cubin "$GENERATED_CUBIN"
    )
  else
    manifest_args+=(--ptx "$PTX" --fatbin "$FATBIN")
  fi
  if [[ "$RUNNER_BUILT" -eq 1 ]]; then
    manifest_args+=(--runner "$runner")
  fi
  if [[ "$CONFIG_SYNCED" -eq 1 ]]; then
    manifest_args+=(--gpgpusim-config "$RUN_DIR/gpgpusim.config")
    local icnt_path
    for icnt_path in "${SYNCED_ICNT_FILES[@]}"; do
      manifest_args+=(--interconnect-config "$icnt_path")
    done
  fi
  local value
  for value in "${CASE_RESULTS[@]}"; do
    manifest_args+=(--result "$value")
  done
  for value in "${ORIGINAL_ARGV[@]}"; do
    manifest_args+=("--invocation-arg=$value")
  done
  "${manifest_args[@]}"
  cp "$ARCHIVE_MANIFEST" "$MANIFEST"
  stage "MANIFEST" "PASS  latest=$MANIFEST record=$ARCHIVE_MANIFEST"
}

run_selected_cases() {
  local row name batch seqlen_q seqlen_k heads head_dim head_dim_v dtype causal
  local status=0
  for row in "${SELECTED_CASE_ROWS[@]}"; do
    IFS=',' read -r name batch seqlen_q seqlen_k heads head_dim head_dim_v dtype causal <<<"$row"
    if run_case "$name" "$batch" "$seqlen_q" "$seqlen_k" "$heads" \
      "$head_dim" "$head_dim_v" "$dtype" "$causal"; then
      :
    else
      status=$?
      break
    fi
  done
  return "$status"
}

main() {
  ORIGINAL_ARGV=("$0" "$@")
  parse_args "$@"
  load_selected_cases
  resolve_artifact_specialization

  if [[ "$LIST_ONLY" -eq 1 ]]; then
    case_table
    exit 0
  fi
  resolve_run_dir
  resolve_cpu_affinity

  if [[ "$ACTION" == "export" && "$LAUNCHER" != "generated" ]]; then
    echo "the export action is only available with --launcher generated" >&2
    exit 2
  fi
  mkdir -p "$RUN_DIR"
  RUN_DIR="$(cd "$RUN_DIR" && pwd)"
  RUN_ID="$(date '+%Y%m%d-%H%M%S')-$$"
  LOG_DIR="$RUN_DIR/logs/$RUN_ID"
  MANIFEST="$RUN_DIR/run-manifest.json"
  ARCHIVE_MANIFEST="$RUN_DIR/manifests/$RUN_ID.json"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$LOG_DIR/cases"
    mkdir -p "$(dirname "$ARCHIVE_MANIFEST")"
  fi
  FATBIN="${FATBIN:-$RUN_DIR/fa4_b200.fatbin}"
  PTX="${PTX:-$RUN_DIR/fa4_b200.1.sm_100a.ptx}"
  GENERATED_HEADER="$RUN_DIR/${GENERATED_EXPORT_NAME}.h"
  GENERATED_OBJECT="$RUN_DIR/${GENERATED_EXPORT_NAME}.o"
  GENERATED_PTX="$RUN_DIR/${GENERATED_EXPORT_NAME}.ptx"
  GENERATED_METADATA="$RUN_DIR/${GENERATED_EXPORT_NAME}.metadata.json"
  GENERATED_CUBIN="$RUN_DIR/${GENERATED_EXPORT_NAME}.cubin"

  if [[ "$LAUNCHER" == "generated" ]]; then
    FA4_PYTHON="$(default_fa4_python)"
  fi
  show_plan

  if [[ "$ACTION" == "run" || "$ACTION" == "export" ]]; then
    if [[ "$LAUNCHER" == "generated" ]]; then
      export_generated_launcher
    fi
  elif [[ "$LAUNCHER" == "generated" && "$DRY_RUN" -eq 0 ]]; then
    require_generated_artifacts
  fi

  if [[ "$LAUNCHER" == "generated" &&
        "$ARTIFACT_DIRECTION" == "bwd" &&
        "$RUN_BWD" -eq 0 &&
        ( "$ACTION" == "run" || "$ACTION" == "sim" ) ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      stage "BWD" "PLAN  export only; launch requires --run-bwd"
    else
      stage "BWD" "STOP  artifact prepared; launch remains opt-in via --run-bwd"
    fi
    write_run_manifest
    exit 0
  fi

  if [[ "$ACTION" == "export" ]]; then
    write_run_manifest
    if [[ "$DRY_RUN" -eq 1 ]]; then
      stage "SUMMARY" "export planned"
    else
      stage "SUMMARY" "export complete; artifacts=$RUN_DIR manifest=$MANIFEST"
    fi
    exit 0
  fi

  setup_sim_env
  sync_config
  if [[ "$LAUNCHER" == "generated" ]]; then
    build_generated_runner
  else
    if [[ "$DRY_RUN" -eq 1 && -z "$KERNEL" && ! -f "$PTX" ]]; then
      KERNEL="<kernel-from-ptx>"
    else
      detect_kernel
    fi
    if [[ "$DRY_RUN" -eq 0 && ! -f "$FATBIN" ]]; then
      echo "fatbin not found: $FATBIN" >&2
      exit 2
    fi
    build_harness
  fi
  if [[ "$DRY_RUN" -eq 0 ]]; then
    RUNNER_BUILT=1
  fi

  local run_status=0
  if run_selected_cases; then
    :
  else
    run_status=$?
  fi
  local manifest_status=0
  if write_run_manifest; then
    :
  else
    manifest_status=$?
  fi
  local summary="selected=${#SELECTED_CASE_ROWS[@]} runnable=$RUN_COUNT"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    summary+=" planned=$RUN_COUNT skip=$SKIP_COUNT"
  else
    summary+=" pass=$PASS_COUNT skip=$SKIP_COUNT fail=$FAIL_COUNT"
  fi
  summary+=" manifest=$MANIFEST"
  stage "SUMMARY" "$summary"
  if [[ "$run_status" -ne 0 ]]; then
    return "$run_status"
  fi
  if [[ "$manifest_status" -ne 0 ]]; then
    return "$manifest_status"
  fi
}

main "$@"
