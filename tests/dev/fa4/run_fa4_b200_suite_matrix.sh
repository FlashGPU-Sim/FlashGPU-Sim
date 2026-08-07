#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CONFIG="SM100_B200"
SUITE="smoke"
DIRECTION="fwd"
RUN_DIR="$ROOT_DIR/temp/fa4-b200-suite-matrix"
TIMEOUT_SECONDS=0
DRY_RUN=0
LIST_ONLY=0
REBUILD_LAUNCHER=0
RUN_BWD=0
FA4_PYTHON="${FA4_PYTHON:-}"
VERBOSE=0
MAX_PARALLEL=4
CPUS_PER_JOB="${FA4_CPUS_PER_JOB:-}"
THREADS_PER_JOB="${FA4_THREADS_PER_JOB:-}"
PIN_CPUS=1
CPU_SETS=()
CASE_COMMAND=()

usage() {
  cat <<'EOF'
usage: tests/dev/fa4/run_fa4_b200_suite_matrix.sh [options]

Run every selected row from tests/dev/fa4/fa4_b200_cases.csv. Workloads are
delegated to run_fa4_b200_cases.sh through tests/scripts/run_sim_queue.py;
output directories remain grouped by artifact specialization.

options:
  --config NAME          GPGPU-Sim config (default: SM100_B200)
  --suite NAME           smoke | small | medium | large | all (default: smoke)
  --direction fwd|bwd    Generated FA4 kernel direction (default: fwd)
  --run-bwd              Actually launch backward artifacts. By default bwd
                         exports only because BWD TMA descriptors are not
                         fully supported yet.
  --run-dir DIR          Base run directory for grouped sub-runs
  --timeout SECONDS      Per-case timeout (default: 0, disabled)
  --fa4-python PATH      Python interpreter with FA4 installed
  --rebuild-launcher     Re-export each generated launcher
  --max-parallel N       Maximum concurrent workloads (default: 4)
  --cpus-per-job N       Physical cores per workload (default: 4, or inferred
                         from manual --cpu-sets)
  --threads-per-job N    Thread-library limit (default: resolved CPU count)
  --cpu-sets LIST...     Manually assign one taskset CPU list per worker slot;
                         otherwise distinct idle physical cores are selected
  --no-pin               Do not bind worker slots to CPUs
  --verbose              Stream delegated exporter/build/simulator logs
  --list                 Print grouped artifact variants and exit
  --dry-run              Print delegated commands without running them
  -h, --help             Show this help
EOF
}

parse_args() {
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
      --direction)
        DIRECTION="$2"
        shift 2
        ;;
      --run-dir)
        RUN_DIR="$2"
        shift 2
        ;;
      --timeout)
        TIMEOUT_SECONDS="$2"
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
      --max-parallel)
        MAX_PARALLEL="$2"
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
      --cpu-sets)
        shift
        local before_count="${#CPU_SETS[@]}"
        while [[ $# -gt 0 && "$1" != -* ]]; do
          CPU_SETS+=("$1")
          shift
        done
        if [[ "${#CPU_SETS[@]}" -eq "$before_count" ]]; then
          echo "--cpu-sets requires at least one CPU list" >&2
          exit 2
        fi
        ;;
      --no-pin)
        PIN_CPUS=0
        shift
        ;;
      --verbose)
        VERBOSE=1
        shift
        ;;
      --run-bwd)
        RUN_BWD=1
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

  case "$DIRECTION" in
    fwd|bwd)
      ;;
    *)
      echo "unknown direction: $DIRECTION" >&2
      exit 2
      ;;
  esac
  if [[ ! "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-parallel must be a positive integer" >&2
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
  if [[ "$PIN_CPUS" -eq 0 && "${#CPU_SETS[@]}" -gt 0 ]]; then
    echo "--no-pin and --cpu-sets cannot be used together" >&2
    exit 2
  fi
  if [[ "${#CPU_SETS[@]}" -eq 0 ]]; then
    CPUS_PER_JOB="${CPUS_PER_JOB:-4}"
  fi
  if [[ -n "$CPUS_PER_JOB" ]]; then
    THREADS_PER_JOB="${THREADS_PER_JOB:-$CPUS_PER_JOB}"
  fi
}

artifact_variants() {
  "$SCRIPT_DIR/run_fa4_b200_cases.sh" --suite "$SUITE" --list |
    awk -F, '
      NF {
        key = $6 "," $7 "," $8 "," $9
        if (!seen[key]++) print key
      }
    '
}

variant_run_dir() {
  local head_dim="$1"
  local head_dim_v="$2"
  local dtype="$3"
  local causal="$4"
  local mode="full"
  case "$causal" in
    true|1|yes)
      mode="causal"
      ;;
  esac
  printf '%s/%s_%s_%s_d%s_dv%s_%s' \
    "$RUN_DIR" "$DIRECTION" "$SUITE" "$dtype" "$head_dim" "$head_dim_v" "$mode"
}

selected_case_rows() {
  "$SCRIPT_DIR/run_fa4_b200_cases.sh" --suite "$SUITE" --list
}

case_run_dir() {
  local name="$1"
  local head_dim="$2"
  local head_dim_v="$3"
  local dtype="$4"
  local causal="$5"
  printf '%s/%s' \
    "$(variant_run_dir "$head_dim" "$head_dim_v" "$dtype" "$causal")" \
    "$name"
}

case_command() {
  local name="$1"
  local head_dim="$2"
  local head_dim_v="$3"
  local dtype="$4"
  local causal="$5"
  local sub_run_dir
  sub_run_dir="$(case_run_dir "$name" "$head_dim" "$head_dim_v" "$dtype" "$causal")"

  CASE_COMMAND=(
    "$SCRIPT_DIR/run_fa4_b200_cases.sh"
    run
    --config "$CONFIG"
    --suite "$SUITE"
    --case "$name"
    --direction "$DIRECTION"
    --run-dir "$sub_run_dir"
    --timeout "$TIMEOUT_SECONDS"
  )
  if [[ -n "$FA4_PYTHON" ]]; then
    CASE_COMMAND+=(--fa4-python "$FA4_PYTHON")
  fi
  if [[ "$REBUILD_LAUNCHER" -eq 1 ]]; then
    CASE_COMMAND+=(--rebuild-launcher)
  fi
  if [[ "$RUN_BWD" -eq 1 ]]; then
    CASE_COMMAND+=(--run-bwd)
  fi
  if [[ "$VERBOSE" -eq 1 ]]; then
    CASE_COMMAND+=(--verbose)
  fi
}

shell_join() {
  local result=""
  local value quoted
  for value in "$@"; do
    printf -v quoted '%q' "$value"
    if [[ -n "$result" ]]; then
      result+=" "
    fi
    result+="$quoted"
  done
  printf '%s' "$result"
}

write_queue_jobs() {
  local jobs_file="$1"
  printf 'job_id\texecutable\targs\tfa4_suite\tfa4_direction\tfa4_case\tfa4_config\tfa4_run_dir\n' >"$jobs_file"

  local row name batch seqlen_q seqlen_k heads head_dim head_dim_v dtype causal
  while IFS= read -r row; do
    [[ -n "$row" ]] || continue
    IFS=',' read -r name batch seqlen_q seqlen_k heads head_dim head_dim_v dtype causal <<<"$row"
    case_command "$name" "$head_dim" "$head_dim_v" "$dtype" "$causal"
    local command_args
    command_args="$(shell_join "${CASE_COMMAND[@]:1}")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$name" "${CASE_COMMAND[0]}" "$command_args" "$SUITE" "$DIRECTION" \
      "$name" "$CONFIG" \
      "$(case_run_dir "$name" "$head_dim" "$head_dim_v" "$dtype" "$causal")" \
      >>"$jobs_file"
  done < <(selected_case_rows)
}

show_dry_run() {
  local row name batch seqlen_q seqlen_k heads head_dim head_dim_v dtype causal
  local affinity="auto"
  if [[ "$PIN_CPUS" -eq 0 ]]; then
    affinity="unpinned"
  elif [[ "${#CPU_SETS[@]}" -gt 0 ]]; then
    affinity="manual"
  fi
  printf '[QUEUE  ] PLAN  suite=%s max_parallel=%s affinity=%s cpus_per_job=%s threads_per_job=%s\n' \
    "$SUITE" "$MAX_PARALLEL" "$affinity" "${CPUS_PER_JOB:-infer}" \
    "${THREADS_PER_JOB:-infer}"
  if [[ "${#CPU_SETS[@]}" -gt 0 ]]; then
    printf '[QUEUE  ] CPU sets: %s\n' "${CPU_SETS[*]}"
  fi
  while IFS= read -r row; do
    [[ -n "$row" ]] || continue
    IFS=',' read -r name batch seqlen_q seqlen_k heads head_dim head_dim_v dtype causal <<<"$row"
    case_command "$name" "$head_dim" "$head_dim_v" "$dtype" "$causal"
    printf '[JOB    ] %-32s ' "$name"
    printf '%q ' "${CASE_COMMAND[@]}"
    printf '\n'
  done < <(selected_case_rows)
}

setup_queue_environment() {
  if [[ -z "${CUDA_INSTALL_PATH:-}" && -d /usr/local/cuda-12.8 ]]; then
    export CUDA_INSTALL_PATH=/usr/local/cuda-12.8
  fi
  if [[ -z "${CUDA_INSTALL_PATH:-}" ]]; then
    echo "CUDA_INSTALL_PATH is not set" >&2
    exit 2
  fi
  if [[ -z "${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" ]]; then
    set +u
    # shellcheck disable=SC1091
    source "$ROOT_DIR/setup_environment" >/dev/null
    set -u
  fi
}

run_queue() {
  local queue_id
  queue_id="$(date '+%Y%m%d-%H%M%S')-$$"
  local queue_run_dir="$RUN_DIR/queues/${DIRECTION}_${SUITE}_$queue_id"
  mkdir -p "$queue_run_dir"
  local jobs_file="$queue_run_dir/jobs.tsv"
  write_queue_jobs "$jobs_file"

  local queue_cmd=(
    python3 "$ROOT_DIR/tests/scripts/run_sim_queue.py"
    --root "$ROOT_DIR"
    --run-root "$queue_run_dir"
    --jobs "$jobs_file"
    --config "$CONFIG"
    --max-parallel "$MAX_PARALLEL"
    --timeout 0
    --heartbeat-interval 5
    --cuda-path "$CUDA_INSTALL_PATH"
    --gpgpusim-config "${GPGPUSIM_CONFIG:-gcc-13.3.0/cuda-12080/release}"
  )
  if [[ -n "$CPUS_PER_JOB" ]]; then
    queue_cmd+=(--cpus-per-job "$CPUS_PER_JOB")
  fi
  if [[ -n "$THREADS_PER_JOB" ]]; then
    queue_cmd+=(--threads-per-job "$THREADS_PER_JOB")
  fi
  if [[ "$PIN_CPUS" -eq 0 ]]; then
    queue_cmd+=(--no-pin)
  elif [[ "${#CPU_SETS[@]}" -gt 0 ]]; then
    queue_cmd+=(--cpu-sets "${CPU_SETS[@]}")
  fi

  printf '[QUEUE  ] START suite=%s workers=%s jobs=%s\n' \
    "$SUITE" "$MAX_PARALLEL" "$(($(wc -l <"$jobs_file") - 1))"
  local status=0
  if "${queue_cmd[@]}"; then
    :
  else
    status=$?
  fi

  local summary="$queue_run_dir/status/summary.tsv"
  if [[ -f "$summary" ]]; then
    cp "$summary" "$RUN_DIR/${DIRECTION}_${SUITE}_summary.tsv"
  fi
  printf '%s\n' "$queue_run_dir" >"$RUN_DIR/latest-queue.txt"
  if [[ "$status" -eq 130 || "$status" -eq 143 ]]; then
    printf '[QUEUE  ] INTERRUPTED rc=%s summary=%s\n' "$status" "$summary"
  else
    printf '[QUEUE  ] DONE  rc=%s summary=%s\n' "$status" "$summary"
  fi
  return "$status"
}

main() {
  parse_args "$@"

  local variants=()
  mapfile -t variants < <(artifact_variants)
  if [[ "${#variants[@]}" -eq 0 ]]; then
    echo "no FA4 variants selected for suite: $SUITE" >&2
    exit 2
  fi

  if [[ "$LIST_ONLY" -eq 1 ]]; then
    local variant head_dim head_dim_v dtype causal
    for variant in "${variants[@]}"; do
      IFS=',' read -r head_dim head_dim_v dtype causal <<<"$variant"
      echo "direction=$DIRECTION suite=$SUITE dtype=$dtype head_dim=$head_dim head_dim_v=$head_dim_v causal=$causal run_dir=$(variant_run_dir "$head_dim" "$head_dim_v" "$dtype" "$causal")"
    done
    exit 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    show_dry_run
    exit 0
  fi

  mkdir -p "$RUN_DIR"
  RUN_DIR="$(cd "$RUN_DIR" && pwd)"
  setup_queue_environment
  run_queue
}

main "$@"
