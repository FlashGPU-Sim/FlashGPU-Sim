#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CONFIG="SM100_B200"
SUITE="smoke"
DIRECTION="fwd"
RUN_DIR="$ROOT_DIR/../fa4-b200-suite-matrix"
TIMEOUT_SECONDS=300
DRY_RUN=0
LIST_ONLY=0
REBUILD_LAUNCHER=0
RUN_BWD=0
FA4_PYTHON=""

usage() {
  cat <<'EOF'
usage: tests/dev/fa4/run_fa4_b200_suite_matrix.sh [options]

Run a full FA4 workload suite by grouping
tests/dev/fa4/fa4_b200_cases.csv rows by the generated artifact specialization:
direction, head_dim, head_dim_v, dtype, and causal mode. Each group is delegated
to run_fa4_b200_cases.sh.

options:
  --config NAME          GPGPU-Sim config (default: SM100_B200)
  --suite NAME           smoke | small | medium | large | all (default: smoke)
  --direction fwd|bwd    Generated FA4 kernel direction (default: fwd)
  --run-bwd              Actually launch backward artifacts. By default bwd
                         exports only because BWD TMA descriptors are not
                         fully supported yet.
  --run-dir DIR          Base run directory for grouped sub-runs
  --timeout SECONDS      Per-case timeout passed to run_fa4_b200_cases.sh
  --fa4-python PATH      Python interpreter with FA4 installed
  --rebuild-launcher     Re-export each generated launcher
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

main() {
  parse_args "$@"

  local variants=()
  mapfile -t variants < <(artifact_variants)
  if [[ "${#variants[@]}" -eq 0 ]]; then
    echo "no FA4 variants selected for suite: $SUITE" >&2
    exit 2
  fi

  if [[ "$LIST_ONLY" -eq 1 ]]; then
    local variant
    for variant in "${variants[@]}"; do
      IFS=',' read -r head_dim head_dim_v dtype causal <<<"$variant"
      echo "direction=$DIRECTION suite=$SUITE dtype=$dtype head_dim=$head_dim head_dim_v=$head_dim_v causal=$causal run_dir=$(variant_run_dir "$head_dim" "$head_dim_v" "$dtype" "$causal")"
    done
    exit 0
  fi

  local variant
  for variant in "${variants[@]}"; do
    IFS=',' read -r head_dim head_dim_v dtype causal <<<"$variant"
    local sub_run_dir
    sub_run_dir="$(variant_run_dir "$head_dim" "$head_dim_v" "$dtype" "$causal")"
    local cmd=(
      "$SCRIPT_DIR/run_fa4_b200_cases.sh"
      --config "$CONFIG"
      --suite "$SUITE"
      --direction "$DIRECTION"
      --run-dir "$sub_run_dir"
      --timeout "$TIMEOUT_SECONDS"
      --artifact-head-dim "$head_dim"
      --artifact-head-dim-v "$head_dim_v"
      --artifact-dtype "$dtype"
    )
    case "$causal" in
      true|1|yes)
        cmd+=(--causal)
        ;;
      false|0|no)
        cmd+=(--non-causal)
        ;;
      *)
        echo "invalid causal field: $causal" >&2
        exit 2
        ;;
    esac
    if [[ -n "$FA4_PYTHON" ]]; then
      cmd+=(--fa4-python "$FA4_PYTHON")
    fi
    if [[ "$REBUILD_LAUNCHER" -eq 1 ]]; then
      cmd+=(--rebuild-launcher)
    fi
    if [[ "$RUN_BWD" -eq 1 ]]; then
      cmd+=(--run-bwd)
    fi
    if [[ "$DRY_RUN" -eq 1 ]]; then
      cmd+=(--dry-run)
    fi

    echo "=== FA4 $DIRECTION $SUITE dtype=$dtype head_dim=$head_dim head_dim_v=$head_dim_v causal=$causal ==="
    "${cmd[@]}"
  done
}

main "$@"
