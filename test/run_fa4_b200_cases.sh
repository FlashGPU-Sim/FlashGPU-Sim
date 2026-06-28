#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG="SM100_B200"
SUITE="smoke"
RUN_DIR="$ROOT_DIR/../fa4-b200-sim-run"
LAUNCHER="generated"
FATBIN=""
PTX=""
KERNEL=""
TIMEOUT_SECONDS=300
DRY_RUN=0
LIST_ONLY=0
REBUILD_LAUNCHER=0
LEGACY_COORDS=0
ARTIFACT_HEAD_DIM=64
ARTIFACT_HEAD_DIM_V=""
ARTIFACT_HEAD_DIM_V_SET=0
ARTIFACT_DTYPE="fp16"
DYNAMIC_SMEM=231424
CASES_FILE="$ROOT_DIR/test/fa4_b200_cases.csv"
FA4_PYTHON="${FA4_PYTHON:-}"
GENERATED_EXPORT_NAME="fa4_b200_launcher"
GENERATED_HEADER=""
GENERATED_OBJECT=""
GENERATED_PTX=""
GENERATED_METADATA=""

usage() {
  cat <<'EOF'
usage: test/run_fa4_b200_cases.sh [options]

options:
  --config NAME             GPGPU-Sim config to copy into the FA4 run dir
                            (default: SM100_B200)
  --suite NAME              smoke | paper-prefill | paper-decode | all
                            (default: smoke)
  --launcher NAME           generated | manual
                            (default: generated)
  --run-dir DIR             FA4 artifact/run directory
                            (default: ../fa4-b200-sim-run)
  --fatbin PATH             FA4 fatbin path
  --ptx PATH                PTX path used to auto-detect the kernel name
  --kernel NAME             Kernel symbol name
  --timeout SECONDS         Per-case timeout (default: 300)
  --artifact-head-dim N     Head dimension compiled into the artifact
                            (default: 64)
  --artifact-head-dim-v N   V/O head dimension compiled into the artifact
                            (default: artifact-head-dim)
  --artifact-dtype fp16|bf16
                            Dtype compiled into the artifact (default: fp16)
  --dynamic-smem BYTES      Dynamic shared memory bytes (default: 231424)
  --fa4-python PATH         Python interpreter with FA4 installed
                            (default: ../fa4-env-cu133/bin/python if present)
  --rebuild-launcher        Re-export the generated FA4 C launcher
  --legacy-coord-pointers   Use the first bring-up harness ABI behavior
  --list                    Print selected cases and exit
  --dry-run                 Print commands without running them
  -h, --help                Show this help

CUDA_INSTALL_PATH must point at a CUDA toolkit usable by this GPGPU-Sim build.
For PTX 9.1 / sm_100a artifacts, PTXAS_CUDA_INSTALL_PATH should point at a
CUDA 13 Blackwell-capable ptxas.
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
      --launcher)
        LAUNCHER="$2"
        shift 2
        ;;
      --run-dir)
        RUN_DIR="$2"
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
      --artifact-head-dim)
        ARTIFACT_HEAD_DIM="$2"
        shift 2
        ;;
      --artifact-head-dim-v)
        ARTIFACT_HEAD_DIM_V="$2"
        ARTIFACT_HEAD_DIM_V_SET=1
        shift 2
        ;;
      --artifact-dtype)
        ARTIFACT_DTYPE="$2"
        shift 2
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
  if [[ "$ARTIFACT_HEAD_DIM_V_SET" -eq 0 ]]; then
    ARTIFACT_HEAD_DIM_V="$ARTIFACT_HEAD_DIM"
  fi
  case "$LAUNCHER" in
    generated|manual)
      ;;
    *)
      echo "unknown launcher: $LAUNCHER" >&2
      exit 2
      ;;
  esac
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
  case "$SUITE" in
    smoke|paper-prefill|paper-decode)
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

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
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
  if [[ -z "${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" ]]; then
    # shellcheck disable=SC1091
    set +u
    source "$ROOT_DIR/setup_environment" >/tmp/fa4_b200_cases_setup.log
    set -u
  fi
}

sync_config() {
  local config_dir="$ROOT_DIR/configs/$CONFIG"
  if [[ ! -f "$config_dir/gpgpusim.config" ]]; then
    echo "config not found: $CONFIG" >&2
    exit 2
  fi
  mkdir -p "$RUN_DIR"
  run_cmd cp "$config_dir/gpgpusim.config" "$RUN_DIR/gpgpusim.config"
  shopt -s nullglob
  local icnt_files=("$config_dir"/*.icnt)
  if [[ "${#icnt_files[@]}" -gt 0 ]]; then
    run_cmd cp "${icnt_files[@]}" "$RUN_DIR/"
  fi
  shopt -u nullglob
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
  local src="$ROOT_DIR/test/fa4_b200_driver_harness.cc"
  local out="$RUN_DIR/fa4_b200_driver_harness"
  local cuda_lib="$CUDA_INSTALL_PATH/lib64"
  local cuda_stub="$CUDA_INSTALL_PATH/lib64/stubs"
  local lib_args=("-L$cuda_lib")
  if [[ -d "$cuda_stub" ]]; then
    lib_args+=("-L$cuda_stub")
  fi
  run_cmd g++ -std=c++17 -O2 "$src" -I"$CUDA_INSTALL_PATH/include" \
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
  "$FA4_PYTHON" - "$GENERATED_METADATA" "$ARTIFACT_HEAD_DIM" \
    "$ARTIFACT_HEAD_DIM_V" "$ARTIFACT_DTYPE" <<'PY'
import json
import sys

metadata_path, head_dim, head_dim_v, dtype = sys.argv[1:5]
try:
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    shape = metadata.get("shape", {})
    ok = (
        int(shape.get("head_dim", -1)) == int(head_dim)
        and int(shape.get("head_dim_v", -1)) == int(head_dim_v)
        and shape.get("dtype") == dtype
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
  local export_script="$ROOT_DIR/test/export_fa4_b200_launcher.py"
  local dump_dir="$RUN_DIR/${GENERATED_EXPORT_NAME}_dump"

  if [[ "$REBUILD_LAUNCHER" -eq 0 ]] &&
      generated_launcher_artifacts_current "$export_script"; then
    return
  fi
  if [[ "$REBUILD_LAUNCHER" -eq 0 && -f "$GENERATED_METADATA" ]]; then
    echo "generated launcher cache is stale or mismatched; rebuilding"
  fi

  run_cmd "$FA4_PYTHON" "$export_script" \
    --export-dir "$RUN_DIR" \
    --export-name "$GENERATED_EXPORT_NAME" \
    --dump-dir "$dump_dir" \
    --batch 1 \
    --seqlen-q 128 \
    --seqlen-k 128 \
    --heads 2 \
    --head-dim "$ARTIFACT_HEAD_DIM" \
    --head-dim-v "$ARTIFACT_HEAD_DIM_V" \
    --dtype "$ARTIFACT_DTYPE"
}

build_generated_runner() {
  local src="$ROOT_DIR/test/fa4_b200_cute_launcher_runner.cc"
  local out="$RUN_DIR/fa4_b200_cute_launcher_runner"
  local sim_lib="$GPGPUSIM_ROOT/lib/$GPGPUSIM_CONFIG"
  local cuda_lib="$CUDA_INSTALL_PATH/lib64"
  local cuda_stub="$CUDA_INSTALL_PATH/lib64/stubs"
  if [[ ! -d "$sim_lib" ]]; then
    echo "GPGPU-Sim runtime library directory not found: $sim_lib" >&2
    exit 2
  fi
  local lib_args=("-L$sim_lib" "-L$cuda_lib")
  if [[ -d "$cuda_stub" ]]; then
    lib_args+=("-L$cuda_stub")
  fi
  run_cmd g++ -std=c++17 -O2 "$src" "$GENERATED_OBJECT" \
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

  local m_blocks
  m_blocks="$(ceil_div "$seqlen_q" 128)"
  local grid_x=$((m_blocks * heads * batch))
  local log="$RUN_DIR/fa4_${CONFIG}_${name}.log"

  if [[ "$head_dim" != "$ARTIFACT_HEAD_DIM" ||
        "$head_dim_v" != "$ARTIFACT_HEAD_DIM_V" ||
        "$dtype" != "$ARTIFACT_DTYPE" ]]; then
    echo "SKIP $name: case requires ${dtype} d=${head_dim}/${head_dim_v}, artifact is ${ARTIFACT_DTYPE} d=${ARTIFACT_HEAD_DIM}/${ARTIFACT_HEAD_DIM_V}"
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

  echo "RUN $name grid_x=$grid_x log=$log"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    if [[ "$LAUNCHER" == "generated" ]]; then
      printf '+ GPGPUSIM_CUDA_LIBRARY_PTX=%q' "$GENERATED_PTX"
    else
      printf '+'
    fi
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi
  if [[ "$LAUNCHER" == "generated" ]]; then
    (
      cd "$RUN_DIR"
      export GPGPUSIM_CUDA_LIBRARY_PTX="$GENERATED_PTX"
      timeout "$TIMEOUT_SECONDS" "${cmd[@]}" >"$log" 2>&1
    )
  else
    (cd "$RUN_DIR" && timeout "$TIMEOUT_SECONDS" "${cmd[@]}" >"$log" 2>&1)
  fi
}

main() {
  parse_args "$@"
  mkdir -p "$RUN_DIR"
  RUN_DIR="$(cd "$RUN_DIR" && pwd)"
  FATBIN="${FATBIN:-$RUN_DIR/fa4_b200.fatbin}"
  PTX="${PTX:-$RUN_DIR/fa4_b200.1.sm_100a.ptx}"
  GENERATED_HEADER="$RUN_DIR/${GENERATED_EXPORT_NAME}.h"
  GENERATED_OBJECT="$RUN_DIR/${GENERATED_EXPORT_NAME}.o"
  GENERATED_PTX="$RUN_DIR/${GENERATED_EXPORT_NAME}.ptx"
  GENERATED_METADATA="$RUN_DIR/${GENERATED_EXPORT_NAME}.metadata.json"

  if [[ "$LIST_ONLY" -eq 1 ]]; then
    case_table
    exit 0
  fi

  sync_config

  if [[ "$DRY_RUN" -eq 1 ]]; then
    if [[ "$LAUNCHER" == "generated" ]]; then
      FA4_PYTHON="$(default_fa4_python)"
    elif [[ -z "$KERNEL" ]]; then
      if [[ -f "$PTX" ]]; then
        detect_kernel
      else
        KERNEL="<kernel>"
      fi
    fi
  else
    if [[ "$LAUNCHER" == "generated" ]]; then
      export_generated_launcher
    fi
    setup_sim_env
    if [[ "$LAUNCHER" == "generated" ]]; then
      build_generated_runner
    else
      detect_kernel
      build_harness

      if [[ ! -f "$FATBIN" ]]; then
        echo "fatbin not found: $FATBIN" >&2
        exit 2
      fi
    fi
  fi

  echo "config=$CONFIG suite=$SUITE launcher=$LAUNCHER run_dir=$RUN_DIR"
  if [[ "$LAUNCHER" == "generated" ]]; then
    echo "generated_header=$GENERATED_HEADER"
    echo "generated_object=$GENERATED_OBJECT"
    echo "generated_ptx=$GENERATED_PTX"
    echo "generated_metadata=$GENERATED_METADATA"
  else
    echo "fatbin=$FATBIN"
    echo "kernel=$KERNEL"
  fi

  local row
  while IFS= read -r row; do
    [[ -n "$row" ]] || continue
    IFS=',' read -r name batch seqlen_q seqlen_k heads head_dim head_dim_v dtype <<<"$row"
    run_case "$name" "$batch" "$seqlen_q" "$seqlen_k" "$heads" "$head_dim" "$head_dim_v" "$dtype"
  done < <(case_table)
}

main "$@"
