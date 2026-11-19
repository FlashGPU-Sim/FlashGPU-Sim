#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd -P)"
CONFIG_DIR="${REPO_ROOT}/configs/SM89_RTX4060"
OUT_ROOT="${REPO_ROOT}/reference/SM89_RTX4060/gemm"
CASES_CSV="${GEMM_CASES_CSV:-${SCRIPT_DIR}/gemm_cases.csv}"
usage() { echo "Usage: $0 {capture|ncu|sim|all} CASE_ID fp16"; }
if [[ $# -ne 3 || "$3" != fp16 ]]; then usage >&2; exit 2; fi
ACTION="$1"; CASE_ID="$2"; DTYPE="$3"
if [[ ( "$ACTION" == capture || "$ACTION" == ncu ) && -v GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN ]]; then
  echo "Use a fresh native CUDA shell for capture and NCU; simulator setup is already active." >&2
  exit 1
fi
split="$(awk -F, -v id="$CASE_ID" -v dtype="$DTYPE" '
  NR==1 { for (i=1; i<=NF; i++) { gsub(/\"/, "", $i); if ($i=="case_id") ci=i; if ($i=="dtype") di=i; if ($i=="split") si=i } next }
  { for (i=1; i<=NF; i++) gsub(/\"/, "", $i) }
  $(ci)==id && $(di)==dtype { print $(si); exit }
' "$CASES_CSV")"
if [[ -z "$split" ]]; then echo "Unknown case/dtype: $CASE_ID/$DTYPE" >&2; exit 2; fi
case_dir="${OUT_ROOT}/${split}/${DTYPE}/${CASE_ID}"
export PYTHONPATH="${REPO_ROOT}/tools${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "$ACTION" == all ]]; then
  for step in capture ncu sim; do bash "${BASH_SOURCE[0]}" "$step" "$CASE_ID" "$DTYPE"; done
  exit 0
fi
if [[ "$ACTION" == capture ]]; then
  if [[ -e "$case_dir/experiment-manifest.json" ]]; then
    echo "Case already captured: $case_dir (use a new case ID to preserve it)" >&2; exit 1
  fi
  python "$SCRIPT_DIR/gemm_calibration.py" --case "$CASE_ID" --dtype "$DTYPE"
  python "$SCRIPT_DIR/finalize_gemm_manifest.py" --case "$CASE_ID" --dtype "$DTYPE" --split "$split"
  launcher_dir="$case_dir/launchers"
  makefile="$(find "$launcher_dir" -maxdepth 1 -type f -name '*_launch1_Makefile' -print -quit)"
  [[ -n "$makefile" ]] || { echo "No launch1 Makefile" >&2; exit 1; }
  make --no-print-directory -C "$launcher_dir" -f "$(basename "$makefile")"
  echo "CAPTURE_AND_LAUNCHER_BUILD=PASS CASE_DIR=$case_dir"; exit 0
fi
[[ -f "$case_dir/experiment-manifest.json" ]] || { echo "Capture $CASE_ID first" >&2; exit 1; }
launcher_dir="$case_dir/launchers"
makefile="$(find "$launcher_dir" -maxdepth 1 -type f -name '*_launch1_Makefile' -print -quit)"
[[ -n "$makefile" ]] || { echo "No launch1 Makefile" >&2; exit 1; }
launcher="${makefile%_Makefile}"
[[ -x "$launcher" ]] || make --no-print-directory -C "$launcher_dir" -f "$(basename "$makefile")"
if [[ "$ACTION" == ncu ]]; then
  mkdir -p "$case_dir/ncu"
  metrics=sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,gpu__time_duration.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum
  for rep in 1 2 3 4 5; do
    stem="$case_dir/ncu/reference_$rep"
    [[ ! -e "$stem.ncu-rep" ]] || { echo "Refusing to overwrite $stem.ncu-rep" >&2; exit 1; }
    ncu --clock-control base --cache-control all --target-processes all --kernel-name 'regex:fixed_gemm_kernel' --launch-count 1 --metrics "$metrics" -o "$stem" "$launcher" > "$stem.log" 2>&1
    grep -q "Validation PASSED" "$stem.log" || { echo "NCU harness output validation failed: $stem.log" >&2; exit 1; }
    ncu --import "$stem.ncu-rep" --page raw --csv > "$stem.csv"
  done
  echo "NCU=PASS REPORTS=$case_dir/ncu"; exit 0
fi
if [[ "$ACTION" == sim ]]; then
  [[ ! -e "$case_dir/simulator.log" ]] || { echo "Refusing to overwrite $case_dir/simulator.log" >&2; exit 1; }
  cp "$CONFIG_DIR/gpgpusim.config" "$launcher_dir/gpgpusim.config"
  rules="$CONFIG_DIR/sass_primary_hints.rules"
  cp "$rules" "$launcher_dir/sass_primary_hints.rules"
  sha256sum "$CONFIG_DIR/gpgpusim.config" "$rules" > "$case_dir/simulator-config.sha256"
  set +u
  if ! source "$REPO_ROOT/setup_environment"; then echo "Simulator setup failed" >&2; exit 1; fi
  set -u
  sha256sum "$GPGPUSIM_ROOT/lib/$GPGPUSIM_CONFIG/libcudart.so" > "$case_dir/simulator-library.sha256"
  if ! (cd "$launcher_dir" && "./$(basename "$launcher")") > "$case_dir/simulator.log" 2>&1; then tail -n 80 "$case_dir/simulator.log" >&2; exit 1; fi
  grep -q 'Validation PASSED' "$case_dir/simulator.log"
  grep -q '^gpu_tot_sim_cycle' "$case_dir/simulator.log"
  grep -E '^gpu_tot_sim_cycle|Validation PASSED' "$case_dir/simulator.log"
  echo "SIMULATION=PASS CASE_DIR=$case_dir"; exit 0
fi
usage >&2; exit 2
