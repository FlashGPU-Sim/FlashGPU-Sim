#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$ROOT/bin/standard_tests"
OUT_DIR="${OUT_DIR:-$ROOT/out/$(date +%Y%m%d_%H%M%S)}"
NCU="${NCU:-/opt/nvidia/nsight-compute/2025.3.1/ncu}"
DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
SM_CLOCK_MHZ="${SM_CLOCK_MHZ:-1500}"
NCU_SET="${NCU_SET:-full}"
RUN_NATIVE="${RUN_NATIVE:-1}"
RUN_NCU="${RUN_NCU:-1}"
declare -a REQUESTED_PROFILES=()

usage() {
  cat <<'EOF'
Usage: run_fa3_standard_prebuilt_h100.sh [OPTIONS]

Run individually isolated FA3 standard GTest cases on an H100 and retain the
exact Nsight Compute report plus exported raw/details tables for every case.

Options:
  --profile NAME       medium or large forward; repeatable (default: both)
  --out-dir PATH       Result directory
  --device INDEX       CUDA device index (default: 0)
  --sm-clock MHZ       Locked SM clock (default: 1500)
  --ncu PATH           Nsight Compute executable
  --ncu-set NAME       Nsight Compute section set (default: full)
  --no-native          Skip the unprofiled correctness run
  --no-ncu             Skip Nsight Compute
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) REQUESTED_PROFILES+=("$2"); shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --sm-clock) SM_CLOCK_MHZ="$2"; shift 2 ;;
    --ncu) NCU="$2"; shift 2 ;;
    --ncu-set) NCU_SET="$2"; shift 2 ;;
    --no-native) RUN_NATIVE=0; shift ;;
    --no-ncu) RUN_NCU=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -x "$BINARY" ]] || { echo "Missing executable: $BINARY" >&2; exit 2; }
[[ "$DEVICE" =~ ^[0-9]+$ ]] || { echo "Invalid device: $DEVICE" >&2; exit 2; }
[[ "$SM_CLOCK_MHZ" =~ ^[1-9][0-9]*$ ]] || {
  echo "Invalid SM clock: $SM_CLOCK_MHZ" >&2
  exit 2
}
[[ "$RUN_NATIVE" -eq 1 || "$RUN_NCU" -eq 1 ]] || {
  echo "Both native and NCU collection are disabled" >&2
  exit 2
}
if [[ ${#REQUESTED_PROFILES[@]} -eq 0 ]]; then
  REQUESTED_PROFILES=(medium large)
fi
for profile in "${REQUESTED_PROFILES[@]}"; do
  case "$profile" in
    medium|large) ;;
    *) echo "Unknown profile: $profile" >&2; exit 2 ;;
  esac
done
if [[ "$RUN_NCU" -eq 1 ]]; then
  [[ -x "$NCU" ]] || { echo "Nsight Compute is not executable: $NCU" >&2; exit 2; }
fi
command -v nvidia-smi >/dev/null
sudo -n true

export CUDA_VISIBLE_DEVICES="$DEVICE"
export GTEST_COLOR=no
export LD_LIBRARY_PATH="$ROOT/lib:${LD_LIBRARY_PATH:-}"

gpu_name="$(nvidia-smi -i "$DEVICE" --query-gpu=name --format=csv,noheader | head -1)"
[[ "$gpu_name" == *H100* ]] || {
  echo "Expected H100 at device $DEVICE, found: $gpu_name" >&2
  exit 2
}

[[ ! -e "$OUT_DIR" ]] || { echo "Use a new output directory: $OUT_DIR" >&2; exit 2; }
mkdir -p "$OUT_DIR/provenance" "$OUT_DIR/native" "$OUT_DIR/ncu" "$OUT_DIR/status"
cp "$ROOT/checksums.sha256" "$OUT_DIR/provenance/package-checksums.sha256"
(
  cd "$ROOT"
  sha256sum -c checksums.sha256
)
sudo -n nvidia-smi -i "$DEVICE" -lgc "$SM_CLOCK_MHZ,$SM_CLOCK_MHZ"
restore_clock() {
  sudo -n nvidia-smi -i "$DEVICE" -rgc >/dev/null 2>&1 || true
}
trap restore_clock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cat >"$OUT_DIR/provenance/run.txt" <<EOF
started_at=$(date -Is)
hostname=$(hostname)
gpu_name=$gpu_name
device=$DEVICE
sm_clock_mhz=$SM_CLOCK_MHZ
ncu=$NCU
ncu_set=$NCU_SET
profiles=${REQUESTED_PROFILES[*]}
EOF
nvidia-smi -i "$DEVICE" -q >"$OUT_DIR/provenance/nvidia_smi_q.txt"
if [[ "$RUN_NCU" -eq 1 ]]; then
  "$NCU" --version >"$OUT_DIR/provenance/ncu_version.txt" 2>&1
fi
ldd "$BINARY" >"$OUT_DIR/provenance/standard_tests.ldd.txt"

mapfile -t listed_tests < <(
  "$BINARY" --gtest_list_tests | awk '
    /^[A-Za-z0-9_]+\.$/ { suite=$1; next }
    /^  [A-Za-z0-9_]+$/ { sub(/^  /, ""); print suite $1 }
  '
)

profile_for_test() {
  [[ "$1" != *.ShapeTableHas* ]] || return 0
  case "$1" in
    Fa3PrefillFp16MediumTest.H*)
      echo medium ;;
    Fa3PrefillFp16IntegrationTest.H*)
      echo large ;;
  esac
}

profile_requested() {
  local candidate="$1"
  local requested
  for requested in "${REQUESTED_PROFILES[@]}"; do
    [[ "$candidate" != "$requested" ]] || return 0
  done
  return 1
}

declare -a CASE_TESTS=()
declare -a CASE_PROFILES=()
for test_name in "${listed_tests[@]}"; do
  profile="$(profile_for_test "$test_name")"
  [[ -n "$profile" ]] || continue
  profile_requested "$profile" || continue
  CASE_TESTS+=("$test_name")
  CASE_PROFILES+=("$profile")
done
[[ ${#CASE_TESTS[@]} -gt 0 ]] || { echo "No matching FA3 cases" >&2; exit 2; }

printf 'profile\tgtest\tcase_id\tnative_status\tncu_status\n' >"$OUT_DIR/status.tsv"
failed=0
for i in "${!CASE_TESTS[@]}"; do
  test_name="${CASE_TESTS[$i]}"
  profile="${CASE_PROFILES[$i]}"
  case_id="${test_name//./__}"
  done_marker="$OUT_DIR/status/${profile}__${case_id}.done"
  if [[ -f "$done_marker" ]]; then
    echo "[skip] profile=$profile test=$test_name"
    continue
  fi

  native_status=0
  ncu_status=0
  echo "[start] profile=$profile test=$test_name"
  if [[ "$RUN_NATIVE" -eq 1 ]]; then
    set +e
    "$BINARY" --gtest_filter="$test_name" \
      >"$OUT_DIR/native/${profile}__${case_id}.log" 2>&1
    native_status=$?
    set -e
  fi

  if [[ "$RUN_NCU" -eq 1 && "$native_status" -eq 0 ]]; then
    ncu_dir="$OUT_DIR/ncu/${profile}__${case_id}"
    mkdir -p "$ncu_dir"
    report="$ncu_dir/report.ncu-rep"
    set +e
    "$NCU" --target-processes all --clock-control none --cache-control all \
      --set "$NCU_SET" --export "$report" --force-overwrite \
      "$BINARY" --gtest_filter="$test_name" \
      >"$ncu_dir/ncu.log" 2>&1
    ncu_status=$?
    set -e
    if [[ -f "$report" ]]; then
      "$NCU" --import "$report" --csv --page raw >"$ncu_dir/raw.csv" 2>>"$ncu_dir/ncu.log" || true
      "$NCU" --import "$report" --csv --page details >"$ncu_dir/details.csv" 2>>"$ncu_dir/ncu.log" || true
    fi
  elif [[ "$RUN_NCU" -eq 1 ]]; then
    ncu_status=125
  fi

  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$profile" "$test_name" "$case_id" "$native_status" "$ncu_status" \
    >>"$OUT_DIR/status.tsv"
  if [[ "$native_status" -eq 0 && "$ncu_status" -eq 0 ]]; then
    touch "$done_marker"
    echo "[done] profile=$profile test=$test_name"
  else
    failed=1
    echo "[failed] profile=$profile test=$test_name native=$native_status ncu=$ncu_status" >&2
  fi
done

if [[ "$RUN_NCU" -eq 1 ]]; then
  "$ROOT/summarize_fa3_standard_ncu.py" "$OUT_DIR"
fi

echo "output=$OUT_DIR"
exit "$failed"
