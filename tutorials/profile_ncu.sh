#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROFILE_NAME="rtx5090"
GPU_ID="0"
TARGET="all"

usage() {
  cat <<'EOF'
Usage: ./tutorials/profile_ncu.sh [options] [target]

Regenerate the Nsight Compute reference profile for one tutorial or all three.

Targets:
  all                       Profile every tutorial (default)
  vectorAdd                 Profile the CUDA vector addition tutorial
  triton-gemm               Profile the captured Triton GEMM tutorial
  triton-flash-attention    Profile the captured Triton FlashAttention tutorial

Options:
  --gpu ID                  Physical GPU index (default: 0)
  --profile-name NAME       Output basename (default: rtx5090)
  -h, --help                Show this help

Lock the GPU to the clock frequencies used by the target configuration before
collecting a reference profile. Triton targets must be captured before they can
be profiled.
EOF
}

while (($#)); do
  case "$1" in
    --gpu)
      [[ $# -ge 2 ]] || {
        echo "Error: --gpu requires an argument." >&2
        exit 2
      }
      GPU_ID="$2"
      shift 2
      ;;
    --profile-name)
      [[ $# -ge 2 ]] || {
        echo "Error: --profile-name requires an argument." >&2
        exit 2
      }
      PROFILE_NAME="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    all | vectorAdd | triton-gemm | triton-flash-attention)
      TARGET="$1"
      shift
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ "${GPU_ID}" =~ ^[0-9]+$ ]] || {
  echo "Error: GPU ID must be a non-negative integer." >&2
  exit 2
}

[[ "${PROFILE_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "Error: profile name may contain only letters, numbers, '.', '_', and '-'." >&2
  exit 2
}

command -v ncu >/dev/null 2>&1 || {
  echo "Error: ncu was not found in PATH." >&2
  exit 1
}

NCU_METRICS="sm__cycles_elapsed.avg"
NCU_METRICS+=",sm__cycles_elapsed.avg.per_second"
NCU_METRICS+=",gpu__time_duration.avg"
NCU_METRICS+=",sm__throughput.avg.pct_of_peak_sustained_elapsed"
NCU_METRICS+=",sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
NCU_METRICS+=",gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
NCU_METRICS+=",dram__bytes.sum"
NCU_METRICS+=",dram__bytes_read.sum"
NCU_METRICS+=",dram__bytes_write.sum"
NCU_METRICS+=",dram__throughput.avg.pct_of_peak_sustained_elapsed"

profile() {
  local label="$1"
  local executable="$2"
  local output_dir="$3"
  local executable_dir
  local executable_name
  local report_base

  if [[ ! -x "${executable}" ]]; then
    echo "Error: ${label} executable not found: ${executable}" >&2
    return 1
  fi

  executable_dir="$(cd -- "$(dirname -- "${executable}")" && pwd -P)"
  executable_name="$(basename -- "${executable}")"
  mkdir -p "${output_dir}"
  output_dir="$(cd -- "${output_dir}" && pwd -P)"
  report_base="${output_dir}/${PROFILE_NAME}"

  echo
  echo "Profiling ${label} on physical GPU ${GPU_ID}"
  echo "Report: ${report_base}.ncu-rep"

  (
    cd "${executable_dir}"
    env \
      -u GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN \
      -u GPGPUSIM_CONFIG \
      -u GPGPUSIM_ROOT \
      -u LD_LIBRARY_PATH \
      CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      ncu -f \
      -o "${report_base}" \
      --cache-control all \
      --clock-control none \
      --pipeline-boost-state stable \
      --target-processes all \
      --metrics "${NCU_METRICS}" \
      "./${executable_name}"
  )

  LC_ALL=C ncu \
    --import "${report_base}.ncu-rep" \
    --csv \
    --page raw \
    --print-units base \
    --metrics "${NCU_METRICS}" \
    >"${report_base}.csv"
}

run_target() {
  case "$1" in
    vectorAdd)
      profile \
        "CUDA vector addition" \
        "${SCRIPT_DIR}/vectorAdd/run/vectorAdd" \
        "${SCRIPT_DIR}/vectorAdd/reference"
      ;;
    triton-gemm)
      profile \
        "Triton GEMM" \
        "${SCRIPT_DIR}/triton-gemm/run/tracking/launchers/kernel_tma_gemm_launch1" \
        "${SCRIPT_DIR}/triton-gemm/reference"
      ;;
    triton-flash-attention)
      profile \
        "Triton FlashAttention" \
        "${SCRIPT_DIR}/triton-flash-attention/run/tracking/launchers/flash_attention_kernel_launch1" \
        "${SCRIPT_DIR}/triton-flash-attention/reference"
      ;;
  esac
}

if [[ "${TARGET}" == "all" ]]; then
  run_target vectorAdd
  run_target triton-gemm
  run_target triton-flash-attention
else
  run_target "${TARGET}"
fi

echo
echo "Nsight Compute profiling completed successfully."
