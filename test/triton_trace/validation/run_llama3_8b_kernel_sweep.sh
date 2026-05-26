#!/usr/bin/env bash
#
# Convenience wrapper for a Llama3-8B-shaped kernel sweep.
#
# Usage:
#   ./run_llama3_8b_kernel_sweep.sh trace
#   ./run_llama3_8b_kernel_sweep.sh ncu
#   ./run_llama3_8b_kernel_sweep.sh run
#   ./run_llama3_8b_kernel_sweep.sh compare
#
# Set LLAMA3_SWEEP_SET=smoke to select configs/llama3_8b_*_shapes_smoke.csv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-run}"
SWEEP_SET="${LLAMA3_SWEEP_SET:-smoke}"
GEMM_CSV="configs/llama3_8b_gemm_shapes_${SWEEP_SET}.csv"
GQA_CSV="configs/llama3_8b_gqa_attn_shapes_${SWEEP_SET}.csv"

cd "$SCRIPT_DIR"

if [[ ! -f "$GEMM_CSV" ]]; then
    echo "ERROR: missing GEMM CSV: $GEMM_CSV" >&2
    exit 1
fi
if [[ ! -f "$GQA_CSV" ]]; then
    echo "ERROR: missing GQA CSV: $GQA_CSV" >&2
    exit 1
fi

case "$MODE" in
    trace|run|ncu)
        ./sweep_tests.sh tma_gemm "$MODE" --csv "$GEMM_CSV"
        ./sweep_tests.sh llama3_gqa_attn "$MODE" --csv "$GQA_CSV"
        ;;
    compare)
        python3 compare_cycles.py test_tma_gemm --csv "$GEMM_CSV"
        python3 compare_cycles.py test_llama3_gqa_attn --csv "$GQA_CSV"
        python3 extract_metrics.py test_tma_gemm
        python3 extract_metrics.py test_llama3_gqa_attn
        ;;
    *)
        echo "Usage: $0 [trace|run|ncu|compare]" >&2
        exit 1
        ;;
esac
