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
LAYER_CSV="configs/llama3_layer_shapes_${SWEEP_SET}.csv"

cd "$SCRIPT_DIR"

if [[ ! -f "$GEMM_CSV" ]]; then
    echo "ERROR: missing GEMM CSV: $GEMM_CSV" >&2
    exit 1
fi
if [[ ! -f "$GQA_CSV" ]]; then
    echo "ERROR: missing GQA CSV: $GQA_CSV" >&2
    exit 1
fi
if [[ ! -f "$LAYER_CSV" ]]; then
    echo "WARNING: missing layer CSV: $LAYER_CSV" >&2
fi

case "$MODE" in
    trace|run|ncu)
        ./sweep_tests.sh tma_gemm "$MODE" --csv "$GEMM_CSV"
        ./sweep_tests.sh llama3_gqa_attn "$MODE" --csv "$GQA_CSV"
        if [[ -f "$LAYER_CSV" && "$MODE" != "ncu" ]]; then
            ./sweep_tests.sh llama3_layer "$MODE" --csv "$LAYER_CSV"
            ./sweep_tests.sh llama3_layer_packed "$MODE" --csv "$LAYER_CSV"
            ./sweep_tests.sh llama3_layer_tiled "$MODE" --csv "$LAYER_CSV"
            ./sweep_tests.sh llama3_layer_token_out "$MODE" --csv "$LAYER_CSV"
            ./sweep_tests.sh llama3_layer_blocked "$MODE" --csv "$LAYER_CSV"
            ./sweep_tests.sh llama3_layer_full_tiled "$MODE" --csv "$LAYER_CSV"
        fi
        ;;
    compare)
        python3 compare_cycles.py test_tma_gemm --csv "$GEMM_CSV"
        python3 compare_cycles.py test_llama3_gqa_attn --csv "$GQA_CSV"
        python3 extract_metrics.py test_tma_gemm
        python3 extract_metrics.py test_llama3_gqa_attn
        if [[ -f "$LAYER_CSV" ]]; then
            python3 extract_metrics.py test_llama3_layer
            python3 extract_metrics.py test_llama3_layer_packed
            python3 extract_metrics.py test_llama3_layer_tiled
            python3 extract_metrics.py test_llama3_layer_token_out
            python3 extract_metrics.py test_llama3_layer_blocked
            python3 extract_metrics.py test_llama3_layer_full_tiled
        fi
        ;;
    *)
        echo "Usage: $0 [trace|run|ncu|compare]" >&2
        exit 1
        ;;
esac
