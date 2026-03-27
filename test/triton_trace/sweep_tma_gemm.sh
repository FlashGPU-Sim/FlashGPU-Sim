#!/usr/bin/env bash
#
# Sweep TMA GEMM across different square matrix sizes.
#
# Usage:
#   ./sweep_tma_gemm.sh [trace|run|ncu] [size1 size2 ...]
#
#   trace  — Generate traces only (requires real GPU, no setup_environment)
#   run    — Run GPGPU-Sim simulation (default). Auto-traces if trace missing.
#   ncu    — Profile on real GPU with Nsight Compute (no setup_environment)
#
# Examples:
#   ./sweep_tma_gemm.sh trace 128 512 1024
#   ./sweep_tma_gemm.sh ncu 128 512
#   ./sweep_tma_gemm.sh run
#   ./sweep_tma_gemm.sh              # same as 'run' with default sizes
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRACKING_DIR="$SCRIPT_DIR/triton_kernel_tracking/test_tma_gemm"
RESULTS_DIR="$TRACKING_DIR/results"
SIM_LOG_DIR="$RESULTS_DIR/sim-log"
NCU_REP_DIR="$RESULTS_DIR/ncu-rep"
VENV_ACTIVATE="$SCRIPT_DIR/.venv/bin/activate"
GPU_CONFIG_DIR="$REPO_ROOT/configs/SM120_RTX5090"

DEFAULT_SIZES=(128 256 512 1024 2048)

# --- Parse arguments ---
MODE="run"
SIZES=()

if [[ $# -gt 0 ]]; then
    case "$1" in
        trace|run|ncu)
            MODE="$1"
            shift
            ;;
    esac
fi

if [[ $# -gt 0 ]]; then
    SIZES=("$@")
else
    SIZES=("${DEFAULT_SIZES[@]}")
fi

echo "========================================"
echo "TMA GEMM Sweep"
echo "  Mode:  $MODE"
echo "  Sizes: ${SIZES[*]}"
echo "========================================"

# --- Helper: check if trace exists for a size ---
trace_exists() {
    local size=$1
    local launcher_dir="$TRACKING_DIR/size_${size}/launchers"
    [[ -f "$launcher_dir/kernel_tma_gemm_launch1" ]]
}

# --- Helper: ensure not in setup_environment ---
require_clean_env() {
    if [[ -n "${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" ]]; then
        echo "ERROR: setup_environment is active. This mode requires a clean shell."
        exit 1
    fi
}

# --- Trace: generate trace + build launcher for a single size ---
do_trace() {
    local size=$1
    echo ""
    echo "--- Tracing size=$size ---"

    require_clean_env

    # Activate venv
    if [[ ! -f "$VENV_ACTIVATE" ]]; then
        echo "ERROR: Triton venv not found at $VENV_ACTIVATE"
        exit 1
    fi
    source "$VENV_ACTIVATE"

    # Generate trace
    python3 "$SCRIPT_DIR/test_tma_gemm.py" --size "$size"

    local launcher_dir="$TRACKING_DIR/size_${size}/launchers"

    # Build launcher
    echo "Building launcher for size=$size ..."
    make -C "$launcher_dir" -f kernel_tma_gemm_launch1_Makefile

    # Copy GPU config
    cp "$GPU_CONFIG_DIR/gpgpusim.config" "$launcher_dir/"
    cp "$GPU_CONFIG_DIR/config_ampere_islip.icnt" "$launcher_dir/"

    echo "--- Trace complete for size=$size ---"
}

# --- Run: simulate a single size on GPGPU-Sim ---
do_run() {
    local size=$1
    echo ""
    echo "--- Running GPGPU-Sim for size=$size ---"

    local launcher_dir="$TRACKING_DIR/size_${size}/launchers"
    local log_file="$SIM_LOG_DIR/size_${size}_gpgpusim.log"

    mkdir -p "$SIM_LOG_DIR"

    # Source simulator environment (allow unbound vars in setup scripts)
    set +u
    source "$REPO_ROOT/setup.sh"
    source "$REPO_ROOT/setup_environment"
    set -u

    # Run simulation, log to file only
    (cd "$launcher_dir" && ./kernel_tma_gemm_launch1) > "$log_file" 2>&1

    # Show brief result
    if grep -q 'Validation PASSED' "$log_file"; then
        local cycles=$(grep -m1 '^gpu_tot_sim_cycle' "$log_file" | awk '{print $3}')
        echo "  PASSED — ${cycles} cycles (log: $log_file)"
    else
        echo "  FAILED — check $log_file"
    fi
}

# --- NCU: profile a single size on real GPU ---
do_ncu() {
    local size=$1
    echo ""
    echo "--- Profiling size=$size with ncu ---"

    require_clean_env

    local launcher_dir="$TRACKING_DIR/size_${size}/launchers"
    local ncu_output="$NCU_REP_DIR/size_${size}"

    mkdir -p "$NCU_REP_DIR"

    ncu -f \
        -o "$ncu_output" \
        --set full \
        --replay-mode application \
        --cache-control none \
        --clock-control none \
        --target-processes all \
        "$launcher_dir/kernel_tma_gemm_launch1"

    echo "--- NCU report saved to ${ncu_output}.ncu-rep ---"
}

# --- Extract metrics from a log file ---
extract_metrics() {
    local size=$1
    local log_file="$SIM_LOG_DIR/size_${size}_gpgpusim.log"

    local sim_cycles tot_insn ipc sim_time validation

    sim_cycles=$(grep -m1 '^gpu_tot_sim_cycle' "$log_file" | awk '{print $3}' || echo "N/A")
    tot_insn=$(grep -m1 '^gpu_tot_sim_insn' "$log_file" | awk '{print $3}' || echo "N/A")
    ipc=$(grep -m1 '^gpu_tot_ipc' "$log_file" | awk '{print $3}' || echo "N/A")
    sim_time=$(grep -oP '\(\K[0-9]+ sec' "$log_file" | head -1 | awk '{print $1}' || echo "N/A")

    if grep -q 'Validation PASSED' "$log_file"; then
        validation="PASSED"
    else
        validation="FAILED"
    fi

    echo "$size,$sim_cycles,$tot_insn,$ipc,$sim_time,$validation"
}

# --- Main ---

if [[ "$MODE" == "trace" ]]; then
    for size in "${SIZES[@]}"; do
        do_trace "$size"
    done
    echo ""
    echo "All traces generated."
    exit 0
fi

if [[ "$MODE" == "ncu" ]]; then
    # Auto-trace if needed
    NEED_TRACE=()
    for size in "${SIZES[@]}"; do
        if ! trace_exists "$size"; then
            NEED_TRACE+=("$size")
        fi
    done

    if [[ ${#NEED_TRACE[@]} -gt 0 ]]; then
        echo ""
        echo "Missing traces for sizes: ${NEED_TRACE[*]}"
        echo "Generating traces first..."
        for size in "${NEED_TRACE[@]}"; do
            do_trace "$size"
        done
    fi

    for size in "${SIZES[@]}"; do
        do_ncu "$size"
    done
    echo ""
    echo "All NCU profiles saved to $NCU_REP_DIR/"
    exit 0
fi

# MODE == run
# Auto-trace if needed, then run
NEED_TRACE=()
for size in "${SIZES[@]}"; do
    if ! trace_exists "$size"; then
        NEED_TRACE+=("$size")
    fi
done

if [[ ${#NEED_TRACE[@]} -gt 0 ]]; then
    echo ""
    echo "Missing traces for sizes: ${NEED_TRACE[*]}"
    echo "Generating traces first..."

    if [[ -n "${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" ]]; then
        echo "ERROR: setup_environment is active but traces need to be generated."
        echo "       Run './sweep_tma_gemm.sh trace ${NEED_TRACE[*]}' in a clean shell first."
        exit 1
    fi

    for size in "${NEED_TRACE[@]}"; do
        do_trace "$size"
    done
fi

# Run simulations
for size in "${SIZES[@]}"; do
    do_run "$size"
done

# Generate summary
SUMMARY_FILE="$RESULTS_DIR/summary.csv"
mkdir -p "$RESULTS_DIR"
echo "size,sim_cycles,tot_insn,ipc,sim_time_sec,validation" > "$SUMMARY_FILE"
for size in "${SIZES[@]}"; do
    extract_metrics "$size" >> "$SUMMARY_FILE"
done

echo ""
echo "========================================"
echo "Summary written to $SUMMARY_FILE"
echo "========================================"
cat "$SUMMARY_FILE"
