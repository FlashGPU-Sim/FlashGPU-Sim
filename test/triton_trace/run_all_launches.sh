#!/bin/bash
# =============================================================================
# Batch build and run all kernel launches for a triton_kernel_tracking test.
#
# Usage:
#   ./run_all_launches.sh <test_dir> [build|run|both] [--config CONFIG]
#
# Examples:
#   # Build all harnesses
#   ./run_all_launches.sh triton_kernel_tracking/gpt2_small build
#
#   # Run all on real GPU (clean shell, no setup_environment)
#   ./run_all_launches.sh triton_kernel_tracking/gpt2_small run
#
#   # Run on GPGPU-Sim (source setup_environment first)
#   source setup.sh && source setup_environment
#   ./run_all_launches.sh triton_kernel_tracking/gpt2_small both --config SM120_RTX5090
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Parse arguments ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 <test_dir> [build|run|both] [--config CONFIG_NAME]"
    echo ""
    echo "  test_dir   Path to triton_kernel_tracking/<test_name>"
    echo "  build      Only compile all harnesses"
    echo "  run        Only run all harnesses (must be built first)"
    echo "  both       Build then run (default)"
    echo "  --config   Copy gpgpusim config before running (e.g. SM120_RTX5090)"
    echo ""
    echo "Mode is auto-detected:"
    echo "  - If setup_environment was sourced → GPGPU-Sim mode (with stats)"
    echo "  - Otherwise → Real GPU mode"
    exit 1
fi

TEST_DIR="$1"
ACTION="${2:-both}"
CONFIG_NAME=""

shift; shift 2>/dev/null || true
while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG_NAME="$2"; shift ;;
        *)        echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Auto-detect sim mode
if [ -n "${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" ]; then
    SIM_MODE=1
else
    SIM_MODE=0
fi

# Resolve paths
if [ ! -d "$TEST_DIR" ]; then
    # Try relative to script dir
    TEST_DIR="$SCRIPT_DIR/$TEST_DIR"
fi
TEST_DIR="$(cd "$TEST_DIR" && pwd)"
LAUNCHERS_DIR="$TEST_DIR/launchers"
SUMMARY_FILE="$TEST_DIR/tracking_summary.json"

if [ ! -d "$LAUNCHERS_DIR" ]; then
    echo "ERROR: launchers directory not found: $LAUNCHERS_DIR"
    exit 1
fi

if [ ! -f "$SUMMARY_FILE" ]; then
    echo "ERROR: tracking_summary.json not found: $SUMMARY_FILE"
    echo "Run the Python tracking script first to generate it."
    exit 1
fi

# --- Copy GPU config if requested ---
if [ -n "$CONFIG_NAME" ]; then
    CONFIG_DIR="$REPO_ROOT/configs/$CONFIG_NAME"
    if [ ! -d "$CONFIG_DIR" ]; then
        echo "ERROR: Config directory not found: $CONFIG_DIR"
        exit 1
    fi
    echo "Copying GPU config from $CONFIG_NAME ..."
    cp "$CONFIG_DIR/gpgpusim.config" "$LAUNCHERS_DIR/"
    cp "$CONFIG_DIR"/config_*.icnt "$LAUNCHERS_DIR/" 2>/dev/null || true
    echo "  Done."
fi

# --- Check simulator mode prerequisites ---
if [ "$SIM_MODE" -eq 1 ]; then
    if [ ! -f "$LAUNCHERS_DIR/gpgpusim.config" ]; then
        echo "ERROR: gpgpusim.config not found in $LAUNCHERS_DIR"
        echo "  Use --config to copy one, e.g.: --config SM120_RTX5090"
        exit 1
    fi
fi

# --- Extract ordered launch list from summary JSON ---
# Each entry: "kernel_name launch_id"
LAUNCHES=$(python3 -c "
import json, sys
with open('$SUMMARY_FILE') as f:
    data = json.load(f)
for launch in data['kernel_launches']:
    print(launch['kernel_name'], launch['launch_id'])
")

TOTAL=$(echo "$LAUNCHES" | wc -l)
echo ""
echo "============================================================"
echo "  Test: $(basename "$TEST_DIR")"
echo "  Launches: $TOTAL"
echo "  Action: $ACTION"
echo "  Mode: $([ "$SIM_MODE" -eq 1 ] && echo "GPGPU-Sim" || echo "Real GPU")"
echo "============================================================"

# --- Build ---
do_build() {
    echo ""
    echo "--- Building all $TOTAL harnesses ---"
    local i=0
    local failed=0
    while IFS=' ' read -r kname lid; do
        i=$((i + 1))
        local makefile="${kname}_launch${lid}_Makefile"
        local target="${kname}_launch${lid}"
        if [ ! -f "$LAUNCHERS_DIR/$makefile" ]; then
            echo "  [$i/$TOTAL] SKIP (no Makefile): $target"
            continue
        fi
        echo -n "  [$i/$TOTAL] Building $target ... "
        if make -C "$LAUNCHERS_DIR" -f "$makefile" -j 2>/dev/null; then
            echo "OK"
        else
            echo "FAILED"
            failed=$((failed + 1))
        fi
    done <<< "$LAUNCHES"

    echo ""
    echo "Build complete: $((TOTAL - failed))/$TOTAL succeeded"
    if [ "$failed" -gt 0 ]; then
        echo "WARNING: $failed builds failed"
    fi
}

# --- Extract GPGPU-Sim stats from output ---
# Parses: gpu_sim_cycle, L2_BW, L2_total_cache_accesses, L2_total_cache_misses, L2_total_cache_miss_rate
extract_sim_stats() {
    local output="$1"
    local cycles="" l2_bw="" l2_accesses="" l2_misses="" l2_miss_rate=""

    cycles=$(echo "$output" | grep -oP '^gpu_sim_cycle\s*=\s*\K[0-9]+' | tail -1)
    l2_bw=$(echo "$output" | grep -oP '^L2_BW\s*=\s*\K[0-9.]+' | tail -1)
    l2_accesses=$(echo "$output" | grep -oP '^L2_total_cache_accesses\s*=\s*\K[0-9]+' | tail -1)
    l2_misses=$(echo "$output" | grep -oP '^L2_total_cache_misses\s*=\s*\K[0-9]+' | tail -1)
    l2_miss_rate=$(echo "$output" | grep -oP '^L2_total_cache_miss_rate\s*=\s*\K[0-9.]+' | tail -1)

    echo "${cycles:-N/A}|${l2_bw:-N/A}|${l2_accesses:-N/A}|${l2_misses:-N/A}|${l2_miss_rate:-N/A}"
}

# --- Run ---
do_run() {
    echo ""
    echo "--- Running all $TOTAL harnesses ---"
    local i=0
    local passed=0
    local failed=0
    local skipped=0
    local failed_list=""

    # Collect stats for summary table (sim mode only)
    local stats_lines=""
    local report_file="$TEST_DIR/run_report.txt"

    while IFS=' ' read -r kname lid; do
        i=$((i + 1))
        local target="${kname}_launch${lid}"
        local binary="$LAUNCHERS_DIR/$target"

        if [ ! -x "$binary" ]; then
            echo "  [$i/$TOTAL] SKIP (not built): $target"
            skipped=$((skipped + 1))
            continue
        fi

        echo -n "  [$i/$TOTAL] Running $target ... "

        # Run from launchers dir (GPGPU-Sim needs gpgpusim.config in cwd)
        local output
        local exit_code=0
        output=$(cd "$LAUNCHERS_DIR" && ./"$target" 2>&1) || exit_code=$?

        local status=""
        if [ "$exit_code" -eq 0 ]; then
            if echo "$output" | grep -qi "PASSED"; then
                status="PASSED"
                passed=$((passed + 1))
            elif echo "$output" | grep -qi "FAILED"; then
                status="FAILED (validation)"
                failed=$((failed + 1))
                failed_list="$failed_list  - $target (validation failed)\n"
            else
                status="OK (no validation)"
                passed=$((passed + 1))
            fi
        else
            status="FAILED (exit code $exit_code)"
            failed=$((failed + 1))
            failed_list="$failed_list  - $target (runtime error)\n"
        fi

        # Extract sim stats if in sim mode
        if [ "$SIM_MODE" -eq 1 ] && [ "$exit_code" -eq 0 ]; then
            local stats
            stats=$(extract_sim_stats "$output")
            local cycles l2_bw l2_acc l2_miss l2_mr
            IFS='|' read -r cycles l2_bw l2_acc l2_miss l2_mr <<< "$stats"
            echo "$status  [cycles=$cycles, L2_BW=${l2_bw} GB/s, L2_miss_rate=$l2_mr]"
            stats_lines="${stats_lines}${i}|${target}|${status}|${cycles}|${l2_bw}|${l2_acc}|${l2_miss}|${l2_mr}\n"
        else
            echo "$status"
            stats_lines="${stats_lines}${i}|${target}|${status}|||||\n"
        fi
    done <<< "$LAUNCHES"

    echo ""
    echo "============================================================"
    echo "  Results: $passed passed, $failed failed, $skipped skipped / $TOTAL total"
    echo "============================================================"
    if [ -n "$failed_list" ]; then
        echo ""
        echo "Failed launches:"
        echo -e "$failed_list"
    fi

    # Print and save stats summary table
    if [ "$SIM_MODE" -eq 1 ]; then
        echo ""
        echo "============================================================"
        echo "  GPGPU-Sim Performance Summary"
        echo "============================================================"
        printf "%-4s  %-40s  %-8s  %12s  %10s  %12s  %12s  %10s\n" \
               "#" "Kernel" "Status" "Cycles" "L2_BW" "L2_Accesses" "L2_Misses" "L2_MR"
        printf "%-4s  %-40s  %-8s  %12s  %10s  %12s  %12s  %10s\n" \
               "----" "----------------------------------------" "--------" "------------" "----------" "------------" "------------" "----------"

        local total_cycles=0
        while IFS='|' read -r idx name st cyc bw acc miss mr; do
            [ -z "$idx" ] && continue
            printf "%-4s  %-40s  %-8s  %12s  %10s  %12s  %12s  %10s\n" \
                   "$idx" "$name" "$st" "$cyc" "${bw:+${bw} GB/s}" "$acc" "$miss" "$mr"
            # Accumulate total cycles
            if [ "$cyc" != "N/A" ] && [ -n "$cyc" ]; then
                total_cycles=$((total_cycles + cyc))
            fi
        done <<< "$(echo -e "$stats_lines")"

        echo ""
        echo "  Total cycles (all kernels): $total_cycles"
        echo "============================================================"

        # Save report to file
        {
            echo "GPGPU-Sim Run Report"
            echo "Date: $(date -Iseconds)"
            echo "Test: $(basename "$TEST_DIR")"
            echo "Total launches: $TOTAL"
            echo "Results: $passed passed, $failed failed, $skipped skipped"
            echo ""
            printf "%-4s  %-40s  %-8s  %12s  %10s  %12s  %12s  %10s\n" \
                   "#" "Kernel" "Status" "Cycles" "L2_BW" "L2_Accesses" "L2_Misses" "L2_MR"
            printf "%-4s  %-40s  %-8s  %12s  %10s  %12s  %12s  %10s\n" \
                   "----" "----------------------------------------" "--------" "------------" "----------" "------------" "------------" "----------"
            while IFS='|' read -r idx name st cyc bw acc miss mr; do
                [ -z "$idx" ] && continue
                printf "%-4s  %-40s  %-8s  %12s  %10s  %12s  %12s  %10s\n" \
                       "$idx" "$name" "$st" "$cyc" "${bw:+${bw} GB/s}" "$acc" "$miss" "$mr"
            done <<< "$(echo -e "$stats_lines")"
            echo ""
            echo "Total cycles (all kernels): $total_cycles"
        } > "$report_file"
        echo "  Report saved to: $report_file"
    fi

    return $failed
}

# --- Execute ---
case "$ACTION" in
    build)
        do_build
        ;;
    run)
        do_run
        ;;
    both)
        do_build
        do_run
        ;;
    *)
        echo "Unknown action: $ACTION (use build, run, or both)"
        exit 1
        ;;
esac
