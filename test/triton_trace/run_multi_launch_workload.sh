#!/bin/bash
# =============================================================================
# Build/run/profile/compare all kernel launches in one traced workload.
#
# This is for a single multi-launch workload directory that already contains
# tracking_summary.json and launchers/*_launchN artifacts. For shape sweeps
# across many workload instances, use sweep_tests.sh instead.
#
# Usage:
#   ./run_multi_launch_workload.sh <test_dir> [build|run|both|profile|compare] [--config CONFIG] [--ncu-args EXTRA_ARGS]
#
# Examples:
#   # Build all harnesses
#   ./run_multi_launch_workload.sh triton_kernel_tracking/gpt2_small build
#
#   # Run all on real GPU (clean shell, no setup_environment)
#   ./run_multi_launch_workload.sh triton_kernel_tracking/gpt2_small run
#
#   # Run on GPGPU-Sim (source setup_environment first)
#   source setup.sh && source setup_environment
#   ./run_multi_launch_workload.sh triton_kernel_tracking/gpt2_small both --config SM120_RTX5090
#
#   # NCU profiling (clean shell, no setup_environment)
#   ./run_multi_launch_workload.sh triton_kernel_tracking/gpt2_small profile
#   ./run_multi_launch_workload.sh triton_kernel_tracking/gpt2_small profile --ncu-args "--kernel-name my_kernel"
#
#   # Compare sim cycles vs NCU cycles (after running both)
#   ./run_multi_launch_workload.sh triton_kernel_tracking/gpt2_small compare
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Parse arguments ---
if [ $# -lt 1 ]; then
    echo "Usage: $0 <test_dir> [build|run|both|profile|compare] [--config CONFIG_NAME] [--ncu-args EXTRA_ARGS]"
    echo ""
    echo "  test_dir       Path to one traced multi-launch workload directory"
    echo "  build          Only compile all harnesses"
    echo "  run            Only run all harnesses (must be built first)"
    echo "  both           Build then run (default)"
    echo "  profile        Run NCU profiling on all harnesses (real GPU only)"
    echo "  compare        Compare sim cycles vs NCU cycles (requires prior run + profile)"
    echo "  --config       Copy gpgpusim config before running (e.g. SM120_RTX5090)"
    echo "  --ncu-args     Extra arguments passed to ncu (appended to default flags)"
    echo ""
    echo "Mode is auto-detected:"
    echo "  - If setup_environment was sourced → GPGPU-Sim mode (with stats)"
    echo "  - Otherwise → Real GPU mode"
    exit 1
fi

TEST_DIR="$1"
ACTION="${2:-both}"
CONFIG_NAME=""
NCU_EXTRA_ARGS=""

shift; shift 2>/dev/null || true
while [ $# -gt 0 ]; do
    case "$1" in
        --config)      CONFIG_NAME="$2"; shift ;;
        --ncu-args)    NCU_EXTRA_ARGS="$2"; shift ;;
        *)             echo "Unknown option: $1"; exit 1 ;;
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
    local sim_csv_file="$TEST_DIR/sim_summary.csv"

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

        # Save machine-readable CSV for compare
        {
            echo "kernel,launch_id,status,sim_cycles,l2_bw,l2_accesses,l2_misses,l2_miss_rate"
            while IFS='|' read -r idx name st cyc bw acc miss mr; do
                [ -z "$idx" ] && continue
                # Extract kernel name and launch id from target name (e.g. "linear_kernel_launch3")
                local kn li
                kn=$(echo "$name" | sed 's/_launch[0-9]*$//')
                li=$(echo "$name" | grep -oP 'launch\K[0-9]+$')
                echo "$kn,$li,$st,$cyc,$bw,$acc,$miss,$mr"
            done <<< "$(echo -e "$stats_lines")"
        } > "$sim_csv_file"
        echo "  CSV saved to: $sim_csv_file"
    fi

    return $failed
}

# --- NCU Profile ---
do_profile() {
    if [ "$SIM_MODE" -eq 1 ]; then
        echo "ERROR: NCU profiling requires real GPU. Do not source setup_environment."
        exit 1
    fi

    if ! command -v ncu &>/dev/null; then
        echo "ERROR: ncu (Nsight Compute) not found in PATH."
        echo "  Ensure CUDA toolkit with Nsight Compute is installed."
        exit 1
    fi

    local ncu_dir="$TEST_DIR/ncu_reports"
    mkdir -p "$ncu_dir"

    echo ""
    echo "--- NCU profiling all $TOTAL harnesses ---"
    echo "  Reports: $ncu_dir/"
    [ -n "$NCU_EXTRA_ARGS" ] && echo "  Extra args: $NCU_EXTRA_ARGS"
    echo ""

    local i=0
    local profiled=0
    local skipped=0
    local failed=0
    local csv_file="$ncu_dir/summary.csv"

    # CSV header (matches sweep_tests.sh metrics)
    echo "kernel,launch_id,sm_cycles,sm_freq_ghz,duration_us,sm_throughput_pct,tensor_pipe_pct,dram_throughput_pct" > "$csv_file"

    # Metrics to extract (same as sweep_tests.sh)
    local EXTRACT_METRICS="sm__cycles_elapsed.avg"
    EXTRACT_METRICS+=",sm__cycles_elapsed.avg.per_second"
    EXTRACT_METRICS+=",gpu__time_duration.avg"
    EXTRACT_METRICS+=",sm__throughput.avg.pct_of_peak_sustained_elapsed"
    EXTRACT_METRICS+=",sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
    EXTRACT_METRICS+=",gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"

    while IFS=' ' read -r kname lid; do
        i=$((i + 1))
        local target="${kname}_launch${lid}"
        local binary="$LAUNCHERS_DIR/$target"
        local report_file="$ncu_dir/${target}"

        if [ ! -x "$binary" ]; then
            echo "  [$i/$TOTAL] SKIP (not built): $target"
            skipped=$((skipped + 1))
            continue
        fi

        echo -n "  [$i/$TOTAL] Profiling $target ... "

        # Collect: same flags as sweep_tests.sh
        local exit_code=0
        ncu -f \
            -o "$report_file" \
            --set full \
            --replay-mode application \
            --cache-control none \
            --clock-control none \
            --target-processes all \
            $NCU_EXTRA_ARGS \
            "$binary" > /dev/null 2>&1 || exit_code=$?

        if [ "$exit_code" -ne 0 ]; then
            echo "FAILED (exit code $exit_code)"
            failed=$((failed + 1))
            continue
        fi

        # Extract metrics from .ncu-rep (same approach as sweep_tests.sh)
        local metric_line
        metric_line=$(ncu --import "${report_file}.ncu-rep" --csv --page raw \
            --metrics "$EXTRACT_METRICS" 2>/dev/null \
        | python3 -c "
import csv, sys
rows = list(csv.reader(sys.stdin))
if len(rows) < 3:
    print('N/A,N/A,N/A,N/A,N/A,N/A')
    sys.exit()
header = rows[0]
data = rows[-1]
d = dict(zip(header, data))
sm_cycles = d.get('sm__cycles_elapsed.avg', 'N/A')
sm_freq = d.get('sm__cycles_elapsed.avg.per_second', 'N/A')
dur = d.get('gpu__time_duration.avg', 'N/A')
sm_tp = d.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', 'N/A')
tensor_tp = d.get('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed', 'N/A')
dram_tp = d.get('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed', 'N/A')
print(f'{sm_cycles},{sm_freq},{dur},{sm_tp},{tensor_tp},{dram_tp}')
" 2>/dev/null) || metric_line="N/A,N/A,N/A,N/A,N/A,N/A"

        local sm_cyc sm_freq dur_us sm_tp tensor_tp dram_tp
        IFS=',' read -r sm_cyc sm_freq dur_us sm_tp tensor_tp dram_tp <<< "$metric_line"

        echo "OK  [cycles=$sm_cyc, duration=${dur_us}us, sm_tp=${sm_tp}%]"
        echo "$kname,$lid,$metric_line" >> "$csv_file"
        profiled=$((profiled + 1))
    done <<< "$LAUNCHES"

    echo ""
    echo "============================================================"
    echo "  NCU Profiling Complete"
    echo "  Profiled: $profiled, Failed: $failed, Skipped: $skipped / $TOTAL"
    echo "============================================================"

    # Print summary table
    echo ""
    printf "%-30s  %12s  %12s  %12s  %10s  %10s  %10s\n" \
           "Kernel" "SM_Cycles" "SM_Freq" "Duration" "SM_TP%" "Tensor%" "DRAM_TP%"
    printf "%-30s  %12s  %12s  %12s  %10s  %10s  %10s\n" \
           "------------------------------" "------------" "------------" "------------" "----------" "----------" "----------"

    tail -n +2 "$csv_file" | while IFS=',' read -r kn li cyc freq dur stp ttp dtp; do
        local label="${kn}_L${li}"
        printf "%-30s  %12s  %12s  %12s  %10s  %10s  %10s\n" \
               "$label" "$cyc" "$freq" "$dur" "$stp" "$ttp" "$dtp"
    done

    echo ""
    echo "  CSV summary: $csv_file"
    echo "  NCU reports: $ncu_dir/*.ncu-rep (open with Nsight Compute GUI)"
    echo "============================================================"
}

# --- Compare sim vs NCU ---
do_compare() {
    local sim_csv="$TEST_DIR/sim_summary.csv"
    local ncu_csv="$TEST_DIR/ncu_reports/summary.csv"
    local compare_report="$TEST_DIR/compare_report.txt"

    if [ ! -f "$sim_csv" ]; then
        echo "ERROR: sim_summary.csv not found: $sim_csv"
        echo "  Run with GPGPU-Sim first:  source setup_environment && $0 $TEST_DIR run"
        exit 1
    fi
    if [ ! -f "$ncu_csv" ]; then
        echo "ERROR: NCU summary.csv not found: $ncu_csv"
        echo "  Run NCU profiling first:  $0 $TEST_DIR profile"
        exit 1
    fi

    echo ""
    echo "============================================================"
    echo "  Sim vs NCU Cycle Comparison"
    echo "  Test: $(basename "$TEST_DIR")"
    echo "============================================================"

    python3 -c "
import csv, sys

# Read sim results
sim = {}
with open('$sim_csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['kernel'], row['launch_id'])
        try:
            sim[key] = int(row['sim_cycles'])
        except (ValueError, KeyError):
            sim[key] = None

# Read NCU results
ncu = {}
with open('$ncu_csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['kernel'], row['launch_id'])
        try:
            val = row['sm_cycles'].strip()
            # Handle float values (NCU sometimes returns float)
            ncu[key] = int(float(val))
        except (ValueError, KeyError):
            ncu[key] = None

# Merge keys preserving order from sim
all_keys = list(sim.keys())
for k in ncu:
    if k not in sim:
        all_keys.append(k)

if not all_keys:
    print('  No kernel data found in either file.')
    sys.exit(0)

# Print comparison table — diff = sim - ncu, diff% = diff / ncu * 100
hdr_fmt = '  {:<35s}  {:>14s}  {:>14s}  {:>14s}  {:>10s}'
row_fmt = '  {:<35s}  {:>14s}  {:>14s}  {:>14s}  {:>10s}'
print()
print(hdr_fmt.format('Kernel', 'Sim Cycles', 'NCU Cycles', 'Diff', 'Diff%'))
print(hdr_fmt.format('-' * 35, '-' * 14, '-' * 14, '-' * 14, '-' * 10))

total_sim = 0
total_ncu = 0
diffs = []

for key in all_keys:
    kname, lid = key
    label = f'{kname}_L{lid}'
    s = sim.get(key)
    n = ncu.get(key)

    s_str = f'{s:,}' if s is not None else 'N/A'
    n_str = f'{n:,}' if n is not None else 'N/A'

    if s is not None and n is not None and n > 0:
        diff = s - n
        diff_pct = diff / n * 100.0
        diff_str = f'{diff:+,}'
        pct_str = f'{diff_pct:+.1f}%'
        diffs.append(diff_pct)
        total_sim += s
        total_ncu += n
    else:
        diff_str = 'N/A'
        pct_str = 'N/A'

    print(row_fmt.format(label, s_str, n_str, diff_str, pct_str))

# Summary
print()
print(hdr_fmt.format('-' * 35, '-' * 14, '-' * 14, '-' * 14, '-' * 10))

if total_ncu > 0:
    total_diff = total_sim - total_ncu
    total_pct = total_diff / total_ncu * 100.0
    print(row_fmt.format('TOTAL', f'{total_sim:,}', f'{total_ncu:,}', f'{total_diff:+,}', f'{total_pct:+.1f}%'))
else:
    print(row_fmt.format('TOTAL', f'{total_sim:,}' if total_sim else 'N/A',
                          f'{total_ncu:,}' if total_ncu else 'N/A', 'N/A', 'N/A'))

if diffs:
    import statistics
    avg = statistics.mean(diffs)
    med = statistics.median(diffs)
    lo = min(diffs)
    hi = max(diffs)
    print()
    print(f'  Diff% stats:  avg={avg:+.1f}%  median={med:+.1f}%  min={lo:+.1f}%  max={hi:+.1f}%')

print()
" | tee "$compare_report"

    echo "  Report saved to: $compare_report"
    echo "============================================================"
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
    profile)
        do_profile
        ;;
    compare)
        do_compare
        ;;
    *)
        echo "Unknown action: $ACTION (use build, run, both, profile, or compare)"
        exit 1
        ;;
esac
