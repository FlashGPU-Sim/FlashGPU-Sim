#!/usr/bin/env bash
#
# Generic sweep script for triton traced workloads.
#
# Usage:
#   ./sweep.sh <workload> [trace|run|gem5|ncu] [options...] [sweep_values...]
#
# Workloads:
#   tma_gemm      TMA GEMM — sweep over M=N=K size
#   flash_attn    Flash Attention — sweep over seq_len
#   llama3_gqa_attn Llama3-style grouped-query attention
#
# Modes:
#   trace  — Generate traces only (requires real GPU, no setup_environment)
#   run    — Run GPGPU-Sim simulation (default). Auto-traces if missing.
#   gem5   — Run FlashGPU-Sim + gem5 Ruby. Auto-traces if missing.
#   ncu    — Profile on real GPU with Nsight Compute
#
# Options:
#   --csv FILE     Read shapes from FILE (relative to script dir, one per line)
#                  Mutually exclusive with sweep_values.
#   --shape CSV    Run a single shape (can be repeated):
#                    tma_gemm:    --shape M,N,K
#                    flash_attn:  --shape B,H,S,D,causal
#
# Options (flash_attn, positional mode only):
#   --head-dim D   Head dimension (default: 64)
#   --batch B      Batch size (default: 2)
#   --heads H      Number of heads (default: 4)
#   --causal       Enable causal masking
#
# Examples:
#   ./sweep.sh tma_gemm trace 128 512 1024
#   ./sweep.sh tma_gemm run --shape 512,6000,2560
#   ./sweep.sh tma_gemm run --csv configs/gemm_shapes_training.csv
#   ./sweep.sh tma_gemm gem5 --shape 512,512,512
#   ./sweep.sh tma_gemm ncu --csv configs/gemm_shapes_training.csv
#   ./sweep.sh tma_gemm run
#   ./sweep.sh flash_attn trace 256 512 1024
#   ./sweep.sh flash_attn run 256 512 --head-dim 128
#   ./sweep.sh flash_attn run --shape 2,4,1024,128,True
#   ./sweep.sh flash_attn ncu 256 512 --head-dim 64 --causal
#   ./sweep.sh llama3_gqa_attn run --csv configs/llama3_8b_gqa_attn_shapes_smoke.csv
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRITON_TRACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$TRITON_TRACE_DIR/../.." && pwd)"
TOP_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
VENV_ACTIVATE="$TRITON_TRACE_DIR/.venv/bin/activate"
GPU_CONFIG_DIR="$REPO_ROOT/configs/SM120_RTX5090"
CSV_MODE=0
CSV_FILE=""

# ============================================================================
# Workload Definitions
# ============================================================================
# Each workload implements:
#   wl_defaults        — default sweep values
#   wl_subdir VAL      — subdirectory name for a sweep value
#   wl_python_args VAL — arguments for the python trace script
#   wl_kernel_name     — triton kernel function name (determines binary name)
#   wl_tflops_expr VAL — python expression: TFLOPS from sweep val & duration_us
#   wl_banner          — display string for the sweep header
# ============================================================================

# --- tma_gemm ---
tma_gemm_init() {
    TRACKING_SUBDIR="test_tma_gemm"
    PYTHON_SCRIPT="test_tma_gemm.py"
}
tma_gemm_defaults()     { echo "128 256 512 1024 2048"; }
tma_gemm_subdir() {
    if [[ "$CSV_MODE" == "1" ]]; then
        local m n k
        IFS=',' read -r m n k <<< "$1"
        echo "m${m}_n${n}_k${k}"
    else
        echo "size_$1"
    fi
}
tma_gemm_python_args() {
    if [[ "$CSV_MODE" == "1" ]]; then
        local m n k
        IFS=',' read -r m n k <<< "$1"
        echo "--m $m --n $n --k $k"
    else
        echo "--size $1"
    fi
}
tma_gemm_kernel_name()  { echo "kernel_tma_gemm"; }
tma_gemm_tflops_expr() {
    if [[ "$CSV_MODE" == "1" ]]; then
        local m n k
        IFS=',' read -r m n k <<< "$1"
        echo "2.0 * ${m} * ${n} * ${k} / (float(dur) * 1e-6) / 1e12"
    else
        echo "2.0 * ${1}**3 / (float(dur) * 1e-6) / 1e12"
    fi
}
tma_gemm_banner() {
    if [[ "$CSV_MODE" == "1" ]]; then
        echo "TMA GEMM Sweep (CSV: $CSV_FILE)"
    else
        echo "TMA GEMM Sweep"
    fi
}
tma_gemm_sweep_label() {
    if [[ "$CSV_MODE" == "1" ]]; then
        echo "m,n,k"
    else
        echo "size"
    fi
}
tma_gemm_summary_key() {
    if [[ "$CSV_MODE" == "1" ]]; then
        echo "$1"
    else
        echo "$1,$1,$1"   # square: size 512 → 512,512,512
    fi
}
tma_gemm_summary_label() { echo "m,n,k"; }

# --- flash_attn ---
FA_HEAD_DIM=128
FA_BATCH=1
FA_HEADS=8
FA_CAUSAL=0

flash_attn_init() {
    TRACKING_SUBDIR="test_flash_attn"
    PYTHON_SCRIPT="test_flash_attn.py"
}
flash_attn_defaults()    { echo "512 1024 2048"; }
flash_attn_subdir() {
    if [[ "$CSV_MODE" == "1" ]]; then
        local batch nheads seqlen headdim causal
        IFS=',' read -r batch nheads seqlen headdim causal <<< "$1"
        local causal_str=""
        [[ "$causal" == "True" ]] && causal_str="_causal"
        echo "b${batch}_h${nheads}_seq${seqlen}_d${headdim}${causal_str}"
    else
        local causal_str=""
        [[ "$FA_CAUSAL" == "1" ]] && causal_str="_causal"
        echo "b${FA_BATCH}_h${FA_HEADS}_seq${1}_d${FA_HEAD_DIM}${causal_str}"
    fi
}
flash_attn_python_args() {
    if [[ "$CSV_MODE" == "1" ]]; then
        local batch nheads seqlen headdim causal
        IFS=',' read -r batch nheads seqlen headdim causal <<< "$1"
        local args="--seq-len $seqlen --head-dim $headdim --batch $batch --heads $nheads"
        [[ "$causal" == "True" ]] && args="$args --causal"
        echo "$args"
    else
        local args="--seq-len $1 --head-dim $FA_HEAD_DIM --batch $FA_BATCH --heads $FA_HEADS"
        [[ "$FA_CAUSAL" == "1" ]] && args="$args --causal"
        echo "$args"
    fi
}
flash_attn_kernel_name()  { echo "_flash_attn_fwd"; }
flash_attn_tflops_expr() {
    if [[ "$CSV_MODE" == "1" ]]; then
        local batch nheads seqlen headdim causal
        IFS=',' read -r batch nheads seqlen headdim causal <<< "$1"
        local factor=4
        [[ "$causal" == "True" ]] && factor=2
        echo "${factor}.0 * ${batch} * ${nheads} * ${seqlen}**2 * ${headdim} / (float(dur) * 1e-6) / 1e12"
    else
        local factor=4
        [[ "$FA_CAUSAL" == "1" ]] && factor=2
        echo "${factor}.0 * ${FA_BATCH} * ${FA_HEADS} * ${1}**2 * ${FA_HEAD_DIM} / (float(dur) * 1e-6) / 1e12"
    fi
}
flash_attn_banner() {
    if [[ "$CSV_MODE" == "1" ]]; then
        echo "Flash Attention Sweep (CSV: $CSV_FILE)"
    else
        local causal_str="non-causal"
        [[ "$FA_CAUSAL" == "1" ]] && causal_str="causal"
        echo "Flash Attention Sweep (d=${FA_HEAD_DIM}, B=${FA_BATCH}, H=${FA_HEADS}, ${causal_str})"
    fi
}
flash_attn_sweep_label() {
    if [[ "$CSV_MODE" == "1" ]]; then
        echo "batch,nheads,seqlen,headdim,causal"
    else
        echo "seq_len"
    fi
}
flash_attn_summary_key() { echo "$1"; }
flash_attn_summary_label() { echo "$(${WORKLOAD}_sweep_label)"; }

# --- llama3_gqa_attn ---
llama3_gqa_attn_init() {
    TRACKING_SUBDIR="test_llama3_gqa_attn"
    PYTHON_SCRIPT="test_llama3_gqa_attn.py"
}
llama3_gqa_attn_defaults() { echo "1,32,8,128,128,True 1,32,8,512,128,True"; }
llama3_gqa_attn_subdir() {
    local batch qheads kvheads seqlen headdim causal
    IFS=',' read -r batch qheads kvheads seqlen headdim causal <<< "$1"
    local causal_str=""
    [[ "$causal" == "True" ]] && causal_str="_causal"
    echo "b${batch}_hq${qheads}_hkv${kvheads}_seq${seqlen}_d${headdim}${causal_str}"
}
llama3_gqa_attn_python_args() {
    local batch qheads kvheads seqlen headdim causal
    IFS=',' read -r batch qheads kvheads seqlen headdim causal <<< "$1"
    local args="--seq-len $seqlen --head-dim $headdim --batch $batch --q-heads $qheads --kv-heads $kvheads"
    [[ "$causal" == "True" ]] && args="$args --causal"
    echo "$args"
}
llama3_gqa_attn_kernel_name() { echo "_llama3_gqa_attn_fwd"; }
llama3_gqa_attn_tflops_expr() {
    local batch qheads kvheads seqlen headdim causal
    IFS=',' read -r batch qheads kvheads seqlen headdim causal <<< "$1"
    local factor=4
    [[ "$causal" == "True" ]] && factor=2
    echo "${factor}.0 * ${batch} * ${qheads} * ${seqlen}**2 * ${headdim} / (float(dur) * 1e-6) / 1e12"
}
llama3_gqa_attn_banner() { echo "Llama3 GQA Attention Sweep"; }
llama3_gqa_attn_sweep_label() { echo "batch,qheads,kvheads,seqlen,headdim,causal"; }
llama3_gqa_attn_summary_key() { echo "$1"; }
llama3_gqa_attn_summary_label() { echo "$(${WORKLOAD}_sweep_label)"; }

# ============================================================================
# Argument Parsing
# ============================================================================

usage() {
    echo "Usage: $0 <workload> [trace|run|ncu] [options...] [sweep_values...]"
    echo ""
    echo "Workloads: tma_gemm, flash_attn"
    echo ""
    echo "Run '$0 <workload> --help' for workload-specific options."
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

WORKLOAD="$1"; shift

# Validate workload
case "$WORKLOAD" in
    tma_gemm|flash_attn|llama3_gqa_attn) ;;
    *) echo "ERROR: Unknown workload '$WORKLOAD'. Available: tma_gemm, flash_attn, llama3_gqa_attn"; exit 1 ;;
esac

# Parse mode
MODE="run"
if [[ $# -gt 0 ]]; then
    case "$1" in
        trace|run|gem5|ncu) MODE="$1"; shift ;;
    esac
fi

# Parse workload-specific options and sweep values
SWEEP_VALUES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv)       CSV_MODE=1; CSV_FILE="$2"; shift 2 ;;
        --shape)     CSV_MODE=1; SWEEP_VALUES+=("$2"); shift 2 ;;
        --head-dim)  FA_HEAD_DIM="$2"; shift 2 ;;
        --batch)     FA_BATCH="$2"; shift 2 ;;
        --heads)     FA_HEADS="$2"; shift 2 ;;
        --causal)    FA_CAUSAL=1; shift ;;
        --help)
            usage ;;
        -*)
            echo "ERROR: Unknown option '$1'"; exit 1 ;;
        *)
            SWEEP_VALUES+=("$1"); shift ;;
    esac
done

# Initialize workload
${WORKLOAD}_init

TRACKING_DIR="$TRITON_TRACE_DIR/triton_kernel_tracking/$TRACKING_SUBDIR"
RESULTS_DIR="$TRACKING_DIR/results"
SIM_LOG_DIR="$RESULTS_DIR/sim-log"
GEM5_LOG_DIR="$RESULTS_DIR/gem5-log"
NCU_REP_DIR="$RESULTS_DIR/ncu-rep"
NCU_PROFILE_METRICS="${NCU_PROFILE_METRICS:-sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,gpu__time_duration.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed}"

# Load sweep values from CSV or apply defaults
if [[ -n "$CSV_FILE" ]]; then
    if [[ ${#SWEEP_VALUES[@]} -gt 0 ]]; then
        echo "ERROR: --csv and sweep values are mutually exclusive"; exit 1
    fi
    CSV_PATH="$SCRIPT_DIR/$CSV_FILE"
    if [[ ! -f "$CSV_PATH" ]]; then
        CSV_PATH="$TRITON_TRACE_DIR/$CSV_FILE"
    fi
    if [[ ! -f "$CSV_PATH" ]]; then
        echo "ERROR: CSV file not found: $CSV_PATH"; exit 1
    fi
    local_first=1
    while IFS= read -r line; do
        if [[ "$local_first" == "1" ]]; then local_first=0; continue; fi
        line="$(echo "$line" | tr -d '[:space:]')"
        [[ -z "$line" ]] && continue
        SWEEP_VALUES+=("$line")
    done < "$CSV_PATH"
elif [[ ${#SWEEP_VALUES[@]} -eq 0 ]]; then
    read -ra SWEEP_VALUES <<< "$(${WORKLOAD}_defaults)"
fi

KERNEL_NAME="$(${WORKLOAD}_kernel_name)"
BANNER="$(${WORKLOAD}_banner)"
SWEEP_LABEL="$(${WORKLOAD}_sweep_label)"

echo "========================================"
echo "$BANNER"
echo "  Mode:   $MODE"
echo "  Values: ${SWEEP_VALUES[*]}"
echo "========================================"

# ============================================================================
# Helpers
# ============================================================================

trace_exists() {
    local val=$1
    local subdir
    subdir="$(${WORKLOAD}_subdir "$val")"
    [[ -f "$TRACKING_DIR/$subdir/launchers/${KERNEL_NAME}_launch1" ]]
}

require_clean_env() {
    if [[ -n "${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" ]]; then
        echo "ERROR: setup_environment is active. This mode requires a clean shell."
        exit 1
    fi
}

install_gpu_config() {
    local launcher_dir=$1
    local skip_l1d="${2:-}"

    cp "$GPU_CONFIG_DIR/gpgpusim.config" "$launcher_dir/"
    cp "$GPU_CONFIG_DIR/config_ampere_islip.icnt" "$launcher_dir/"

    if [[ -n "$skip_l1d" ]]; then
        case "$skip_l1d" in
            0|1) ;;
            *)
                echo "ERROR: GPGPUSIM_SKIP_L1D must be 0 or 1, got '$skip_l1d'" >&2
                exit 2
                ;;
        esac
        if ! grep -q '^-gpgpu_gmem_skip_L1D[[:space:]]' "$launcher_dir/gpgpusim.config"; then
            echo "ERROR: gpgpusim.config does not contain -gpgpu_gmem_skip_L1D" >&2
            exit 2
        fi
        sed -i -E \
            "s/^-gpgpu_gmem_skip_L1D[[:space:]]+[01]/-gpgpu_gmem_skip_L1D ${skip_l1d}/" \
            "$launcher_dir/gpgpusim.config"
    fi
}

# ============================================================================
# Trace
# ============================================================================

do_trace() {
    local val=$1
    local subdir
    subdir="$(${WORKLOAD}_subdir "$val")"

    echo ""
    echo "--- Tracing ${SWEEP_LABEL}=${val} ---"

    require_clean_env

    if [[ ! -f "$VENV_ACTIVATE" ]]; then
        echo "ERROR: Triton venv not found at $VENV_ACTIVATE"
        exit 1
    fi
    source "$VENV_ACTIVATE"
    VENV_CUDA_BIN="$(python - <<'PY'
import site
from pathlib import Path

for root in site.getsitepackages():
    for path in sorted(Path(root).glob("nvidia/cu*/bin/nvcc"), reverse=True):
        print(path.parent)
        raise SystemExit
PY
)"
    if [[ -n "$VENV_CUDA_BIN" ]]; then
        export PATH="$VENV_CUDA_BIN:$PATH"
    fi

    # Generate trace
    local py_args
    py_args="$(${WORKLOAD}_python_args "$val")"
    python3 "$SCRIPT_DIR/$PYTHON_SCRIPT" $py_args

    local launcher_dir="$TRACKING_DIR/$subdir/launchers"

    # Build launcher
    echo "Building launcher for ${SWEEP_LABEL}=${val} ..."
    make -C "$launcher_dir" -f "${KERNEL_NAME}_launch1_Makefile"

    # Copy default full-GPU config. Run modes may override selected options.
    install_gpu_config "$launcher_dir"

    echo "--- Trace complete for ${SWEEP_LABEL}=${val} ---"
}

# ============================================================================
# Run (GPGPU-Sim)
# ============================================================================

do_run() {
    local val=$1
    local subdir
    subdir="$(${WORKLOAD}_subdir "$val")"

    echo ""
    echo "--- Running GPGPU-Sim for ${SWEEP_LABEL}=${val} ---"

    local launcher_dir="$TRACKING_DIR/$subdir/launchers"
    local log_file="$SIM_LOG_DIR/${subdir}_gpgpusim.log"

    mkdir -p "$SIM_LOG_DIR"
    install_gpu_config "$launcher_dir" "${GPGPUSIM_SKIP_L1D:-}"

    if [[ -f "$TOP_ROOT/scripts/env.sh" ]]; then
        export GEM5_ARCH="${GEM5_ARCH:-X86_MESI_Two_Level_Stream}"
        source "$TOP_ROOT/scripts/env.sh"
    fi

    set +u
    source "$REPO_ROOT/setup.sh"
    source "$REPO_ROOT/setup_environment"
    set -u

    (cd "$launcher_dir" && ./${KERNEL_NAME}_launch1) > "$log_file" 2>&1

    if grep -q 'Validation PASSED' "$log_file"; then
        local cycles
        cycles=$(grep -m1 '^gpu_tot_sim_cycle' "$log_file" | awk '{print $3}' || true)
        if [[ -n "$cycles" ]]; then
            echo "  PASSED — ${cycles} cycles (log: $log_file)"
        else
            echo "  PASSED validation, but no simulator cycle stats found (log: $log_file)"
        fi
    else
        echo "  FAILED — check $log_file"
    fi
}

# ============================================================================
# Run (FlashGPU-Sim + gem5)
# ============================================================================

do_gem5() {
    local val=$1
    local subdir
    subdir="$(${WORKLOAD}_subdir "$val")"

    echo ""
    echo "--- Running FlashGPU-Sim + gem5 for ${SWEEP_LABEL}=${val} ---"

    local launcher_dir="$TRACKING_DIR/$subdir/launchers"
    local log_file="$GEM5_LOG_DIR/${subdir}_gem5.log"
    local gem5_arch="${GEM5_ARCH:-X86_MESI_Two_Level_Stream}"
    local gem5_config_name="${FLASHGPU_GEM5_CONFIG_NAME:-example_config_crossbar_garnet.txt}"
    local skip_l1d="${GPGPUSIM_SKIP_L1D:-1}"
    local timeout_sec="${GEM5_TIMEOUT:-${TIMEOUT:-3600}}"

    if [[ ! -f "$TOP_ROOT/scripts/env.sh" ]]; then
        echo "ERROR: gem5 top-level scripts/env.sh not found at $TOP_ROOT/scripts/env.sh" >&2
        echo "       Run gem5 mode from the flashgpu-gem5-top checkout." >&2
        exit 2
    fi

    mkdir -p "$GEM5_LOG_DIR"
    install_gpu_config "$launcher_dir" "$skip_l1d"

    if [[ "${FLASHGPU_GEM5_BUILD:-0}" == "1" ]]; then
        make -C "$TOP_ROOT" GEM5_ARCH="$gem5_arch" build
    fi

    (
        export GEM5_ARCH="$gem5_arch"
        export FLASHGPU_GEM5_TOP="$TOP_ROOT"
        export GEM_FORGE_TOP="$TOP_ROOT"
        export FLASHGPU_GEM5_ENABLE=1
        export FLASHGPU_GEM5_CONFIG_NAME="$gem5_config_name"

        source "$TOP_ROOT/scripts/env.sh"

        cd "$REPO_ROOT"
        set +u
        source setup.sh
        source setup_environment
        set -u

        cd "$launcher_dir"
        timeout "$timeout_sec" "./${KERNEL_NAME}_launch1"
    ) > "$log_file" 2>&1

    if grep -q 'Validation PASSED' "$log_file"; then
        local cycles
        cycles=$(grep -m1 '^gpu_tot_sim_cycle' "$log_file" | awk '{print $3}')
        echo "  PASSED — ${cycles} cycles (log: $log_file)"
    else
        echo "  FAILED — check $log_file"
    fi
}

# ============================================================================
# NCU Profiling
# ============================================================================

do_ncu() {
    local val=$1
    local subdir
    subdir="$(${WORKLOAD}_subdir "$val")"

    echo ""
    echo "--- Profiling ${SWEEP_LABEL}=${val} with ncu ---"

    require_clean_env

    local launcher_dir="$TRACKING_DIR/$subdir/launchers"
    local ncu_output="$NCU_REP_DIR/${subdir}"

    mkdir -p "$NCU_REP_DIR"

    local ncu_args=(
        ncu -f
        -o "$ncu_output"
        --replay-mode application
        --cache-control none
        --clock-control none
        --target-processes all
    )

    if [[ -n "${NCU_SET:-}" ]]; then
        ncu_args+=(--set "$NCU_SET")
    else
        ncu_args+=(--metrics "$NCU_PROFILE_METRICS")
    fi

    "${ncu_args[@]}" "$launcher_dir/${KERNEL_NAME}_launch1"

    echo "--- NCU report saved to ${ncu_output}.ncu-rep ---"
}

# ============================================================================
# Metric Extraction
# ============================================================================

extract_ncu_metrics() {
    local val=$1
    local subdir
    subdir="$(${WORKLOAD}_subdir "$val")"
    local ncu_file="$NCU_REP_DIR/${subdir}.ncu-rep"

    if [[ ! -f "$ncu_file" ]]; then
        echo "${val},N/A,N/A,N/A,N/A,N/A,N/A,N/A"
        return
    fi

    local NCU_METRICS="sm__cycles_elapsed.avg"
    NCU_METRICS+=",sm__cycles_elapsed.avg.per_second"
    NCU_METRICS+=",gpu__time_duration.avg"
    NCU_METRICS+=",sm__throughput.avg.pct_of_peak_sustained_elapsed"
    NCU_METRICS+=",sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
    NCU_METRICS+=",gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
    NCU_METRICS+=",dram__bytes.sum"
    NCU_METRICS+=",dram__bytes_op_read.sum"
    NCU_METRICS+=",dram__bytes_op_write.sum"
    NCU_METRICS+=",dram__throughput.avg.pct_of_peak_sustained_elapsed"

    local tflops_expr
    tflops_expr="$(${WORKLOAD}_tflops_expr "$val")"

    ncu --import "$ncu_file" --csv --page raw --metrics "$NCU_METRICS" 2>/dev/null \
    | python3 -c "
import csv, sys
rows = list(csv.reader(sys.stdin))
if len(rows) < 3:
    print('${val},N/A,N/A,N/A,N/A,N/A,N/A,N/A')
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
dram_bytes_kb = d.get('dram__bytes.sum', 'N/A')
dram_read_kb = d.get('dram__bytes_op_read.sum', 'N/A')
dram_write_b = d.get('dram__bytes_op_write.sum', 'N/A')
dram_bw_pct = d.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', 'N/A')
try:
    tflops = f'{${tflops_expr}:.2f}'
except:
    tflops = 'N/A'
try:
    total_bytes = float(dram_bytes_kb) * 1000.0
    dur_s = float(dur) * 1e-6
    dram_bw_gbs = f'{total_bytes / dur_s / 1e9:.2f}' if dur_s > 0 else 'N/A'
except:
    dram_bw_gbs = 'N/A'
print(f'${val},{sm_cycles},{sm_freq},{dur},{tflops},{dram_bytes_kb},{dram_read_kb},{dram_write_b},{dram_bw_gbs},{dram_tp},{dram_bw_pct},{sm_tp},{tensor_tp}')
"
}

extract_sim_metrics() {
    local val=$1
    local subdir
    subdir="$(${WORKLOAD}_subdir "$val")"
    local log_file="$SIM_LOG_DIR/${subdir}_gpgpusim.log"
    local summary_key
    summary_key="$(${WORKLOAD}_summary_key "$val")"

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

    echo "${summary_key},${sim_cycles},${tot_insn},${ipc},${sim_time},${validation}"
}

extract_gem5_metrics() {
    local val=$1
    local subdir
    subdir="$(${WORKLOAD}_subdir "$val")"
    local log_file="$GEM5_LOG_DIR/${subdir}_gem5.log"
    local summary_key
    summary_key="$(${WORKLOAD}_summary_key "$val")"

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

    echo "${summary_key},${sim_cycles},${tot_insn},${ipc},${sim_time},${validation}"
}

# ============================================================================
# Main
# ============================================================================

if [[ "$MODE" == "trace" ]]; then
    for val in "${SWEEP_VALUES[@]}"; do
        do_trace "$val"
    done
    echo ""
    echo "All traces generated."
    exit 0
fi

if [[ "$MODE" == "ncu" ]]; then
    NEED_TRACE=()
    for val in "${SWEEP_VALUES[@]}"; do
        if ! trace_exists "$val"; then
            NEED_TRACE+=("$val")
        fi
    done

    if [[ ${#NEED_TRACE[@]} -gt 0 ]]; then
        echo ""
        echo "Missing traces for: ${NEED_TRACE[*]}"
        echo "Generating traces first..."
        for val in "${NEED_TRACE[@]}"; do
            do_trace "$val"
        done
    fi

    for val in "${SWEEP_VALUES[@]}"; do
        do_ncu "$val"
    done

    # Generate/update summary (upsert: update existing rows, append new ones)
    NCU_SUMMARY="$NCU_REP_DIR/summary.csv"
    NCU_HEADER="${SWEEP_LABEL},sm_cycles,sm_freq_ghz,duration_us,tflops,dram_bytes_kb,dram_read_kb,dram_write_b,dram_bw_gbs,gpu_dram_throughput_pct,dram_throughput_pct,sm_throughput_pct,tensor_pipe_pct"
    mkdir -p "$NCU_REP_DIR"

    if [[ ! -f "$NCU_SUMMARY" ]]; then
        echo "$NCU_HEADER" > "$NCU_SUMMARY"
    fi

    for val in "${SWEEP_VALUES[@]}"; do
        new_line="$(extract_ncu_metrics "$val")"
        summary_key="$(${WORKLOAD}_summary_key "$val")"
        escaped_key="$(printf '%s' "$summary_key" | sed 's/[.[\*^$()+?{|]/\\&/g')"
        if grep -q "^${escaped_key}," "$NCU_SUMMARY"; then
            sed -i "s|^${escaped_key},.*|${new_line}|" "$NCU_SUMMARY"
        else
            echo "$new_line" >> "$NCU_SUMMARY"
        fi
    done

    echo ""
    echo "========================================"
    echo "NCU summary written to $NCU_SUMMARY"
    echo "========================================"
    cat "$NCU_SUMMARY"
    exit 0
fi

if [[ "$MODE" == "gem5" ]]; then
    NEED_TRACE=()
    for val in "${SWEEP_VALUES[@]}"; do
        if ! trace_exists "$val"; then
            NEED_TRACE+=("$val")
        fi
    done

    if [[ ${#NEED_TRACE[@]} -gt 0 ]]; then
        echo ""
        echo "Missing traces for: ${NEED_TRACE[*]}"
        echo "Generating traces first..."

        if [[ -n "${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" ]]; then
            echo "ERROR: setup_environment is active but traces need to be generated."
            echo "       Run '$0 $WORKLOAD trace ${NEED_TRACE[*]}' in a clean shell first."
            exit 1
        fi

        for val in "${NEED_TRACE[@]}"; do
            do_trace "$val"
        done
    fi

    for val in "${SWEEP_VALUES[@]}"; do
        do_gem5 "$val"
    done

    SUMMARY_FILE="$GEM5_LOG_DIR/summary.csv"
    SUMMARY_LABEL="$(${WORKLOAD}_summary_label)"
    HEADER="${SUMMARY_LABEL},sim_cycles,tot_insn,ipc,sim_time_sec,validation"
    mkdir -p "$GEM5_LOG_DIR"

    if [[ ! -f "$SUMMARY_FILE" ]]; then
        echo "$HEADER" > "$SUMMARY_FILE"
    fi

    for val in "${SWEEP_VALUES[@]}"; do
        new_line="$(extract_gem5_metrics "$val")"
        summary_key="$(${WORKLOAD}_summary_key "$val")"
        escaped_key="$(printf '%s' "$summary_key" | sed 's/[.[\*^$()+?{|]/\\&/g')"
        if grep -q "^${escaped_key}," "$SUMMARY_FILE"; then
            sed -i "s|^${escaped_key},.*|${new_line}|" "$SUMMARY_FILE"
        else
            echo "$new_line" >> "$SUMMARY_FILE"
        fi
    done

    echo ""
    echo "========================================"
    echo "gem5 summary written to $SUMMARY_FILE"
    echo "========================================"
    cat "$SUMMARY_FILE"
    exit 0
fi

# MODE == run
NEED_TRACE=()
for val in "${SWEEP_VALUES[@]}"; do
    if ! trace_exists "$val"; then
        NEED_TRACE+=("$val")
    fi
done

if [[ ${#NEED_TRACE[@]} -gt 0 ]]; then
    echo ""
    echo "Missing traces for: ${NEED_TRACE[*]}"
    echo "Generating traces first..."

    if [[ -n "${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" ]]; then
        echo "ERROR: setup_environment is active but traces need to be generated."
        echo "       Run '$0 $WORKLOAD trace ${NEED_TRACE[*]}' in a clean shell first."
        exit 1
    fi

    for val in "${NEED_TRACE[@]}"; do
        do_trace "$val"
    done
fi

for val in "${SWEEP_VALUES[@]}"; do
    do_run "$val"
done

# Generate/update summary (upsert: update existing rows, append new ones)
SUMMARY_FILE="$SIM_LOG_DIR/summary.csv"
SUMMARY_LABEL="$(${WORKLOAD}_summary_label)"
HEADER="${SUMMARY_LABEL},sim_cycles,tot_insn,ipc,sim_time_sec,validation"
mkdir -p "$SIM_LOG_DIR"

if [[ ! -f "$SUMMARY_FILE" ]]; then
    echo "$HEADER" > "$SUMMARY_FILE"
fi

for val in "${SWEEP_VALUES[@]}"; do
    new_line="$(extract_sim_metrics "$val")"
    # summary_key expands val to canonical form (e.g. size 512 → 512,512,512)
    summary_key="$(${WORKLOAD}_summary_key "$val")"
    escaped_key="$(printf '%s' "$summary_key" | sed 's/[.[\*^$()+?{|]/\\&/g')"
    if grep -q "^${escaped_key}," "$SUMMARY_FILE"; then
        sed -i "s|^${escaped_key},.*|${new_line}|" "$SUMMARY_FILE"
    else
        echo "$new_line" >> "$SUMMARY_FILE"
    fi
done

echo ""
echo "========================================"
echo "Summary written to $SUMMARY_FILE"
echo "========================================"
cat "$SUMMARY_FILE"
