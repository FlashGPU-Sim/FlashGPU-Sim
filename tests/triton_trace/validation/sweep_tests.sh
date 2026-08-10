#!/usr/bin/env bash
#
# Generic sweep script for triton traced workloads.
#
# Usage:
#   ./sweep_tests.sh <workload> [trace|run|gem5|ncu] [options...] [sweep_values...]
#
# Workloads:
#   tma_gemm      TMA GEMM — sweep over M=N=K size
#   flash_attn    Flash Attention — sweep over seq_len
#   llama3_gqa_attn Llama3-style grouped-query attention
#   llama3_layer   Triton-only Llama3-style decoder layer
#   llama3_layer_packed Packed QKV and gate/up decoder layer
#   llama3_layer_tiled Packed QKV with head-major QKV output
#   llama3_layer_token_out Tiled QKV with token-major attention output
#   llama3_layer_blocked Tiled QKV with blocked attention output
#   llama3_layer_full_tiled Tile-contract layer tensors and GEMMs
#   llama3_decode_layer Single-token Llama3 decode layer with contiguous KV cache
#   llama3_decode_layer_full_tiled Full-tiled decode layer with tiled KV cache
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
#                    llama3_decode_layer:
#                                  --shape B,QH,KVH,QLEN,KVLEN,D,H,I
#
# Options (flash_attn, positional mode only):
#   --head-dim D   Head dimension (default: 64)
#   --batch B      Batch size (default: 2)
#   --heads H      Number of heads (default: 4)
#   --causal       Enable causal masking
#
# Environment:
#   CUDA_INSTALL_PATH
#                  CUDA Toolkit root used for simulator builds and replay.
#   TRITON_TRACKING_ROOT
#                  Absolute output root for supported trace generators.
#   GPGPUSIM_CLOCK_DOMAINS_OVERRIDE
#                  Optional effective simulator clocks, e.g.
#                  1800:1800:1800:14001. Only copied launcher configs change.
#
# Examples:
#   ./sweep_tests.sh tma_gemm trace 128 512 1024
#   ./sweep_tests.sh tma_gemm run --shape 512,6000,2560
#   ./sweep_tests.sh tma_gemm run --csv configs/gemm_shapes_training.csv
#   ./sweep_tests.sh tma_gemm gem5 --shape 512,512,512
#   ./sweep_tests.sh tma_gemm ncu --csv configs/gemm_shapes_training.csv
#   ./sweep_tests.sh tma_gemm run
#   ./sweep_tests.sh flash_attn trace 256 512 1024
#   ./sweep_tests.sh flash_attn run 256 512 --head-dim 128
#   ./sweep_tests.sh flash_attn run --shape 2,4,1024,128,True
#   ./sweep_tests.sh flash_attn ncu 256 512 --head-dim 64 --causal
#   ./sweep_tests.sh llama3_gqa_attn run --csv configs/llama3_8b_gqa_attn_shapes_smoke.csv
#   ./sweep_tests.sh llama3_layer run --csv configs/llama3_layer_shapes_smoke.csv
#   ./sweep_tests.sh llama3_decode_layer_full_tiled run --csv configs/llama3_decode_layer_full_tiled_shapes_smoke.csv
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
#   wl_multi_launch    — optional: returns true for workloads with many launchers
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
LLAMA3_BLOCK_S_TILE="${LLAMA3_BLOCK_S_TILE:-64}"
LLAMA3_BLOCK_H_TILE="${LLAMA3_BLOCK_H_TILE:-4}"

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

# --- llama3_layer ---
llama3_layer_init() {
    TRACKING_SUBDIR="test_llama3_layer"
    PYTHON_SCRIPT="test_llama3_layer.py"
}
llama3_layer_defaults() { echo "1,32,8,128,128,4096,14336,True"; }
llama3_layer_subdir() {
    local batch qheads kvheads seqlen headdim hidden intermediate causal
    IFS=',' read -r batch qheads kvheads seqlen headdim hidden intermediate causal <<< "$1"
    local causal_str=""
    [[ "$causal" == "True" ]] && causal_str="_causal"
    echo "b${batch}_hq${qheads}_hkv${kvheads}_seq${seqlen}_d${headdim}_h${hidden}_i${intermediate}${causal_str}"
}
llama3_layer_python_args() {
    local batch qheads kvheads seqlen headdim hidden intermediate causal
    IFS=',' read -r batch qheads kvheads seqlen headdim hidden intermediate causal <<< "$1"
    local args="--seq-len $seqlen --head-dim $headdim --batch $batch --q-heads $qheads --kv-heads $kvheads --hidden $hidden --intermediate $intermediate"
    [[ "$causal" == "True" ]] && args="$args --causal"
    echo "$args"
}
llama3_layer_kernel_name() { echo "*"; }
llama3_layer_tflops_expr() { echo "0.0"; }
llama3_layer_banner() { echo "Triton Llama3 Layer Sweep"; }
llama3_layer_sweep_label() { echo "batch,qheads,kvheads,seqlen,headdim,hidden,intermediate,causal"; }
llama3_layer_summary_key() { echo "$1"; }
llama3_layer_summary_label() { echo "$(${WORKLOAD}_sweep_label)"; }
llama3_layer_multi_launch() { return 0; }

llama3_layer_variant_subdir() {
    local base
    base="$(llama3_layer_subdir "$1")"
    if [[ "${LLAMA3_LAYER_VARIANT:-}" == "blocked" ]]; then
        echo "${base}_st${LLAMA3_BLOCK_S_TILE}_ht${LLAMA3_BLOCK_H_TILE}"
    elif [[ "${LLAMA3_LAYER_VARIANT:-}" == "full_tiled" ]]; then
        echo "${base}_mt64_nt128"
    else
        echo "$base"
    fi
}
llama3_layer_variant_python_args() {
    local args
    args="$(llama3_layer_python_args "$1") --variant ${LLAMA3_LAYER_VARIANT}"
    if [[ "${LLAMA3_LAYER_VARIANT}" == "blocked" ]]; then
        args="$args --s-tile ${LLAMA3_BLOCK_S_TILE} --h-tile ${LLAMA3_BLOCK_H_TILE}"
    fi
    echo "$args"
}
llama3_layer_variant_defaults() { llama3_layer_defaults; }
llama3_layer_variant_kernel_name() { llama3_layer_kernel_name; }
llama3_layer_variant_tflops_expr() { llama3_layer_tflops_expr "$1"; }
llama3_layer_variant_sweep_label() { llama3_layer_sweep_label; }
llama3_layer_variant_summary_key() { llama3_layer_summary_key "$1"; }
llama3_layer_variant_summary_label() { llama3_layer_summary_label; }
llama3_layer_variant_multi_launch() { return 0; }

# --- llama3_layer_packed ---
llama3_layer_packed_init() {
    TRACKING_SUBDIR="test_llama3_layer_packed"
    PYTHON_SCRIPT="test_llama3_layer.py"
    LLAMA3_LAYER_VARIANT="packed"
}
llama3_layer_packed_defaults() { llama3_layer_variant_defaults; }
llama3_layer_packed_subdir() { llama3_layer_variant_subdir "$1"; }
llama3_layer_packed_python_args() { llama3_layer_variant_python_args "$1"; }
llama3_layer_packed_kernel_name() { llama3_layer_variant_kernel_name; }
llama3_layer_packed_tflops_expr() { llama3_layer_variant_tflops_expr "$1"; }
llama3_layer_packed_banner() { echo "Triton Llama3 Layer Packed Sweep"; }
llama3_layer_packed_sweep_label() { llama3_layer_variant_sweep_label; }
llama3_layer_packed_summary_key() { llama3_layer_variant_summary_key "$1"; }
llama3_layer_packed_summary_label() { llama3_layer_variant_summary_label; }
llama3_layer_packed_multi_launch() { return 0; }

# --- llama3_layer_tiled ---
llama3_layer_tiled_init() {
    TRACKING_SUBDIR="test_llama3_layer_tiled"
    PYTHON_SCRIPT="test_llama3_layer.py"
    LLAMA3_LAYER_VARIANT="tiled"
}
llama3_layer_tiled_defaults() { llama3_layer_variant_defaults; }
llama3_layer_tiled_subdir() { llama3_layer_variant_subdir "$1"; }
llama3_layer_tiled_python_args() { llama3_layer_variant_python_args "$1"; }
llama3_layer_tiled_kernel_name() { llama3_layer_variant_kernel_name; }
llama3_layer_tiled_tflops_expr() { llama3_layer_variant_tflops_expr "$1"; }
llama3_layer_tiled_banner() { echo "Triton Llama3 Layer Tiled-QKV Sweep"; }
llama3_layer_tiled_sweep_label() { llama3_layer_variant_sweep_label; }
llama3_layer_tiled_summary_key() { llama3_layer_variant_summary_key "$1"; }
llama3_layer_tiled_summary_label() { llama3_layer_variant_summary_label; }
llama3_layer_tiled_multi_launch() { return 0; }

# --- llama3_layer_token_out ---
llama3_layer_token_out_init() {
    TRACKING_SUBDIR="test_llama3_layer_token_out"
    PYTHON_SCRIPT="test_llama3_layer.py"
    LLAMA3_LAYER_VARIANT="token_out"
}
llama3_layer_token_out_defaults() { llama3_layer_variant_defaults; }
llama3_layer_token_out_subdir() { llama3_layer_variant_subdir "$1"; }
llama3_layer_token_out_python_args() { llama3_layer_variant_python_args "$1"; }
llama3_layer_token_out_kernel_name() { llama3_layer_variant_kernel_name; }
llama3_layer_token_out_tflops_expr() { llama3_layer_variant_tflops_expr "$1"; }
llama3_layer_token_out_banner() { echo "Triton Llama3 Layer Token-Out Attention Sweep"; }
llama3_layer_token_out_sweep_label() { llama3_layer_variant_sweep_label; }
llama3_layer_token_out_summary_key() { llama3_layer_variant_summary_key "$1"; }
llama3_layer_token_out_summary_label() { llama3_layer_variant_summary_label; }
llama3_layer_token_out_multi_launch() { return 0; }

# --- llama3_layer_blocked ---
llama3_layer_blocked_init() {
    TRACKING_SUBDIR="test_llama3_layer_blocked"
    PYTHON_SCRIPT="test_llama3_layer.py"
    LLAMA3_LAYER_VARIANT="blocked"
}
llama3_layer_blocked_defaults() { llama3_layer_variant_defaults; }
llama3_layer_blocked_subdir() { llama3_layer_variant_subdir "$1"; }
llama3_layer_blocked_python_args() { llama3_layer_variant_python_args "$1"; }
llama3_layer_blocked_kernel_name() { llama3_layer_variant_kernel_name; }
llama3_layer_blocked_tflops_expr() { llama3_layer_variant_tflops_expr "$1"; }
llama3_layer_blocked_banner() {
    echo "Triton Llama3 Layer Blocked Attention-Out Sweep (S_TILE=${LLAMA3_BLOCK_S_TILE}, H_TILE=${LLAMA3_BLOCK_H_TILE})"
}
llama3_layer_blocked_sweep_label() { llama3_layer_variant_sweep_label; }
llama3_layer_blocked_summary_key() { llama3_layer_variant_summary_key "$1"; }
llama3_layer_blocked_summary_label() { llama3_layer_variant_summary_label; }
llama3_layer_blocked_multi_launch() { return 0; }

# --- llama3_layer_full_tiled ---
llama3_layer_full_tiled_init() {
    TRACKING_SUBDIR="test_llama3_layer_full_tiled"
    PYTHON_SCRIPT="test_llama3_layer.py"
    LLAMA3_LAYER_VARIANT="full_tiled"
}
llama3_layer_full_tiled_defaults() { llama3_layer_variant_defaults; }
llama3_layer_full_tiled_subdir() { llama3_layer_variant_subdir "$1"; }
llama3_layer_full_tiled_python_args() { llama3_layer_variant_python_args "$1"; }
llama3_layer_full_tiled_kernel_name() { llama3_layer_variant_kernel_name; }
llama3_layer_full_tiled_tflops_expr() { llama3_layer_variant_tflops_expr "$1"; }
llama3_layer_full_tiled_banner() {
    echo "Triton Llama3 Layer Full-Tiled Sweep (M_TILE=64, N_TILE=128)"
}
llama3_layer_full_tiled_sweep_label() { llama3_layer_variant_sweep_label; }
llama3_layer_full_tiled_summary_key() { llama3_layer_variant_summary_key "$1"; }
llama3_layer_full_tiled_summary_label() { llama3_layer_variant_summary_label; }
llama3_layer_full_tiled_multi_launch() { return 0; }

# --- llama3_decode_layer ---
llama3_decode_layer_init() {
    TRACKING_SUBDIR="test_llama3_decode_layer"
    PYTHON_SCRIPT="test_llama3_decode_layer.py"
}
llama3_decode_layer_defaults() { echo "256,32,8,1,128,128,4096,14336"; }
llama3_decode_layer_subdir() {
    local batch qheads kvheads qlen kvlen headdim hidden intermediate
    IFS=',' read -r batch qheads kvheads qlen kvlen headdim hidden intermediate <<< "$1"
    echo "b${batch}_hq${qheads}_hkv${kvheads}_q${qlen}_kv${kvlen}_d${headdim}_h${hidden}_i${intermediate}"
}
llama3_decode_layer_python_args() {
    local batch qheads kvheads qlen kvlen headdim hidden intermediate
    IFS=',' read -r batch qheads kvheads qlen kvlen headdim hidden intermediate <<< "$1"
    if [[ "$qlen" != "1" ]]; then
        echo "ERROR: llama3_decode_layer requires qlen=1, got $qlen" >&2
        return 1
    fi
    echo "--batch $batch --q-heads $qheads --kv-heads $kvheads --kv-len $kvlen --head-dim $headdim --hidden $hidden --intermediate $intermediate"
}
llama3_decode_layer_kernel_name() { echo "*"; }
llama3_decode_layer_tflops_expr() { echo "0.0"; }
llama3_decode_layer_banner() { echo "Triton Llama3 Decode Layer Sweep"; }
llama3_decode_layer_sweep_label() { echo "batch,qheads,kvheads,qlen,kvlen,headdim,hidden,intermediate"; }
llama3_decode_layer_summary_key() { echo "$1"; }
llama3_decode_layer_summary_label() { echo "$(${WORKLOAD}_sweep_label)"; }
llama3_decode_layer_multi_launch() { return 0; }

# --- llama3_decode_layer_full_tiled ---
llama3_decode_layer_full_tiled_init() {
    TRACKING_SUBDIR="test_llama3_decode_layer_full_tiled"
    PYTHON_SCRIPT="test_llama3_decode_layer_full_tiled.py"
}
llama3_decode_layer_full_tiled_defaults() { llama3_decode_layer_defaults; }
llama3_decode_layer_full_tiled_subdir() {
    local base
    base="$(llama3_decode_layer_subdir "$1")"
    echo "${base}_mt64_nt128"
}
llama3_decode_layer_full_tiled_python_args() {
    llama3_decode_layer_python_args "$1"
}
llama3_decode_layer_full_tiled_kernel_name() { echo "*"; }
llama3_decode_layer_full_tiled_tflops_expr() { echo "0.0"; }
llama3_decode_layer_full_tiled_banner() {
    echo "Full-Tiled Triton Llama3 Decode Layer Sweep (M_TILE=64, N_TILE=128)"
}
llama3_decode_layer_full_tiled_sweep_label() { llama3_decode_layer_sweep_label; }
llama3_decode_layer_full_tiled_summary_key() { echo "$1"; }
llama3_decode_layer_full_tiled_summary_label() { llama3_decode_layer_summary_label; }
llama3_decode_layer_full_tiled_multi_launch() { return 0; }

# ============================================================================
# Argument Parsing
# ============================================================================

usage() {
    echo "Usage: $0 <workload> [trace|run|ncu] [options...] [sweep_values...]"
    echo ""
    echo "Workloads: tma_gemm, flash_attn, llama3_gqa_attn, llama3_layer, llama3_layer_packed, llama3_layer_tiled, llama3_layer_token_out, llama3_layer_blocked, llama3_layer_full_tiled, llama3_decode_layer, llama3_decode_layer_full_tiled"
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
    tma_gemm|flash_attn|llama3_gqa_attn|llama3_layer|llama3_layer_packed|llama3_layer_tiled|llama3_layer_token_out|llama3_layer_blocked|llama3_layer_full_tiled|llama3_decode_layer|llama3_decode_layer_full_tiled) ;;
    *) echo "ERROR: Unknown workload '$WORKLOAD'. Available: tma_gemm, flash_attn, llama3_gqa_attn, llama3_layer, llama3_layer_packed, llama3_layer_tiled, llama3_layer_token_out, llama3_layer_blocked, llama3_layer_full_tiled, llama3_decode_layer, llama3_decode_layer_full_tiled"; exit 1 ;;
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

TRACKING_ROOT="$TRITON_TRACE_DIR/triton_kernel_tracking"
if [[ -n "${TRITON_TRACKING_ROOT:-}" ]]; then
    case "$WORKLOAD" in
        tma_gemm|flash_attn|llama3_layer*|llama3_decode_layer*)
            TRACKING_ROOT="$TRITON_TRACKING_ROOT"
            ;;
        *)
            echo "ERROR: TRITON_TRACKING_ROOT is not supported for workload '$WORKLOAD'" >&2
            exit 1
            ;;
    esac
fi
if [[ "$TRACKING_ROOT" != /* ]]; then
    echo "ERROR: TRITON_TRACKING_ROOT must be an absolute path: $TRACKING_ROOT" >&2
    exit 1
fi
case "$WORKLOAD" in
    tma_gemm|flash_attn|llama3_layer*|llama3_decode_layer*)
        export TRITON_TRACKING_ROOT="$TRACKING_ROOT"
        ;;
esac

TRACKING_DIR="$TRACKING_ROOT/$TRACKING_SUBDIR"
RESULTS_DIR="$TRACKING_DIR/results"
SIM_LOG_DIR="$RESULTS_DIR/sim-log"
GEM5_LOG_DIR="$RESULTS_DIR/gem5-log"
NCU_REP_DIR="$RESULTS_DIR/ncu-rep"
NCU_PROFILE_METRICS="${NCU_PROFILE_METRICS:-sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,gpu__time_duration.avg,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed}"

# Load sweep values from CSV or apply defaults
if [[ -n "$CSV_FILE" ]]; then
    if [[ ${#SWEEP_VALUES[@]} -gt 0 ]]; then
        echo "ERROR: --csv and sweep values are mutually exclusive"; exit 1
    fi
    if [[ "$CSV_FILE" == /* ]]; then
        CSV_PATH="$CSV_FILE"
    else
        CSV_PATH="$SCRIPT_DIR/$CSV_FILE"
        if [[ ! -f "$CSV_PATH" ]]; then
            CSV_PATH="$TRITON_TRACE_DIR/$CSV_FILE"
        fi
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
    if is_multi_launch; then
        [[ -f "$TRACKING_DIR/$subdir/tracking_summary.json" ]] &&
            find "$TRACKING_DIR/$subdir/launchers" -maxdepth 1 -type f \
                -executable -name '*_launch*' ! -name '*.*' 2>/dev/null | grep -q .
        return
    fi
    [[ -f "$TRACKING_DIR/$subdir/launchers/${KERNEL_NAME}_launch1" ]]
}

is_multi_launch() {
    if declare -F "${WORKLOAD}_multi_launch" >/dev/null; then
        "${WORKLOAD}_multi_launch"
        return
    fi
    return 1
}

launcher_makefiles() {
    local launcher_dir=$1
    find "$launcher_dir" -maxdepth 1 -type f -name '*_launch*_Makefile' \
        | awk 'match($0, /launch[0-9]+/) { print substr($0, RSTART + 6, RLENGTH - 6) "\t" $0 }' \
        | sort -n | cut -f2-
}

launcher_executables() {
    local launcher_dir=$1
    find "$launcher_dir" -maxdepth 1 -type f -executable -name '*_launch*' ! -name '*.*' \
        | awk 'match($0, /launch[0-9]+/) { print substr($0, RSTART + 6, RLENGTH - 6) "\t" $0 }' \
        | sort -n | cut -f2-
}

require_clean_env() {
    if [[ -n "${GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" ]]; then
        echo "ERROR: setup_environment is active. This mode requires a clean shell."
        exit 1
    fi
}

source_simulator_environment() {
    if [[ -z "${CUDA_INSTALL_PATH:-}" ]]; then
        echo "ERROR: set CUDA_INSTALL_PATH to the CUDA Toolkit root before simulator replay."
        exit 1
    fi

    # setup_environment is a legacy sourced script and is not nounset-safe.
    set +u
    source "$REPO_ROOT/setup_environment"
    set -u
}

activate_triton_env() {
    if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        return
    fi
    if [[ -f "$VENV_ACTIVATE" ]]; then
        source "$VENV_ACTIVATE"
        return
    fi
    echo "ERROR: Triton venv not found at $VENV_ACTIVATE and no active VIRTUAL_ENV is set."
    exit 1
}

install_gpu_config() {
    local launcher_dir=$1
    local skip_l1d="${2:-}"
    local clock_domains="${GPGPUSIM_CLOCK_DOMAINS_OVERRIDE:-}"

    cp "$GPU_CONFIG_DIR/gpgpusim.config" "$launcher_dir/"
    cp "$GPU_CONFIG_DIR/config_ampere_islip.icnt" "$launcher_dir/"

    if [[ -n "$clock_domains" ]]; then
        if [[ ! "$clock_domains" =~ ^[0-9]+:[0-9]+:[0-9]+:[0-9]+$ ]]; then
            echo "ERROR: GPGPUSIM_CLOCK_DOMAINS_OVERRIDE must contain four colon-separated clocks, got '$clock_domains'" >&2
            exit 2
        fi
        local clock_lines
        clock_lines=$(grep -c '^-gpgpu_clock_domains[[:space:]]' "$launcher_dir/gpgpusim.config" || true)
        if [[ "$clock_lines" != "1" ]]; then
            echo "ERROR: expected exactly one -gpgpu_clock_domains line, found $clock_lines" >&2
            exit 2
        fi
        sed -i -E \
            "s/^-gpgpu_clock_domains[[:space:]]+.*/-gpgpu_clock_domains ${clock_domains}/" \
            "$launcher_dir/gpgpusim.config"
    fi

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

    activate_triton_env

    # Generate trace
    local py_args
    py_args="$(${WORKLOAD}_python_args "$val")"
    python3 "$SCRIPT_DIR/$PYTHON_SCRIPT" $py_args

    local launcher_dir="$TRACKING_DIR/$subdir/launchers"

    # Build launcher(s)
    echo "Building launcher(s) for ${SWEEP_LABEL}=${val} ..."
    if is_multi_launch; then
        while IFS= read -r makefile; do
            make -C "$launcher_dir" -f "$(basename "$makefile")"
        done < <(launcher_makefiles "$launcher_dir")
    else
        make -C "$launcher_dir" -f "${KERNEL_NAME}_launch1_Makefile"
    fi

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

    mkdir -p "$SIM_LOG_DIR"
    install_gpu_config "$launcher_dir" "${GPGPUSIM_SKIP_L1D:-}"

    if [[ -f "$TOP_ROOT/scripts/env.sh" ]]; then
        export GEM5_ARCH="${GEM5_ARCH:-X86_MESI_Two_Level_Stream}"
        source "$TOP_ROOT/scripts/env.sh"
    fi

    source_simulator_environment

    if is_multi_launch; then
        local shape_log_dir="$SIM_LOG_DIR/${subdir}"
        local aggregate_log="$SIM_LOG_DIR/${subdir}_gpgpusim.log"
        local total_cycles=0
        local total_insn=0
        local total_sim_time=0
        local all_passed=1
        local ran_any=0
        local start_wall
        start_wall="$(date +%s)"

        mkdir -p "$shape_log_dir"
        : > "$aggregate_log"

        while IFS= read -r exe_path; do
            ran_any=1
            local exe
            exe="$(basename "$exe_path")"
            local log_file="$shape_log_dir/${exe}_gpgpusim.log"
            echo "=== $exe ===" >> "$aggregate_log"
            set +e
            (cd "$launcher_dir" && "./$exe") > "$log_file" 2>&1
            local status=$?
            set -e
            cat "$log_file" >> "$aggregate_log"
            echo "" >> "$aggregate_log"

            local kernel_sim_time
            kernel_sim_time=$(grep -oP '\(\K[0-9]+ sec' "$log_file" | head -1 | awk '{print $1}' || true)
            [[ -n "$kernel_sim_time" ]] && total_sim_time=$((total_sim_time + kernel_sim_time))

            if [[ "$status" == "0" ]] && grep -q 'Validation PASSED' "$log_file"; then
                local cycles insn
                cycles=$(grep -m1 '^gpu_tot_sim_cycle' "$log_file" | awk '{print $3}' || true)
                insn=$(grep -m1 '^gpu_tot_sim_insn' "$log_file" | awk '{print $3}' || true)
                [[ -n "$cycles" ]] && total_cycles=$((total_cycles + cycles))
                [[ -n "$insn" ]] && total_insn=$((total_insn + insn))
                echo "  ${exe}: PASSED - ${cycles:-N/A} cycles"
            else
                all_passed=0
                echo "  ${exe}: FAILED - check $log_file"
            fi
        done < <(launcher_executables "$launcher_dir")

        local end_wall wall_time
        end_wall="$(date +%s)"
        wall_time=$((end_wall - start_wall))

        if [[ "$all_passed" == "1" && "$ran_any" == "1" ]]; then
            {
                echo "multi_launch_validation PASSED"
                echo "multi_launch_total_cycles $total_cycles"
                echo "multi_launch_total_insn $total_insn"
                echo "multi_launch_total_sim_time_sec $total_sim_time"
                echo "multi_launch_wall_time_sec $wall_time"
            } >> "$aggregate_log"
            echo "  ALL PASSED - aggregate ${total_cycles} cycles, ${total_sim_time}s sim time, ${wall_time}s wall time (log: $aggregate_log)"
        else
            {
                echo "multi_launch_validation FAILED"
                echo "multi_launch_total_cycles $total_cycles"
                echo "multi_launch_total_insn $total_insn"
                echo "multi_launch_total_sim_time_sec $total_sim_time"
                echo "multi_launch_wall_time_sec $wall_time"
            } >> "$aggregate_log"
            echo "  FAILED - aggregate log: $aggregate_log"
        fi
        return
    fi

    local log_file="$SIM_LOG_DIR/${subdir}_gpgpusim.log"
    local start_wall end_wall
    start_wall="$(date +%s)"
    (cd "$launcher_dir" && ./${KERNEL_NAME}_launch1) > "$log_file" 2>&1
    end_wall="$(date +%s)"
    echo "single_launch_wall_time_sec $((end_wall - start_wall))" >> "$log_file"

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

    if is_multi_launch; then
        local shape_log_dir="$GEM5_LOG_DIR/${subdir}"
        local aggregate_log="$GEM5_LOG_DIR/${subdir}_gem5.log"
        local total_cycles=0
        local total_insn=0
        local total_sim_time=0
        local all_passed=1
        local ran_any=0
        local start_wall
        start_wall="$(date +%s)"

        mkdir -p "$shape_log_dir"
        : > "$aggregate_log"

        while IFS= read -r exe_path; do
            ran_any=1
            local exe
            exe="$(basename "$exe_path")"
            local kernel_log_file="$shape_log_dir/${exe}_gem5.log"
            echo "=== $exe ===" >> "$aggregate_log"
            set +e
            (
                export GEM5_ARCH="$gem5_arch"
                export FLASHGPU_GEM5_TOP="$TOP_ROOT"
                export GEM_FORGE_TOP="$TOP_ROOT"
                export FLASHGPU_GEM5_ENABLE=1
                export FLASHGPU_GEM5_CONFIG_NAME="$gem5_config_name"

                source "$TOP_ROOT/scripts/env.sh"

                cd "$REPO_ROOT"
                source_simulator_environment

                cd "$launcher_dir"
                timeout "$timeout_sec" "./$exe"
            ) > "$kernel_log_file" 2>&1
            local status=$?
            set -e
            cat "$kernel_log_file" >> "$aggregate_log"
            echo "" >> "$aggregate_log"

            local kernel_sim_time
            kernel_sim_time=$(grep -oP '\(\K[0-9]+ sec' "$kernel_log_file" | head -1 | awk '{print $1}' || true)
            [[ -n "$kernel_sim_time" ]] && total_sim_time=$((total_sim_time + kernel_sim_time))

            if [[ "$status" == "0" ]] && grep -q 'Validation PASSED' "$kernel_log_file"; then
                local cycles insn
                cycles=$(grep -m1 '^gpu_tot_sim_cycle' "$kernel_log_file" | awk '{print $3}' || true)
                insn=$(grep -m1 '^gpu_tot_sim_insn' "$kernel_log_file" | awk '{print $3}' || true)
                [[ -n "$cycles" ]] && total_cycles=$((total_cycles + cycles))
                [[ -n "$insn" ]] && total_insn=$((total_insn + insn))
                echo "  ${exe}: PASSED - ${cycles:-N/A} cycles"
            else
                all_passed=0
                echo "  ${exe}: FAILED - check $kernel_log_file"
            fi
        done < <(launcher_executables "$launcher_dir")

        local end_wall wall_time
        end_wall="$(date +%s)"
        wall_time=$((end_wall - start_wall))

        if [[ "$all_passed" == "1" && "$ran_any" == "1" ]]; then
            {
                echo "multi_launch_validation PASSED"
                echo "multi_launch_total_cycles $total_cycles"
                echo "multi_launch_total_insn $total_insn"
                echo "multi_launch_total_sim_time_sec $total_sim_time"
                echo "multi_launch_wall_time_sec $wall_time"
            } >> "$aggregate_log"
            echo "  ALL PASSED - aggregate ${total_cycles} cycles, ${total_sim_time}s sim time, ${wall_time}s wall time (log: $aggregate_log)"
        else
            {
                echo "multi_launch_validation FAILED"
                echo "multi_launch_total_cycles $total_cycles"
                echo "multi_launch_total_insn $total_insn"
                echo "multi_launch_total_sim_time_sec $total_sim_time"
                echo "multi_launch_wall_time_sec $wall_time"
            } >> "$aggregate_log"
            echo "  FAILED - aggregate log: $aggregate_log"
        fi
        return
    fi

    local start_wall end_wall
    start_wall="$(date +%s)"
    (
        export GEM5_ARCH="$gem5_arch"
        export FLASHGPU_GEM5_TOP="$TOP_ROOT"
        export GEM_FORGE_TOP="$TOP_ROOT"
        export FLASHGPU_GEM5_ENABLE=1
        export FLASHGPU_GEM5_CONFIG_NAME="$gem5_config_name"

        source "$TOP_ROOT/scripts/env.sh"

        cd "$REPO_ROOT"
        source_simulator_environment

        cd "$launcher_dir"
        timeout "$timeout_sec" "./${KERNEL_NAME}_launch1"
    ) > "$log_file" 2>&1
    end_wall="$(date +%s)"
    echo "single_launch_wall_time_sec $((end_wall - start_wall))" >> "$log_file"

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

    if is_multi_launch; then
        local shape_ncu_dir="$NCU_REP_DIR/${subdir}"
        local shape_summary="$shape_ncu_dir/summary.csv"
        local failed=0
        local profiled=0

        mkdir -p "$shape_ncu_dir"
        echo "kernel,launch_id,sm_cycles,sm_freq_ghz,duration_us,dram_bytes_kb,dram_read_kb,dram_write_b,dram_bw_gbs,gpu_dram_throughput_pct,dram_throughput_pct,sm_throughput_pct,tensor_pipe_pct" > "$shape_summary"

        while IFS= read -r makefile; do
            make -C "$launcher_dir" -f "$(basename "$makefile")"
        done < <(launcher_makefiles "$launcher_dir")

        while IFS= read -r exe_path; do
            local exe kernel launch_id report_base exit_code metric_line
            exe="$(basename "$exe_path")"
            kernel="$(sed -E 's/_launch[0-9]+$//' <<< "$exe")"
            launch_id="$(sed -nE 's/.*_launch([0-9]+)$/\1/p' <<< "$exe")"
            report_base="$shape_ncu_dir/$exe"
            echo -n "  ${exe}: "

            local ncu_args=(
                ncu -f
                -o "$report_base"
                --cache-control all
                --clock-control none
                --pipeline-boost-state stable
                --target-processes all
            )
            if [[ -n "${NCU_SET:-}" ]]; then
                ncu_args+=(--set "$NCU_SET")
            fi
            # Section sets do not guarantee that summary-only raw metrics are
            # collected, so request the CSV contract metrics explicitly.
            ncu_args+=(--metrics "$NCU_PROFILE_METRICS")

            exit_code=0
            "${ncu_args[@]}" "$exe_path" > "${report_base}.log" 2>&1 || exit_code=$?
            if [[ "$exit_code" -ne 0 ]]; then
                failed=$((failed + 1))
                echo "FAILED (exit code $exit_code; log: ${report_base}.log)"
                continue
            fi

            metric_line=$(LC_ALL=C ncu --import "${report_base}.ncu-rep" --csv --page raw \
                --print-units base \
                --metrics "$NCU_PROFILE_METRICS" 2>/dev/null \
            | python3 -c '
import csv
import sys

rows = list(csv.reader(sys.stdin))
if len(rows) < 3:
    print("N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A")
    raise SystemExit
header = rows[0]
data = rows[-1]
d = dict(zip(header, data))


def number(name):
    value = d.get(name)
    try:
        return float(value.replace(",", ""))
    except (AttributeError, TypeError, ValueError):
        return None


def scaled(name, divisor, digits):
    value = number(name)
    return "N/A" if value is None else f"{value / divisor:.{digits}f}"


duration_ns = number("gpu__time_duration.avg")
dram_bytes_b = number("dram__bytes.sum")
dram_read_b = number("dram__bytes_read.sum")
dram_write_b = number("dram__bytes_write.sum")
sm_cycles = scaled("sm__cycles_elapsed.avg", 1, 2)
sm_freq = scaled("sm__cycles_elapsed.avg.per_second", 1e9, 6)
duration = scaled("gpu__time_duration.avg", 1e3, 6)
sm_tp = scaled("sm__throughput.avg.pct_of_peak_sustained_elapsed", 1, 2)
tensor_tp = scaled("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed", 1, 2)
gpu_dram_tp = scaled("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 1, 2)
dram_tp = scaled("dram__throughput.avg.pct_of_peak_sustained_elapsed", 1, 2)
dram_bytes = "N/A" if dram_bytes_b is None else f"{dram_bytes_b / 1e3:.3f}"
dram_read = "N/A" if dram_read_b is None else f"{dram_read_b / 1e3:.3f}"
dram_write = "N/A" if dram_write_b is None else f"{dram_write_b:.0f}"
if dram_bytes_b is None or duration_ns is None or duration_ns <= 0:
    dram_bw = "N/A"
else:
    dram_bw = f"{dram_bytes_b / (duration_ns * 1e-9) / 1e9:.2f}"
print(",".join((sm_cycles, sm_freq, duration, dram_bytes, dram_read,
                dram_write, dram_bw, gpu_dram_tp, dram_tp, sm_tp, tensor_tp)))
')

            if [[ "$metric_line" == N/A,* ]]; then
                failed=$((failed + 1))
                echo "FAILED (could not extract cycle metrics)"
                continue
            fi
            echo "$kernel,$launch_id,$metric_line" >> "$shape_summary"
            profiled=$((profiled + 1))

            local sm_cycles sm_freq duration rest
            IFS=',' read -r sm_cycles sm_freq duration rest <<< "$metric_line"
            echo "OK - cycles=${sm_cycles}, duration=${duration}us"
        done < <(launcher_executables "$launcher_dir")

        echo "  Profiled ${profiled}, failed ${failed} (summary: $shape_summary)"
        [[ "$failed" -eq 0 && "$profiled" -gt 0 ]]
        return
    fi

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
    fi
    # Keep the raw summary metrics in reports even when a section set is used.
    ncu_args+=(--metrics "$NCU_PROFILE_METRICS")

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

    if is_multi_launch; then
        local shape_summary="$NCU_REP_DIR/${subdir}/summary.csv"
        local summary_key
        summary_key="$(${WORKLOAD}_summary_key "$val")"

        if [[ ! -f "$shape_summary" ]]; then
            echo "${summary_key},N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A"
            return
        fi

        python3 - "$shape_summary" "$summary_key" <<'PY'
import csv
import sys

path, key = sys.argv[1], sys.argv[2]


def number(value):
    try:
        return float(value.replace(",", ""))
    except (AttributeError, TypeError, ValueError):
        return None


with open(path, newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))


def sum_column(name):
    values = [number(row.get(name)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def weighted_average(name, weight_name="duration_us"):
    pairs = []
    for row in rows:
        value = number(row.get(name))
        weight = number(row.get(weight_name))
        if value is not None and weight is not None and weight > 0:
            pairs.append((value, weight))
    if not pairs:
        return None
    return sum(value * weight for value, weight in pairs) / sum(
        weight for _, weight in pairs
    )


sm_cycles = sum_column("sm_cycles")
duration_us = sum_column("duration_us")
dram_bytes = sum_column("dram_bytes_kb")
dram_read = sum_column("dram_read_kb")
dram_write = sum_column("dram_write_b")
sm_freq = weighted_average("sm_freq_ghz")
gpu_dram_tp = weighted_average("gpu_dram_throughput_pct")
dram_tp = weighted_average("dram_throughput_pct")
sm_tp = weighted_average("sm_throughput_pct")
tensor_tp = weighted_average("tensor_pipe_pct")

if dram_bytes is not None and duration_us is not None and duration_us > 0:
    dram_bw = dram_bytes * 1000.0 / (duration_us * 1e-6) / 1e9
else:
    dram_bw = None


def fmt(value, digits=2):
    if value is None:
        return "N/A"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.{digits}f}"


print(
    ",".join(
        (
            key,
            fmt(sm_cycles, 1),
            fmt(sm_freq),
            fmt(duration_us),
            "N/A",
            fmt(dram_bytes),
            fmt(dram_read),
            fmt(dram_write),
            fmt(dram_bw),
            fmt(gpu_dram_tp),
            fmt(dram_tp),
            fmt(sm_tp),
            fmt(tensor_tp),
        )
    )
)
PY
        return
    fi

    if [[ ! -f "$ncu_file" ]]; then
        echo "${val},N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A"
        return
    fi

    local NCU_METRICS="sm__cycles_elapsed.avg"
    NCU_METRICS+=",sm__cycles_elapsed.avg.per_second"
    NCU_METRICS+=",gpu__time_duration.avg"
    NCU_METRICS+=",sm__throughput.avg.pct_of_peak_sustained_elapsed"
    NCU_METRICS+=",sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
    NCU_METRICS+=",gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
    NCU_METRICS+=",dram__bytes.sum"
    NCU_METRICS+=",dram__bytes_read.sum"
    NCU_METRICS+=",dram__bytes_write.sum"
    NCU_METRICS+=",dram__throughput.avg.pct_of_peak_sustained_elapsed"

    local tflops_expr
    tflops_expr="$(${WORKLOAD}_tflops_expr "$val")"

    LC_ALL=C ncu --import "$ncu_file" --csv --page raw --print-units base \
        --metrics "$NCU_METRICS" 2>/dev/null \
    | python3 -c "
import csv, sys
rows = list(csv.reader(sys.stdin))
if len(rows) < 3:
    print('${val},N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A')
    sys.exit()
header = rows[0]
data = rows[-1]
d = dict(zip(header, data))

def number(name):
    value = d.get(name)
    try:
        return float(value.replace(',', ''))
    except (AttributeError, TypeError, ValueError):
        return None

def scaled(name, divisor, digits):
    value = number(name)
    return 'N/A' if value is None else f'{value / divisor:.{digits}f}'

duration_ns = number('gpu__time_duration.avg')
dram_bytes_b = number('dram__bytes.sum')
dram_read_b = number('dram__bytes_read.sum')
dram_write_value = number('dram__bytes_write.sum')
sm_cycles = scaled('sm__cycles_elapsed.avg', 1, 2)
sm_freq = scaled('sm__cycles_elapsed.avg.per_second', 1e9, 6)
dur = scaled('gpu__time_duration.avg', 1e3, 6)
sm_tp = scaled('sm__throughput.avg.pct_of_peak_sustained_elapsed', 1, 2)
tensor_tp = scaled('sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed', 1, 2)
dram_tp = scaled('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed', 1, 2)
dram_bw_pct = scaled('dram__throughput.avg.pct_of_peak_sustained_elapsed', 1, 2)
dram_bytes_kb = 'N/A' if dram_bytes_b is None else f'{dram_bytes_b / 1e3:.3f}'
dram_read_kb = 'N/A' if dram_read_b is None else f'{dram_read_b / 1e3:.3f}'
dram_write_b = 'N/A' if dram_write_value is None else f'{dram_write_value:.0f}'
try:
    tflops = f'{${tflops_expr}:.2f}'
except (TypeError, ValueError, ZeroDivisionError):
    tflops = 'N/A'
if dram_bytes_b is None or duration_ns is None or duration_ns <= 0:
    dram_bw_gbs = 'N/A'
else:
    dram_bw_gbs = f'{dram_bytes_b / (duration_ns * 1e-9) / 1e9:.2f}'
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

    local sim_cycles tot_insn ipc sim_time wall_time validation

    if is_multi_launch; then
        sim_cycles=$(grep -m1 '^multi_launch_total_cycles' "$log_file" | awk '{print $2}' || echo "N/A")
        tot_insn=$(grep -m1 '^multi_launch_total_insn' "$log_file" | awk '{print $2}' || echo "N/A")
        sim_time=$(grep -m1 '^multi_launch_total_sim_time_sec' "$log_file" | awk '{print $2}' || echo "N/A")
        wall_time=$(grep -m1 '^multi_launch_wall_time_sec' "$log_file" | awk '{print $2}' || echo "N/A")
        ipc=$(awk -v insn="$tot_insn" -v cyc="$sim_cycles" 'BEGIN { if (cyc > 0) printf "%.4f", insn / cyc; else printf "N/A" }')
    else
        sim_cycles=$(grep -m1 '^gpu_tot_sim_cycle' "$log_file" | awk '{print $3}' || echo "N/A")
        tot_insn=$(grep -m1 '^gpu_tot_sim_insn' "$log_file" | awk '{print $3}' || echo "N/A")
        ipc=$(grep -m1 '^gpu_tot_ipc' "$log_file" | awk '{print $3}' || echo "N/A")
        sim_time=$(grep -oP '\(\K[0-9]+ sec' "$log_file" | head -1 | awk '{print $1}' || echo "N/A")
        wall_time=$(grep -m1 '^single_launch_wall_time_sec' "$log_file" | awk '{print $2}' || echo "N/A")
    fi

    if is_multi_launch; then
        if grep -q '^multi_launch_validation PASSED' "$log_file"; then
            validation="PASSED"
        else
            validation="FAILED"
        fi
    elif grep -q 'Validation PASSED' "$log_file"; then
        validation="PASSED"
    else
        validation="FAILED"
    fi

    echo "${summary_key},${sim_cycles},${tot_insn},${ipc},${sim_time},${wall_time},${validation}"
}

extract_gem5_metrics() {
    local val=$1
    local subdir
    subdir="$(${WORKLOAD}_subdir "$val")"
    local log_file="$GEM5_LOG_DIR/${subdir}_gem5.log"
    local summary_key
    summary_key="$(${WORKLOAD}_summary_key "$val")"

    local sim_cycles tot_insn ipc sim_time wall_time validation

    if is_multi_launch; then
        sim_cycles=$(grep -m1 '^multi_launch_total_cycles' "$log_file" | awk '{print $2}' || echo "N/A")
        tot_insn=$(grep -m1 '^multi_launch_total_insn' "$log_file" | awk '{print $2}' || echo "N/A")
        sim_time=$(grep -m1 '^multi_launch_total_sim_time_sec' "$log_file" | awk '{print $2}' || echo "N/A")
        wall_time=$(grep -m1 '^multi_launch_wall_time_sec' "$log_file" | awk '{print $2}' || echo "N/A")
        ipc=$(awk -v insn="$tot_insn" -v cyc="$sim_cycles" 'BEGIN { if (cyc > 0) printf "%.4f", insn / cyc; else printf "N/A" }')
    else
        sim_cycles=$(grep -m1 '^gpu_tot_sim_cycle' "$log_file" | awk '{print $3}' || echo "N/A")
        tot_insn=$(grep -m1 '^gpu_tot_sim_insn' "$log_file" | awk '{print $3}' || echo "N/A")
        ipc=$(grep -m1 '^gpu_tot_ipc' "$log_file" | awk '{print $3}' || echo "N/A")
        sim_time=$(grep -oP '\(\K[0-9]+ sec' "$log_file" | head -1 | awk '{print $1}' || echo "N/A")
        wall_time=$(grep -m1 '^single_launch_wall_time_sec' "$log_file" | awk '{print $2}' || echo "N/A")
    fi

    if is_multi_launch; then
        if grep -q '^multi_launch_validation PASSED' "$log_file"; then
            validation="PASSED"
        else
            validation="FAILED"
        fi
    elif grep -q 'Validation PASSED' "$log_file"; then
        validation="PASSED"
    else
        validation="FAILED"
    fi

    echo "${summary_key},${sim_cycles},${tot_insn},${ipc},${sim_time},${wall_time},${validation}"
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
    HEADER="${SUMMARY_LABEL},sim_cycles,tot_insn,ipc,sim_time_sec,wall_time_sec,validation"
    mkdir -p "$GEM5_LOG_DIR"

    expected_cols=$(awk -F, -v h="$HEADER" 'BEGIN { print split(h, cols, ",") }')
    if [[ ! -f "$SUMMARY_FILE" ]]; then
        echo "$HEADER" > "$SUMMARY_FILE"
    elif [[ "$(head -n1 "$SUMMARY_FILE")" != "$HEADER" ]] || \
        awk -F, -v n="$expected_cols" 'NR > 1 && NF != n { bad = 1 } END { exit bad ? 0 : 1 }' "$SUMMARY_FILE"; then
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
HEADER="${SUMMARY_LABEL},sim_cycles,tot_insn,ipc,sim_time_sec,wall_time_sec,validation"
mkdir -p "$SIM_LOG_DIR"

expected_cols=$(awk -F, -v h="$HEADER" 'BEGIN { print split(h, cols, ",") }')
if [[ ! -f "$SUMMARY_FILE" ]]; then
    echo "$HEADER" > "$SUMMARY_FILE"
elif [[ "$(head -n1 "$SUMMARY_FILE")" != "$HEADER" ]] || \
    awk -F, -v n="$expected_cols" 'NR > 1 && NF != n { bad = 1 } END { exit bad ? 0 : 1 }' "$SUMMARY_FILE"; then
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
