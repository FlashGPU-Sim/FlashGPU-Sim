#!/usr/bin/env python3

"""Compile and capture a Triton FlashAttention kernel without a GPU."""

import argparse
import shutil
from pathlib import Path

import torch
import triton
import triton.language as tl
import TritonTrace


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "run" / "tracking"
BLOCK_M = 64
BLOCK_N = 64


def positive_int(value):
    """Parse a positive tensor dimension."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("tensor dimensions must be positive")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture a Triton FlashAttention kernel for FlashGPU-Sim replay."
    )
    parser.add_argument("--batch", type=positive_int, default=32)
    parser.add_argument("--heads", type=positive_int, default=32)
    parser.add_argument("--seq-len", type=positive_int, default=512)
    parser.add_argument("--head-dim", type=positive_int, default=64)
    parser.add_argument(
        "--target",
        default="sm120",
        help="CUDA architecture used for offline Triton compilation",
    )
    parser.add_argument(
        "--causal", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--tma",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use tensor descriptors for Q/K/V loads and the output store",
    )
    parser.add_argument(
        "--mode",
        choices=("full", "tma-only"),
        default="full",
        help="run the full kernel or retain only its tensor-descriptor traffic",
    )
    parser.add_argument(
        "--profile-phases",
        action="store_true",
        help="record four per-CTA clock64 boundaries for the full TMA kernel",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory that receives the TritonTrace capture",
    )
    args = parser.parse_args()

    if args.seq_len % BLOCK_M:
        parser.error(f"sequence length must be a multiple of {BLOCK_M}")
    if args.head_dim not in (64, 128):
        parser.error("head dimension must be 64 or 128")
    if args.mode == "tma-only" and not args.tma:
        parser.error("tma-only mode requires --tma")
    if args.profile_phases and (args.mode != "full" or not args.tma):
        parser.error("phase profiling requires --mode full --tma")

    return args


@triton.jit
def read_clock64():
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %clock64;",
        "=l",
        [],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def flash_attention_kernel(
    query,
    key,
    value,
    output,
    scale,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Compute one block of the FlashAttention forward pass."""
    start_m = tl.program_id(0)
    head_index = tl.program_id(1)
    head_offset = head_index.to(tl.int64) * SEQ_LEN * HEAD_DIM

    query_desc = tl.make_tensor_descriptor(
        query + head_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    key_desc = tl.make_tensor_descriptor(
        key + head_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    value_desc = tl.make_tensor_descriptor(
        value + head_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )

    running_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    running_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    query_block = tl.load_tensor_descriptor(
        query_desc, [start_m * BLOCK_M, 0]
    )

    if CAUSAL:
        stop_n = tl.minimum((start_m + 1) * BLOCK_M, SEQ_LEN)
    else:
        stop_n = SEQ_LEN

    query_offsets = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)

    for start_n in range(0, stop_n, BLOCK_N):
        key_block = tl.load_tensor_descriptor(key_desc, [start_n, 0])
        scores = tl.dot(query_block, tl.trans(key_block)) * scale

        if CAUSAL:
            causal_mask = query_offsets[:, None] >= (
                start_n + key_offsets[None, :]
            )
            scores = tl.where(causal_mask, scores, float("-inf"))

        block_max = tl.maximum(running_max, tl.max(scores, 1))
        normalized_scores = tl.where(
            block_max[:, None] == float("-inf"),
            float("-inf"),
            scores - block_max[:, None],
        )
        probabilities = tl.math.exp(normalized_scores)
        block_sum = tl.sum(probabilities, 1)
        correction = tl.math.exp(running_max - block_max)

        accumulator *= correction[:, None]
        value_block = tl.load_tensor_descriptor(value_desc, [start_n, 0])
        accumulator = tl.dot(
            probabilities.to(value_block.dtype), value_block, accumulator
        )
        running_sum = running_sum * correction + block_sum
        running_max = block_max

    accumulator /= running_sum[:, None]
    output_desc = tl.make_tensor_descriptor(
        output + head_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    tl.store_tensor_descriptor(
        output_desc,
        [start_m * BLOCK_M, 0],
        accumulator.to(output.type.element_ty),
    )


@triton.jit
def flash_attention_kernel_profiled(
    query,
    key,
    value,
    output,
    phase_clocks,
    scale,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Full TMA kernel with entry, main-loop, and epilogue boundaries."""
    start_m = tl.program_id(0)
    head_index = tl.program_id(1)
    head_offset = head_index.to(tl.int64) * SEQ_LEN * HEAD_DIM
    clock_offset = (head_index * tl.cdiv(SEQ_LEN, BLOCK_M) + start_m) * 4

    query_desc = tl.make_tensor_descriptor(
        query + head_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    key_desc = tl.make_tensor_descriptor(
        key + head_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    value_desc = tl.make_tensor_descriptor(
        value + head_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )

    running_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    running_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    phase_0 = read_clock64()
    query_block = tl.load_tensor_descriptor(
        query_desc, [start_m * BLOCK_M, 0]
    )

    if CAUSAL:
        stop_n = tl.minimum((start_m + 1) * BLOCK_M, SEQ_LEN)
    else:
        stop_n = SEQ_LEN

    query_offsets = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    tl.debug_barrier()
    phase_1 = read_clock64()

    for start_n in range(0, stop_n, BLOCK_N):
        key_block = tl.load_tensor_descriptor(key_desc, [start_n, 0])
        scores = tl.dot(query_block, tl.trans(key_block)) * scale

        if CAUSAL:
            causal_mask = query_offsets[:, None] >= (
                start_n + key_offsets[None, :]
            )
            scores = tl.where(causal_mask, scores, float("-inf"))

        block_max = tl.maximum(running_max, tl.max(scores, 1))
        normalized_scores = tl.where(
            block_max[:, None] == float("-inf"),
            float("-inf"),
            scores - block_max[:, None],
        )
        probabilities = tl.math.exp(normalized_scores)
        block_sum = tl.sum(probabilities, 1)
        correction = tl.math.exp(running_max - block_max)

        accumulator *= correction[:, None]
        value_block = tl.load_tensor_descriptor(value_desc, [start_n, 0])
        accumulator = tl.dot(
            probabilities.to(value_block.dtype), value_block, accumulator
        )
        running_sum = running_sum * correction + block_sum
        running_max = block_max

    tl.debug_barrier()
    phase_2 = read_clock64()
    accumulator /= running_sum[:, None]
    output_desc = tl.make_tensor_descriptor(
        output + head_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    tl.store_tensor_descriptor(
        output_desc,
        [start_m * BLOCK_M, 0],
        accumulator.to(output.type.element_ty),
    )
    tl.debug_barrier()
    phase_3 = read_clock64()
    tl.store(phase_clocks + clock_offset + 0, phase_0)
    tl.store(phase_clocks + clock_offset + 1, phase_1)
    tl.store(phase_clocks + clock_offset + 2, phase_2)
    tl.store(phase_clocks + clock_offset + 3, phase_3)


@triton.jit
def flash_attention_kernel_tma_only(
    query,
    key,
    value,
    output,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Retain the full kernel's tensor-descriptor traffic without its math."""
    start_m = tl.program_id(0)
    head_index = tl.program_id(1)
    head_offset = head_index.to(tl.int64) * SEQ_LEN * HEAD_DIM

    query_desc = tl.make_tensor_descriptor(
        query + head_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    key_desc = tl.make_tensor_descriptor(
        key + head_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    value_desc = tl.make_tensor_descriptor(
        value + head_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    output_desc = tl.make_tensor_descriptor(
        output + head_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

    output_block = tl.load_tensor_descriptor(
        query_desc, [start_m * BLOCK_M, 0]
    )
    sample_rows = tl.arange(0, BLOCK_N)
    sample_cols = tl.arange(0, HEAD_DIM)
    sample_offsets = sample_rows[:, None] * HEAD_DIM + sample_cols[None, :]
    sample_mask = (sample_rows[:, None] == 0) & (sample_cols[None, :] == 0)
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        key_block = tl.load_tensor_descriptor(key_desc, [start_n, 0])
        value_block = tl.load_tensor_descriptor(value_desc, [start_n, 0])
        # Sink one element from each tile so Triton cannot delete either TMA
        # load. The final descriptor store overwrites these probe values.
        sink_offset = (
            start_m * BLOCK_M * HEAD_DIM + (start_n // BLOCK_N) * 2
        )
        tl.store(
            output + head_offset + sink_offset + sample_offsets,
            key_block,
            mask=sample_mask,
        )
        tl.store(
            output + head_offset + sink_offset + 1 + sample_offsets,
            value_block,
            mask=sample_mask,
        )

    tl.store_tensor_descriptor(
        output_desc, [start_m * BLOCK_M, 0], output_block
    )


@triton.jit
def flash_attention_kernel_no_tma(
    query,
    key,
    value,
    output,
    scale,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Compute the same FlashAttention kernel using ordinary global loads.

    This intentionally duplicates the math instead of adding a constexpr
    branch to the TMA kernel. Even a compile-time branch perturbs the original
    TMA-on SASS schedule, which would contaminate the A/B comparison.
    """
    start_m = tl.program_id(0)
    head_index = tl.program_id(1)
    head_offset = head_index.to(tl.int64) * SEQ_LEN * HEAD_DIM

    running_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    running_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    accumulator = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    query_offsets = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    head_dim_offsets = tl.arange(0, HEAD_DIM)
    query_block = tl.load(
        query
        + head_offset
        + query_offsets[:, None] * HEAD_DIM
        + head_dim_offsets[None, :]
    )

    if CAUSAL:
        stop_n = tl.minimum((start_m + 1) * BLOCK_M, SEQ_LEN)
    else:
        stop_n = SEQ_LEN

    for start_n in range(0, stop_n, BLOCK_N):
        key_block = tl.load(
            key
            + head_offset
            + (start_n + key_offsets[:, None]) * HEAD_DIM
            + head_dim_offsets[None, :]
        )
        scores = tl.dot(query_block, tl.trans(key_block)) * scale

        if CAUSAL:
            causal_mask = query_offsets[:, None] >= (
                start_n + key_offsets[None, :]
            )
            scores = tl.where(causal_mask, scores, float("-inf"))

        block_max = tl.maximum(running_max, tl.max(scores, 1))
        normalized_scores = tl.where(
            block_max[:, None] == float("-inf"),
            float("-inf"),
            scores - block_max[:, None],
        )
        probabilities = tl.math.exp(normalized_scores)
        block_sum = tl.sum(probabilities, 1)
        correction = tl.math.exp(running_max - block_max)

        accumulator *= correction[:, None]
        value_block = tl.load(
            value
            + head_offset
            + (start_n + key_offsets[:, None]) * HEAD_DIM
            + head_dim_offsets[None, :]
        )
        accumulator = tl.dot(
            probabilities.to(value_block.dtype), value_block, accumulator
        )
        running_sum = running_sum * correction + block_sum
        running_max = block_max

    accumulator /= running_sum[:, None]
    tl.store(
        output
        + head_offset
        + query_offsets[:, None] * HEAD_DIM
        + head_dim_offsets[None, :],
        accumulator.to(output.type.element_ty),
    )


def flash_attention(
    query,
    key,
    value,
    causal=False,
    use_tma=True,
    mode="full",
    profile_phases=False,
):
    """Launch the Triton FlashAttention forward kernel."""
    batch, heads, seq_len, head_dim = query.shape
    assert query.shape == key.shape == value.shape

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    output = torch.zeros_like(query)
    phase_clocks = None
    if profile_phases:
        phase_clocks = torch.zeros(
            triton.cdiv(seq_len, BLOCK_M) * batch * heads * 4,
            dtype=torch.int64,
            device=query.device,
        )
    scale = 1.0 / (head_dim**0.5)
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)

    common_meta = dict(
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    if mode == "tma-only":
        flash_attention_kernel_tma_only[grid](
            query, key, value, output, **common_meta
        )
    elif profile_phases:
        assert phase_clocks is not None
        flash_attention_kernel_profiled[grid](
            query,
            key,
            value,
            output,
            phase_clocks,
            scale,
            CAUSAL=causal,
            **common_meta,
        )
    else:
        kernel = (
            flash_attention_kernel if use_tma else flash_attention_kernel_no_tma
        )
        kernel[grid](
            query,
            key,
            value,
            output,
            scale,
            CAUSAL=causal,
            **common_meta,
        )
    if profile_phases:
        return output, phase_clocks
    return output


def main(args):
    shape = (
        args.batch,
        args.heads,
        args.seq_len,
        args.head_dim,
    )
    print(
        "Compiling Triton FlashAttention offline "
        f"(batch={shape[0]}, heads={shape[1]}, "
        f"sequence={shape[2]}, head dimension={shape[3]}, "
        f"causal={args.causal}, tma={args.tma}, mode={args.mode}, "
        f"profile_phases={args.profile_phases}, target={args.target})"
    )

    output_dir = args.output_dir.resolve()
    run_dir = (SCRIPT_DIR / "run").resolve()
    if output_dir.parent != run_dir:
        raise ValueError(
            "--output-dir must be a direct child of "
            "tutorials/triton-flash-attention/run"
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)

    tracker = TritonTrace.Tracker(
        output_dir,
        save_binaries=True,
        capture_args=True,
        mode="offline",
        target=args.target,
    )

    torch.manual_seed(42)
    query = torch.randn(shape, dtype=torch.float16)
    key = torch.randn(shape, dtype=torch.float16)
    value = torch.randn(shape, dtype=torch.float16)

    flash_attention(
        query,
        key,
        value,
        causal=args.causal,
        use_tma=args.tma,
        mode=args.mode,
        profile_phases=args.profile_phases,
    )
    tracker.save_summary()

    print(f"Offline capture completed: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
