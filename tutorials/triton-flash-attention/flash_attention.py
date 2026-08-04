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
    args = parser.parse_args()

    if args.seq_len % BLOCK_M:
        parser.error(f"sequence length must be a multiple of {BLOCK_M}")
    if args.head_dim not in (64, 128):
        parser.error("head dimension must be 64 or 128")

    return args


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


def flash_attention(query, key, value, causal=False):
    """Launch the Triton FlashAttention forward kernel."""
    batch, heads, seq_len, head_dim = query.shape
    assert query.shape == key.shape == value.shape

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    output = torch.zeros_like(query)
    scale = 1.0 / (head_dim**0.5)
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)

    flash_attention_kernel[grid](
        query,
        key,
        value,
        output,
        scale,
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CAUSAL=causal,
    )
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
        f"causal={args.causal}, target={args.target})"
    )

    output_dir = OUTPUT_DIR.resolve()
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

    flash_attention(query, key, value, causal=args.causal)
    tracker.save_summary()

    print(f"Offline capture completed: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
