#!/usr/bin/env python3
"""
Parameterized Flash Attention trace generator for shape sweep testing.

Usage:
    python3 test_flash_attn.py --seq-len 512
    python3 test_flash_attn.py --seq-len 1024 --head-dim 128
    python3 test_flash_attn.py --seq-len 2048 --head-dim 64 --batch 2 --heads 8 --causal

Generates trace data in triton_kernel_tracking/test_flash_attn/seq<S>_d<D>/
"""

import argparse
import shutil
from pathlib import Path

import torch
import triton
import triton.language as tl

from track_triton_kernels import TritonKernelTracker

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# ============================================================================
# Flash Attention Forward Kernel (TMA-based)
# ============================================================================

@triton.jit
def _flash_attn_fwd(
    Q, K, V, Out,
    sm_scale,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    Flash Attention forward kernel using TMA (Tensor Memory Accelerator).

    Assumes Q, K, V, Out are contiguous with shape (batch * heads, seq_len, head_dim).
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    qkv_offset = off_hz.to(tl.int64) * SEQ_LEN * HEAD_DIM

    Q_desc = tl.make_tensor_descriptor(
        Q + qkv_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    K_desc = tl.make_tensor_descriptor(
        K + qkv_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    V_desc = tl.make_tensor_descriptor(
        V + qkv_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load_tensor_descriptor(Q_desc, [start_m * BLOCK_M, 0])

    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, SEQ_LEN)
    else:
        hi = SEQ_LEN

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    for start_n in range(0, hi, BLOCK_N):
        k = tl.load_tensor_descriptor(K_desc, [start_n, 0])
        k = tl.trans(k)

        qk = tl.dot(q, k)
        qk = qk * sm_scale

        if CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk_normalized = tl.where(
            m_ij[:, None] == float("-inf"),
            float("-inf"),
            qk - m_ij[:, None],
        )

        p = tl.math.exp(qk_normalized)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp(m_i - m_ij)
        acc = acc * alpha[:, None]

        v = tl.load_tensor_descriptor(V_desc, [start_n, 0])
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    acc = acc / l_i[:, None]

    Out_desc = tl.make_tensor_descriptor(
        Out + qkv_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    tl.store_tensor_descriptor(Out_desc, [start_m * BLOCK_M, 0], acc.to(Out.type.element_ty))


# ============================================================================
# Python Wrappers
# ============================================================================

def flash_attention_forward(q, k, v, causal=False, sm_scale=None):
    assert q.dim() == 4 and q.shape == k.shape == v.shape
    batch, heads, seq_len, head_dim = q.shape

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    q_r = q.view(batch * heads, seq_len, head_dim)
    k_r = k.view(batch * heads, seq_len, head_dim)
    v_r = v.view(batch * heads, seq_len, head_dim)
    o_r = torch.empty_like(q_r)

    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)

    _flash_attn_fwd[grid](
        q_r, k_r, v_r, o_r,
        sm_scale,
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CAUSAL=causal,
    )

    return o_r.view(batch, heads, seq_len, head_dim)


def reference_attention(q, k, v, causal=False, sm_scale=None):
    batch, heads, seq_len, head_dim = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale
    if causal:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights.to(v.dtype), v)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Flash Attention trace for a given shape")
    parser.add_argument("--seq-len", type=int, required=True, help="Sequence length")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension (default: 64)")
    parser.add_argument("--batch", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads (default: 4)")
    parser.add_argument("--causal", action="store_true", help="Enable causal masking")
    args = parser.parse_args()

    seq_len = args.seq_len
    head_dim = args.head_dim
    batch = args.batch
    heads = args.heads
    causal = args.causal

    print(f"{'='*80}")
    print(f"Flash Attention Trace Generator")
    print(f"  seq_len={seq_len}, head_dim={head_dim}, batch={batch}, heads={heads}, causal={causal}")
    print(f"{'='*80}")

    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    )

    causal_str = "_causal" if causal else ""
    subdir = f"b{batch}_h{heads}_seq{seq_len}_d{head_dim}{causal_str}"
    output_dir = (Path(__file__).parent / f"triton_kernel_tracking/test_flash_attn/{subdir}").resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)

    tracker = TritonKernelTracker(output_dir, save_binaries=True, capture_args=True)
    tracker.disable()
    print(f"\nOutput directory: {output_dir}")

    # Create inputs
    torch.manual_seed(42)
    q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

    # Warmup + autotune + validate
    print(f"\nLaunching Flash Attention kernel...")
    tri_out = flash_attention_forward(q, k, v, causal=causal)
    ref_out = reference_attention(q, k, v, causal=causal)

    if torch.allclose(tri_out.float(), ref_out.float(), atol=1e-2, rtol=1e-2):
        print("Kernel output verified — PASSED")
    else:
        max_diff = (tri_out.float() - ref_out.float()).abs().max().item()
        print(f"Kernel output mismatch — FAILED (max diff: {max_diff})")
        return 1

    # Trace
    tracker.enable()
    flash_attention_forward(q, k, v, causal=causal)
    tracker.save_summary()

    print(f"\nTrace saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
