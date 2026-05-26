#!/usr/bin/env python3
"""
Parameterized Llama3-style grouped-query attention trace generator.

This is a kernel-level workload, not a full model replay.  It captures one
causal GQA forward kernel with Q heads and fewer KV heads, matching the main
attention shape difference between Llama3 and vanilla multi-head attention.
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

TRITON_TRACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRITON_TRACE_DIR))

from track_triton_kernels import TritonKernelTracker

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def get_gqa_autotune_config():
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=2, num_warps=4),
    ]


@triton.autotune(
    configs=get_gqa_autotune_config(),
    key=["SEQ_LEN", "HEAD_DIM", "Q_HEADS", "KV_HEADS"],
)
@triton.jit
def _llama3_gqa_attn_fwd(
    Q,
    K,
    V,
    Out,
    sm_scale,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    batch_qh = tl.program_id(1)

    q_head = batch_qh % Q_HEADS
    batch = batch_qh // Q_HEADS
    kv_group = Q_HEADS // KV_HEADS
    kv_head = q_head // kv_group

    q_offset = (batch * Q_HEADS + q_head).to(tl.int64) * SEQ_LEN * HEAD_DIM
    kv_offset = (batch * KV_HEADS + kv_head).to(tl.int64) * SEQ_LEN * HEAD_DIM

    Q_desc = tl.make_tensor_descriptor(
        Q + q_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    K_desc = tl.make_tensor_descriptor(
        K + kv_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    V_desc = tl.make_tensor_descriptor(
        V + kv_offset,
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

        qk = tl.dot(q, k) * sm_scale

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
        Out + q_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    tl.store_tensor_descriptor(Out_desc, [start_m * BLOCK_M, 0], acc.to(Out.type.element_ty))


def llama3_gqa_attention(q, k, v, causal=True, sm_scale=None):
    assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4
    batch, q_heads, seq_len, head_dim = q.shape
    k_batch, kv_heads, k_seq_len, k_head_dim = k.shape
    assert v.shape == k.shape
    assert batch == k_batch and seq_len == k_seq_len and head_dim == k_head_dim
    assert q_heads % kv_heads == 0

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    q_r = q.contiguous().view(batch * q_heads, seq_len, head_dim)
    k_r = k.contiguous().view(batch * kv_heads, seq_len, head_dim)
    v_r = v.contiguous().view(batch * kv_heads, seq_len, head_dim)
    out_r = torch.empty_like(q_r)

    grid = lambda META: (triton.cdiv(seq_len, META["BLOCK_M"]), batch * q_heads)

    _llama3_gqa_attn_fwd[grid](
        q_r,
        k_r,
        v_r,
        out_r,
        sm_scale,
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim,
        Q_HEADS=q_heads,
        KV_HEADS=kv_heads,
        CAUSAL=causal,
    )

    return out_r.view(batch, q_heads, seq_len, head_dim)


def reference_gqa_attention(q, k, v, causal=True, sm_scale=None):
    batch, q_heads, seq_len, head_dim = q.shape
    kv_heads = k.shape[1]
    group = q_heads // kv_heads

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    out = torch.empty_like(q)
    causal_mask = None
    if causal:
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool)
        )

    for qh in range(q_heads):
        kvh = qh // group
        scores = torch.matmul(
            q[:, qh].float(),
            k[:, kvh].float().transpose(-2, -1),
        ) * sm_scale
        if causal:
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[:, qh] = torch.matmul(probs.to(v.dtype), v[:, kvh])

    return out


def main():
    parser = argparse.ArgumentParser(description="Generate a Llama3 GQA attention trace")
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--causal", action="store_true", default=False)
    args = parser.parse_args()

    if args.q_heads % args.kv_heads != 0:
        parser.error("--q-heads must be divisible by --kv-heads")

    print("=" * 80)
    print("Llama3 GQA Attention Trace Generator")
    print(
        f"  batch={args.batch}, q_heads={args.q_heads}, kv_heads={args.kv_heads}, "
        f"seq_len={args.seq_len}, head_dim={args.head_dim}, causal={args.causal}"
    )
    print("=" * 80)

    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    )

    causal_str = "_causal" if args.causal else ""
    subdir = (
        f"b{args.batch}_hq{args.q_heads}_hkv{args.kv_heads}"
        f"_seq{args.seq_len}_d{args.head_dim}{causal_str}"
    )
    output_dir = (
        TRITON_TRACE_DIR / f"triton_kernel_tracking/test_llama3_gqa_attn/{subdir}"
    ).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)

    tracker = TritonKernelTracker(output_dir, save_binaries=True, capture_args=True)
    tracker.disable()
    print(f"\nOutput directory: {output_dir}")

    torch.manual_seed(42)
    q = torch.randn(
        args.batch,
        args.q_heads,
        args.seq_len,
        args.head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    k = torch.randn(
        args.batch,
        args.kv_heads,
        args.seq_len,
        args.head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    v = torch.randn_like(k)

    print("\nLaunching Llama3 GQA attention kernel...")
    tri_out = llama3_gqa_attention(q, k, v, causal=args.causal)
    ref_out = reference_gqa_attention(q, k, v, causal=args.causal)

    if torch.allclose(tri_out.float(), ref_out.float(), atol=2e-2, rtol=2e-2):
        print("Kernel output verified - PASSED")
    else:
        max_diff = (tri_out.float() - ref_out.float()).abs().max().item()
        print(f"Kernel output mismatch - FAILED (max diff: {max_diff})")
        return 1

    tracker.enable()
    llama3_gqa_attention(q, k, v, causal=args.causal)
    tracker.save_summary()

    print(f"\nTrace saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
