#!/usr/bin/env python3
"""
Triton Llama3-style single-layer decode benchmark.

This traces one decode step for one decoder layer.  The layer input is the
current token only, while K/V are written into and read from a per-layer KV
cache.  The cache is contiguous:

    K/V cache: [batch, kv_heads, kv_len, head_dim]

kv_len is the number of keys visible to the current query after appending the
current token's K/V at position kv_len - 1.
"""

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

TRITON_TRACE_DIR = Path(__file__).resolve().parent.parent
TRACKING_ROOT = Path(
    os.environ.get("TRITON_TRACKING_ROOT", TRITON_TRACE_DIR / "triton_kernel_tracking")
).expanduser().resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tritontrace
from test_llama3_layer import (
    add_triton,
    head_to_token,
    make_weights,
    matmul_triton,
    rmsnorm_triton,
    silu_mul_split_triton,
    token_to_head_strided,
)


@triton.jit
def _llama3_decode_store_kv(
    K_NEW,
    V_NEW,
    K_CACHE,
    V_CACHE,
    TOTAL,
    KV_LEN: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    d = offs % HEAD_DIM
    tmp = offs // HEAD_DIM
    kv_head = tmp % KV_HEADS
    batch = tmp // KV_HEADS

    new_idx = (batch.to(tl.int64) * KV_HEADS + kv_head.to(tl.int64)) * HEAD_DIM + d
    cache_idx = (
        ((batch.to(tl.int64) * KV_HEADS + kv_head.to(tl.int64)) * KV_LEN + (KV_LEN - 1))
        * HEAD_DIM
        + d
    )

    k = tl.load(K_NEW + new_idx, mask=mask)
    v = tl.load(V_NEW + new_idx, mask=mask)
    tl.store(K_CACHE + cache_idx, k, mask=mask)
    tl.store(V_CACHE + cache_idx, v, mask=mask)


@triton.jit
def _llama3_gqa_decode_attn(
    Q,
    K_CACHE,
    V_CACHE,
    OUT,
    SM_SCALE: tl.constexpr,
    KV_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_qh = tl.program_id(0)
    q_head = batch_qh % Q_HEADS
    batch = batch_qh // Q_HEADS
    kv_group = Q_HEADS // KV_HEADS
    kv_head = q_head // kv_group

    offs_d = tl.arange(0, HEAD_DIM)
    q_base = (batch.to(tl.int64) * Q_HEADS + q_head.to(tl.int64)) * HEAD_DIM
    q = tl.load(Q + q_base + offs_d).to(tl.float32)

    m_i = tl.full((), -float("inf"), dtype=tl.float32)
    l_i = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_N)
    cache_base = (
        (batch.to(tl.int64) * KV_HEADS + kv_head.to(tl.int64)) * KV_LEN * HEAD_DIM
    )

    for start_n in range(0, KV_LEN, BLOCK_N):
        n = start_n + offs_n
        n_mask = n < KV_LEN
        cache_idx = cache_base + n[:, None].to(tl.int64) * HEAD_DIM + offs_d[None, :]

        k = tl.load(K_CACHE + cache_idx, mask=n_mask[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * SM_SCALE
        scores = tl.where(n_mask, scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)

        v = tl.load(V_CACHE + cache_idx, mask=n_mask[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(OUT + q_base + offs_d, out.to(tl.float16))


def write_decode_kv_cache(k_new, v_new, k_cache, v_cache, *, kv_len, kv_heads, head_dim):
    total = k_new.numel()
    block = 256
    _llama3_decode_store_kv[(triton.cdiv(total, block),)](
        k_new,
        v_new,
        k_cache,
        v_cache,
        total,
        KV_LEN=kv_len,
        KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        BLOCK=block,
    )


def llama3_gqa_decode_attention(q, k_cache, v_cache, *, kv_len, sm_scale=None):
    assert q.dim() == 4
    assert k_cache.dim() == 4 and v_cache.shape == k_cache.shape
    batch, q_heads, q_len, head_dim = q.shape
    k_batch, kv_heads, cache_len, k_head_dim = k_cache.shape
    assert q_len == 1
    assert batch == k_batch and cache_len == kv_len and head_dim == k_head_dim
    assert q_heads % kv_heads == 0

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    out = torch.empty_like(q)
    block_n = 64
    _llama3_gqa_decode_attn[(batch * q_heads,)](
        q,
        k_cache,
        v_cache,
        out,
        SM_SCALE=sm_scale,
        KV_LEN=kv_len,
        HEAD_DIM=head_dim,
        Q_HEADS=q_heads,
        KV_HEADS=kv_heads,
        BLOCK_N=block_n,
    )
    return out


def llama3_decode_layer_triton(
    x, weights, k_cache, v_cache, *, q_heads, kv_heads, head_dim, kv_len
):
    batch, hidden = x.shape
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    intermediate = weights["w_down_t"].shape[1]

    norm1 = rmsnorm_triton(x, weights["rms_attn"])
    qkv_tok = matmul_triton(norm1, weights["wqkv_t"])
    q = token_to_head_strided(qkv_tok, batch, 1, q_heads, head_dim, qkv_dim, 0)
    k_new = token_to_head_strided(qkv_tok, batch, 1, kv_heads, head_dim, qkv_dim, q_dim)
    v_new = token_to_head_strided(
        qkv_tok, batch, 1, kv_heads, head_dim, qkv_dim, q_dim + kv_dim
    )

    write_decode_kv_cache(
        k_new,
        v_new,
        k_cache,
        v_cache,
        kv_len=kv_len,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    attn = llama3_gqa_decode_attention(q, k_cache, v_cache, kv_len=kv_len)
    attn_tok = head_to_token(attn, batch, 1, q_heads, head_dim)
    attn_proj = matmul_triton(attn_tok, weights["wo_t"])
    h1 = add_triton(x, attn_proj)

    norm2 = rmsnorm_triton(h1, weights["rms_mlp"])
    gate_up = matmul_triton(norm2, weights["w_gate_up_t"])
    act = silu_mul_split_triton(gate_up, intermediate)
    down = matmul_triton(act, weights["w_down_t"])
    return add_triton(h1, down)


def rmsnorm_reference(x, weight, eps=1e-5):
    scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() * scale * weight.float()).to(x.dtype)


def llama3_decode_layer_reference(
    x, weights, k_cache, v_cache, *, q_heads, kv_heads, head_dim, kv_len
):
    batch, hidden = x.shape
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim

    norm1 = rmsnorm_reference(x, weights["rms_attn"])
    qkv_tok = torch.matmul(norm1.float(), weights["wqkv_t"].float().T).to(torch.float16)
    q = qkv_tok[:, :q_dim].contiguous().view(batch, q_heads, head_dim)
    k_new = qkv_tok[:, q_dim : q_dim + kv_dim].contiguous().view(batch, kv_heads, head_dim)
    v_new = qkv_tok[:, q_dim + kv_dim : q_dim + 2 * kv_dim].contiguous().view(
        batch, kv_heads, head_dim
    )

    k_cache = k_cache.clone()
    v_cache = v_cache.clone()
    k_cache[:, :, kv_len - 1, :] = k_new
    v_cache[:, :, kv_len - 1, :] = v_new

    group = q_heads // kv_heads
    k = k_cache.repeat_interleave(group, dim=1)
    v = v_cache.repeat_interleave(group, dim=1)
    scores = torch.sum(q.float().unsqueeze(2) * k.float(), dim=-1) / math.sqrt(head_dim)
    probs = torch.softmax(scores, dim=-1)
    attn = torch.matmul(probs.to(v.dtype).unsqueeze(2), v).squeeze(2)

    attn_tok = attn.contiguous().view(batch, hidden)
    attn_proj = torch.matmul(attn_tok.float(), weights["wo_t"].float().T).to(torch.float16)
    h1 = (x + attn_proj).to(torch.float16)

    norm2 = rmsnorm_reference(h1, weights["rms_mlp"])
    gate_up = torch.matmul(norm2.float(), weights["w_gate_up_t"].float().T)
    intermediate = weights["w_down_t"].shape[1]
    gate, up = gate_up[:, :intermediate], gate_up[:, intermediate:]
    act = (F.silu(gate) * up).to(torch.float16)
    down = torch.matmul(act.float(), weights["w_down_t"].float().T).to(torch.float16)
    return (h1 + down).to(torch.float16)


def is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def validate_args(args):
    for name in ("batch", "q_heads", "kv_heads"):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("--q-heads must be divisible by --kv-heads")
    if args.hidden != args.q_heads * args.head_dim:
        raise ValueError("--hidden must equal --q-heads * --head-dim")
    if not is_power_of_two(args.head_dim):
        raise ValueError("--head-dim must be a power of two")
    for name in ("kv_len", "hidden", "head_dim", "intermediate"):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("hidden", "head_dim", "intermediate"):
        value = getattr(args, name)
        if value % 64 != 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a multiple of 64")


def main():
    parser = argparse.ArgumentParser(description="Generate a Triton Llama3 decode-layer trace")
    parser.add_argument("--kv-len", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=14336)
    parser.add_argument("--no-trace", action="store_true", help="Run correctness only")
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    print("=" * 80)
    print("Triton Llama3 Decode Layer Trace Generator")
    print(
        f"  batch={args.batch}, q_heads={args.q_heads}, kv_heads={args.kv_heads}, "
        f"q_len=1, kv_len={args.kv_len}, head_dim={args.head_dim}, "
        f"hidden={args.hidden}, intermediate={args.intermediate}"
    )
    print("=" * 80)

    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    )

    subdir = (
        f"b{args.batch}_hq{args.q_heads}_hkv{args.kv_heads}_q1_kv{args.kv_len}"
        f"_d{args.head_dim}_h{args.hidden}_i{args.intermediate}"
    )
    output_dir = (TRACKING_ROOT / "test_llama3_decode_layer" / subdir).resolve()

    tracker = None
    if not args.no_trace:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        tracker = tritontrace.Tracker(output_dir, save_binaries=True, capture_args=True)
        tracker.disable()
        print(f"\nOutput directory: {output_dir}")

    torch.manual_seed(42)
    x = torch.randn(args.batch, args.hidden, device="cuda", dtype=torch.float16)
    weights = make_weights(
        args.hidden,
        args.q_heads,
        args.kv_heads,
        args.head_dim,
        args.intermediate,
        x.device,
    )
    k_cache = torch.randn(
        args.batch,
        args.kv_heads,
        args.kv_len,
        args.head_dim,
        device=x.device,
        dtype=torch.float16,
    )
    v_cache = torch.randn_like(k_cache)

    print("\nLaunching Triton decode layer kernels...")
    tri_out = llama3_decode_layer_triton(
        x,
        weights,
        k_cache.clone(),
        v_cache.clone(),
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        kv_len=args.kv_len,
    )
    ref_out = llama3_decode_layer_reference(
        x,
        weights,
        k_cache,
        v_cache,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        kv_len=args.kv_len,
    )
    torch.cuda.synchronize()

    max_diff = (tri_out.float() - ref_out.float()).abs().max().item()
    rel_l2 = ((tri_out.float() - ref_out.float()).norm() / ref_out.float().norm()).item()
    if torch.allclose(tri_out.float(), ref_out.float(), atol=8e-2, rtol=5e-2):
        print(f"Decode layer output verified - PASSED (max_diff={max_diff:.5f}, rel_l2={rel_l2:.5f})")
    else:
        print(f"Decode layer output mismatch - FAILED (max_diff={max_diff:.5f}, rel_l2={rel_l2:.5f})")
        return 1

    if args.no_trace:
        print("\nTrace generation skipped (--no-trace)")
        return 0

    tracker.enable()
    llama3_decode_layer_triton(
        x,
        weights,
        k_cache.clone(),
        v_cache.clone(),
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        kv_len=args.kv_len,
    )
    tracker.save_summary()

    print(f"\nTrace saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
