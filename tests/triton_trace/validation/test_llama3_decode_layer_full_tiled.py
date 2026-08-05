#!/usr/bin/env python3
"""
Full-tiled Triton Llama3-style single-layer decode benchmark.

This is the decode counterpart of test_llama3_layer.py --variant full_tiled.
Layer tensors and GEMMs use the same [M-block, N-block, 64, 128] tiled layout.
The current token's K/V cache write is fused into the tiled QKV projection.
The per-layer KV cache is tiled by batch:

    K/V cache tile: [batch / 64, kv_heads, kv_len, 64, head_dim]
"""

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

TRITON_TRACE_DIR = Path(__file__).resolve().parent.parent
TRACKING_ROOT = Path(
    os.environ.get("TRITON_TRACKING_ROOT", TRITON_TRACE_DIR / "triton_kernel_tracking")
).expanduser().resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent))

import TritonTrace
from test_llama3_decode_layer import llama3_decode_layer_reference
from test_llama3_layer import (
    FULL_TILE_M,
    FULL_TILE_N,
    make_full_tiled_weights,
    make_weights,
    matmul_residual_tile_triton,
    matmul_tile_triton,
    rmsnorm_tile_triton,
    silu_mul_split_tile_triton,
    tile_matrix_host,
    untile_matrix_host,
)


@triton.jit
def _llama3_layer_matmul_qkv_store_kv_tile(
    A_TILE,
    B_TILE,
    C_TILE,
    K_CACHE_TILE,
    V_CACHE_TILE,
    M_BLK: tl.constexpr,
    N_BLK: tl.constexpr,
    K_BLK: tl.constexpr,
    KV_LEN: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = M_BLK
    num_pid_n = N_BLK
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    desc_a = tl.make_tensor_descriptor(
        A_TILE,
        shape=[M_BLK * K_BLK * BLOCK_M, BLOCK_K],
        strides=[BLOCK_K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    desc_b = tl.make_tensor_descriptor(
        B_TILE,
        shape=[N_BLK * K_BLK * BLOCK_N, BLOCK_K],
        strides=[BLOCK_K, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )
    desc_c = tl.make_tensor_descriptor(
        C_TILE,
        shape=[M_BLK * N_BLK * BLOCK_M, BLOCK_N],
        strides=[BLOCK_N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_tile in range(0, K_BLK):
        a = tl.load_tensor_descriptor(
            desc_a, [(pid_m * K_BLK + k_tile) * BLOCK_M, 0]
        )
        b = tl.load_tensor_descriptor(
            desc_b, [(pid_n * K_BLK + k_tile) * BLOCK_N, 0]
        )
        acc = tl.dot(a, b.T, acc)

    out = acc.to(tl.float16)
    tl.store_tensor_descriptor(desc_c, [(pid_m * N_BLK + pid_n) * BLOCK_M, 0], out)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    k_head = pid_n - Q_HEADS
    v_head = pid_n - Q_HEADS - KV_HEADS
    k_idx = (
        ((((pid_m * KV_HEADS + k_head) * KV_LEN + (KV_LEN - 1)) * BLOCK_M
          + offs_m[:, None]) * BLOCK_N)
        + offs_n[None, :]
    )
    v_idx = (
        ((((pid_m * KV_HEADS + v_head) * KV_LEN + (KV_LEN - 1)) * BLOCK_M
          + offs_m[:, None]) * BLOCK_N)
        + offs_n[None, :]
    )
    k_mask = (pid_n >= Q_HEADS) & (pid_n < Q_HEADS + KV_HEADS)
    v_mask = (pid_n >= Q_HEADS + KV_HEADS) & (pid_n < Q_HEADS + 2 * KV_HEADS)
    tl.store(K_CACHE_TILE + k_idx, out, mask=k_mask)
    tl.store(V_CACHE_TILE + v_idx, out, mask=v_mask)


@triton.jit
def _llama3_gqa_decode_attn_qkv_tile(
    QKV_TILE,
    K_CACHE_TILE,
    V_CACHE_TILE,
    OUT_TILE,
    SM_SCALE: tl.constexpr,
    KV_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    B_TILE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    bi = pid % B_TILE
    q_head = (pid // B_TILE) % Q_HEADS
    b_blk = pid // (B_TILE * Q_HEADS)
    kv_group = Q_HEADS // KV_HEADS
    kv_head = q_head // kv_group
    h_total = Q_HEADS + 2 * KV_HEADS

    K_desc = tl.make_tensor_descriptor(
        K_CACHE_TILE
        + ((b_blk * KV_HEADS + kv_head) * KV_LEN * B_TILE + bi) * HEAD_DIM,
        shape=[KV_LEN, HEAD_DIM],
        strides=[B_TILE * HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    V_desc = tl.make_tensor_descriptor(
        V_CACHE_TILE
        + ((b_blk * KV_HEADS + kv_head) * KV_LEN * B_TILE + bi) * HEAD_DIM,
        shape=[KV_LEN, HEAD_DIM],
        strides=[B_TILE * HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    offs_d = tl.arange(0, HEAD_DIM)
    q_idx = ((b_blk * h_total + q_head) * B_TILE + bi) * HEAD_DIM + offs_d
    q = tl.load(QKV_TILE + q_idx).to(tl.float32)

    m_i = tl.full((), -float("inf"), dtype=tl.float32)
    l_i = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, KV_LEN, BLOCK_N):
        k = tl.load_tensor_descriptor(K_desc, [start_n, 0]).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * SM_SCALE

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)

        v = tl.load_tensor_descriptor(V_desc, [start_n, 0]).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    out_idx = ((b_blk * Q_HEADS + q_head) * B_TILE + bi) * HEAD_DIM + offs_d
    tl.store(OUT_TILE + out_idx, (acc / l_i).to(tl.float16))


def tile_decode_cache_host(cache):
    batch, kv_heads, kv_len, head_dim = cache.shape
    assert batch % FULL_TILE_M == 0
    assert head_dim == FULL_TILE_N
    return (
        cache.view(batch // FULL_TILE_M, FULL_TILE_M, kv_heads, kv_len, head_dim)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )


def matmul_qkv_store_kv_tile_triton(
    a_tile,
    wqkv_tile,
    k_cache_tile,
    v_cache_tile,
    *,
    kv_len,
    q_heads,
    kv_heads,
):
    assert a_tile.dim() == 4 and wqkv_tile.dim() == 4
    m_blk, k_blk, m_tile, k_tile = a_tile.shape
    qkv_blk, b_k_blk, n_tile, b_k_tile = wqkv_tile.shape
    assert qkv_blk == q_heads + 2 * kv_heads
    assert k_blk == b_k_blk
    assert m_tile == FULL_TILE_M
    assert n_tile == FULL_TILE_N
    assert k_tile == FULL_TILE_N and b_k_tile == FULL_TILE_N
    assert tuple(k_cache_tile.shape) == (m_blk, kv_heads, kv_len, FULL_TILE_M, FULL_TILE_N)
    assert v_cache_tile.shape == k_cache_tile.shape

    qkv_tile = torch.empty(
        (m_blk, qkv_blk, FULL_TILE_M, FULL_TILE_N),
        device=a_tile.device,
        dtype=torch.float16,
    )
    _llama3_layer_matmul_qkv_store_kv_tile[(m_blk * qkv_blk,)](
        a_tile,
        wqkv_tile,
        qkv_tile,
        k_cache_tile,
        v_cache_tile,
        M_BLK=m_blk,
        N_BLK=qkv_blk,
        K_BLK=k_blk,
        KV_LEN=kv_len,
        Q_HEADS=q_heads,
        KV_HEADS=kv_heads,
        BLOCK_M=FULL_TILE_M,
        BLOCK_N=FULL_TILE_N,
        BLOCK_K=FULL_TILE_N,
        GROUP_M=8,
    )
    return qkv_tile


def llama3_gqa_decode_attention_qkv_tile(qkv_tile, k_cache_tile, v_cache_tile, *, kv_len, q_heads, kv_heads, head_dim):
    assert qkv_tile.dim() == 4
    b_blk, h_total, b_tile, tile_n = qkv_tile.shape
    assert h_total == q_heads + 2 * kv_heads
    assert b_tile == FULL_TILE_M and tile_n == head_dim
    assert tuple(k_cache_tile.shape) == (b_blk, kv_heads, kv_len, FULL_TILE_M, head_dim)
    assert v_cache_tile.shape == k_cache_tile.shape

    out = torch.empty((b_blk, q_heads, FULL_TILE_M, head_dim), device=qkv_tile.device, dtype=qkv_tile.dtype)
    block_n = 64
    _llama3_gqa_decode_attn_qkv_tile[(b_blk * q_heads * FULL_TILE_M,)](
        qkv_tile,
        k_cache_tile,
        v_cache_tile,
        out,
        SM_SCALE=1.0 / math.sqrt(head_dim),
        KV_LEN=kv_len,
        HEAD_DIM=head_dim,
        Q_HEADS=q_heads,
        KV_HEADS=kv_heads,
        B_TILE=FULL_TILE_M,
        BLOCK_N=block_n,
    )
    return out


def llama3_decode_layer_full_tiled_triton(
    x_tile,
    weights,
    k_cache_tile,
    v_cache_tile,
    *,
    batch,
    kv_len,
    q_heads,
    kv_heads,
    head_dim,
):
    hidden = q_heads * head_dim
    b_blk = batch // FULL_TILE_M
    h_blk = hidden // FULL_TILE_N
    qkv_blk = q_heads + 2 * kv_heads
    assert tuple(x_tile.shape) == (b_blk, h_blk, FULL_TILE_M, FULL_TILE_N)
    assert weights["wqkv_tile"].shape == (qkv_blk, h_blk, FULL_TILE_N, FULL_TILE_N)
    assert weights["wo_tile"].shape == (h_blk, q_heads, FULL_TILE_N, FULL_TILE_N)

    norm1 = rmsnorm_tile_triton(x_tile, weights["rms_attn"])
    qkv_tile = matmul_qkv_store_kv_tile_triton(
        norm1,
        weights["wqkv_tile"],
        k_cache_tile,
        v_cache_tile,
        kv_len=kv_len,
        q_heads=q_heads,
        kv_heads=kv_heads,
    )
    attn_tile = llama3_gqa_decode_attention_qkv_tile(
        qkv_tile,
        k_cache_tile,
        v_cache_tile,
        kv_len=kv_len,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )
    h1 = matmul_residual_tile_triton(attn_tile, weights["wo_tile"], x_tile)

    norm2 = rmsnorm_tile_triton(h1, weights["rms_mlp"])
    gate_up = matmul_tile_triton(norm2, weights["w_gate_up_tile"])
    act = silu_mul_split_tile_triton(gate_up)
    return matmul_residual_tile_triton(act, weights["w_down_tile"], h1)


def validate_args(args):
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("--q-heads must be divisible by --kv-heads")
    if args.hidden != args.q_heads * args.head_dim:
        raise ValueError("--hidden must equal --q-heads * --head-dim")
    if args.batch % FULL_TILE_M != 0:
        raise ValueError(f"--batch must be a multiple of {FULL_TILE_M}")
    if args.head_dim != FULL_TILE_N:
        raise ValueError(f"--head-dim must be {FULL_TILE_N}")
    if args.hidden % FULL_TILE_N != 0:
        raise ValueError(f"--hidden must be a multiple of {FULL_TILE_N}")
    if args.intermediate % FULL_TILE_N != 0:
        raise ValueError(f"--intermediate must be a multiple of {FULL_TILE_N}")
    if args.kv_len <= 0:
        raise ValueError("--kv-len must be positive")
    if args.kv_len % 64 != 0:
        raise ValueError("--kv-len must be a multiple of 64")


def main():
    parser = argparse.ArgumentParser(description="Generate a full-tiled Triton Llama3 decode-layer trace")
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
    print("Full-Tiled Triton Llama3 Decode Layer Trace Generator")
    print(
        f"  batch={args.batch}, q_heads={args.q_heads}, kv_heads={args.kv_heads}, "
        f"q_len=1, kv_len={args.kv_len}, head_dim={args.head_dim}, "
        f"hidden={args.hidden}, intermediate={args.intermediate}, "
        f"m_tile={FULL_TILE_M}, n_tile={FULL_TILE_N}"
    )
    print("=" * 80)

    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    )

    subdir = (
        f"b{args.batch}_hq{args.q_heads}_hkv{args.kv_heads}_q1_kv{args.kv_len}"
        f"_d{args.head_dim}_h{args.hidden}_i{args.intermediate}_mt{FULL_TILE_M}_nt{FULL_TILE_N}"
    )
    output_dir = (
        TRACKING_ROOT / "test_llama3_decode_layer_full_tiled" / subdir
    ).resolve()

    tracker = None
    if not args.no_trace:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        tracker = TritonTrace.Tracker(output_dir, save_binaries=True, capture_args=True)
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

    x_tile = tile_matrix_host(x)
    tiled_weights = make_full_tiled_weights(weights)
    k_cache_tile = tile_decode_cache_host(k_cache)
    v_cache_tile = tile_decode_cache_host(v_cache)

    print("\nLaunching full-tiled Triton decode layer kernels...")
    tri_out_tile = llama3_decode_layer_full_tiled_triton(
        x_tile,
        tiled_weights,
        k_cache_tile.clone(),
        v_cache_tile.clone(),
        batch=args.batch,
        kv_len=args.kv_len,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
    )
    tri_out = untile_matrix_host(tri_out_tile, m=args.batch, n=args.hidden)
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
        print(f"Full-tiled decode layer output verified - PASSED (max_diff={max_diff:.5f}, rel_l2={rel_l2:.5f})")
    else:
        print(f"Full-tiled decode layer output mismatch - FAILED (max_diff={max_diff:.5f}, rel_l2={rel_l2:.5f})")
        return 1

    if args.no_trace:
        print("\nTrace generation skipped (--no-trace)")
        return 0

    tracker.enable()
    llama3_decode_layer_full_tiled_triton(
        x_tile,
        tiled_weights,
        k_cache_tile.clone(),
        v_cache_tile.clone(),
        batch=args.batch,
        kv_len=args.kv_len,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
    )
    tracker.save_summary()

    print(f"\nTrace saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
