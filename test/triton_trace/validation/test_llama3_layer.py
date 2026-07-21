#!/usr/bin/env python3
"""
Triton-only Llama3-style decoder layer trace generator.

This is a simulation-oriented layer workload.  It implements RMSNorm, all linear
projections, grouped-query attention, residual adds, and the SwiGLU MLP with
Triton kernels.  PyTorch is used only for tensor allocation and correctness
reference.
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
sys.path.insert(0, str(TRITON_TRACE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_triton_kernels import TritonKernelTracker
from test_llama3_gqa_attn import (
    llama3_gqa_attention,
    llama3_gqa_attention_blocked_out,
    llama3_gqa_attention_qkv_tiled,
    llama3_gqa_attention_token_out,
)


@triton.jit
def _llama3_layer_rmsnorm(
    X,
    W,
    Y,
    N_COLS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N_COLS
    x = tl.load(X + row * N_COLS + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / N_COLS
    y = x * tl.rsqrt(mean_square + EPS) * w
    tl.store(Y + row * N_COLS + offs, y.to(tl.float16), mask=mask)


@triton.jit
def _llama3_layer_matmul(
    A,
    B_T,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    desc_a = tl.make_tensor_descriptor(
        A,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    desc_b = tl.make_tensor_descriptor(
        B_T,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )
    desc_c = tl.make_tensor_descriptor(
        C,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_tile * BLOCK_K
        a = tl.load_tensor_descriptor(desc_a, [pid_m * BLOCK_M, offs_k])
        b = tl.load_tensor_descriptor(desc_b, [pid_n * BLOCK_N, offs_k])
        acc = tl.dot(a, b.T, acc)

    tl.store_tensor_descriptor(
        desc_c, [pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(tl.float16)
    )


@triton.jit
def _llama3_layer_matmul_qkv_head_out(
    A,
    B_T,
    Q,
    K_OUT,
    V,
    M: tl.constexpr,
    QKV_N: tl.constexpr,
    K_IN: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(QKV_N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    desc_a = tl.make_tensor_descriptor(
        A,
        shape=[M, K_IN],
        strides=[K_IN, 1],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    desc_b = tl.make_tensor_descriptor(
        B_T,
        shape=[QKV_N, K_IN],
        strides=[K_IN, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_tile in range(0, tl.cdiv(K_IN, BLOCK_K)):
        offs_k = k_tile * BLOCK_K
        a = tl.load_tensor_descriptor(desc_a, [pid_m * BLOCK_M, offs_k])
        b = tl.load_tensor_descriptor(desc_b, [pid_n * BLOCK_N, offs_k])
        acc = tl.dot(a, b.T, acc)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = offs_m[:, None] < M
    col_mask = offs_n[None, :] < QKV_N
    b_idx = offs_m[:, None] // SEQ_LEN
    s_idx = offs_m[:, None] - b_idx * SEQ_LEN
    d_idx = offs_n[None, :] % HEAD_DIM
    q_dim = Q_HEADS * HEAD_DIM
    kv_dim = KV_HEADS * HEAD_DIM

    q_mask = row_mask & col_mask & (offs_n[None, :] < q_dim)
    q_h = offs_n[None, :] // HEAD_DIM
    q_idx = (
        ((b_idx.to(tl.int64) * Q_HEADS + q_h.to(tl.int64)) * SEQ_LEN
         + s_idx.to(tl.int64))
        * HEAD_DIM
        + d_idx.to(tl.int64)
    )
    tl.store(Q + q_idx, acc.to(tl.float16), mask=q_mask)

    k_n = offs_n[None, :] - q_dim
    k_mask = row_mask & col_mask & (k_n >= 0) & (k_n < kv_dim)
    k_h = k_n // HEAD_DIM
    k_d = k_n % HEAD_DIM
    k_idx = (
        ((b_idx.to(tl.int64) * KV_HEADS + k_h.to(tl.int64)) * SEQ_LEN
         + s_idx.to(tl.int64))
        * HEAD_DIM
        + k_d.to(tl.int64)
    )
    tl.store(K_OUT + k_idx, acc.to(tl.float16), mask=k_mask)

    v_n = offs_n[None, :] - q_dim - kv_dim
    v_mask = row_mask & col_mask & (v_n >= 0) & (v_n < kv_dim)
    v_h = v_n // HEAD_DIM
    v_d = v_n % HEAD_DIM
    v_idx = (
        ((b_idx.to(tl.int64) * KV_HEADS + v_h.to(tl.int64)) * SEQ_LEN
         + s_idx.to(tl.int64))
        * HEAD_DIM
        + v_d.to(tl.int64)
    )
    tl.store(V + v_idx, acc.to(tl.float16), mask=v_mask)


@triton.jit
def _llama3_layer_matmul_o_blocked_in(
    A_BLK,
    B_T,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K_IN: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    S_TILE: tl.constexpr,
    H_TILE: tl.constexpr,
    S_BLK: tl.constexpr,
    H_BLK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    desc_b = tl.make_tensor_descriptor(
        B_T,
        shape=[N, K_IN],
        strides=[K_IN, 1],
        block_shape=[BLOCK_N, BLOCK_K],
    )
    desc_c = tl.make_tensor_descriptor(
        C,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    b_idx = offs_m // SEQ_LEN
    s_idx = offs_m - b_idx * SEQ_LEN
    sb = s_idx // S_TILE
    si = s_idx - sb * S_TILE

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_tile in range(0, tl.cdiv(K_IN, BLOCK_K)):
        offs_k = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        h_idx = offs_k // HEAD_DIM
        d_idx = offs_k - h_idx * HEAD_DIM
        hb = h_idx // H_TILE
        hi = h_idx - hb * H_TILE
        a_idx = (
            (((((b_idx[:, None].to(tl.int64) * S_BLK + sb[:, None].to(tl.int64))
                * H_BLK + hb[None, :].to(tl.int64))
               * S_TILE + si[:, None].to(tl.int64))
              * H_TILE + hi[None, :].to(tl.int64))
             * HEAD_DIM)
            + d_idx[None, :].to(tl.int64)
        )
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K_IN) & (
            h_idx[None, :] < Q_HEADS
        )
        a = tl.load(A_BLK + a_idx, mask=a_mask, other=0.0)
        b = tl.load_tensor_descriptor(desc_b, [pid_n * BLOCK_N, k_tile * BLOCK_K])
        acc = tl.dot(a, b.T, acc)

    tl.store_tensor_descriptor(
        desc_c, [pid_m * BLOCK_M, pid_n * BLOCK_N], acc.to(tl.float16)
    )


@triton.jit
def _llama3_layer_matmul_tile(
    A_TILE,
    B_TILE,
    C_TILE,
    M_BLK: tl.constexpr,
    N_BLK: tl.constexpr,
    K_BLK: tl.constexpr,
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

    tl.store_tensor_descriptor(
        desc_c, [(pid_m * N_BLK + pid_n) * BLOCK_M, 0], acc.to(tl.float16)
    )


@triton.jit
def _llama3_layer_matmul_residual_tile(
    A_TILE,
    B_TILE,
    RESIDUAL_TILE,
    C_TILE,
    M_BLK: tl.constexpr,
    N_BLK: tl.constexpr,
    K_BLK: tl.constexpr,
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
    desc_residual = tl.make_tensor_descriptor(
        RESIDUAL_TILE,
        shape=[M_BLK * N_BLK * BLOCK_M, BLOCK_N],
        strides=[BLOCK_N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
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

    residual = tl.load_tensor_descriptor(
        desc_residual, [(pid_m * N_BLK + pid_n) * BLOCK_M, 0]
    )
    out = acc.to(tl.float16) + residual
    tl.store_tensor_descriptor(desc_c, [(pid_m * N_BLK + pid_n) * BLOCK_M, 0], out)


@triton.jit
def _llama3_layer_rmsnorm_tile(
    X_TILE,
    W,
    Y_TILE,
    M: tl.constexpr,
    N_BLK: tl.constexpr,
    M_TILE: tl.constexpr,
    N_TILE: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    m_blk = row // M_TILE
    mi = row - m_blk * M_TILE
    offs = tl.arange(0, N_TILE)
    sum_sq = tl.full((), 0.0, dtype=tl.float32)

    for n_blk in range(0, N_BLK):
        idx = ((m_blk * N_BLK + n_blk) * M_TILE + mi) * N_TILE + offs
        x = tl.load(X_TILE + idx).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)

    scale = tl.rsqrt(sum_sq / (N_BLK * N_TILE) + EPS)
    for n_blk in range(0, N_BLK):
        idx = ((m_blk * N_BLK + n_blk) * M_TILE + mi) * N_TILE + offs
        w = tl.load(W + n_blk * N_TILE + offs).to(tl.float32)
        x = tl.load(X_TILE + idx).to(tl.float32)
        tl.store(Y_TILE + idx, (x * scale * w).to(tl.float16))


@triton.jit
def _llama3_layer_silu_mul_split_tile(
    GATE_UP_TILE,
    OUT_TILE,
    TOTAL,
    I_BLK: tl.constexpr,
    M_TILE: tl.constexpr,
    N_TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    d = offs % N_TILE
    tmp = offs // N_TILE
    mi = tmp % M_TILE
    i_blk = (tmp // M_TILE) % I_BLK
    m_blk = tmp // (M_TILE * I_BLK)

    gate_idx = ((m_blk * (2 * I_BLK) + i_blk) * M_TILE + mi) * N_TILE + d
    up_idx = ((m_blk * (2 * I_BLK) + i_blk + I_BLK) * M_TILE + mi) * N_TILE + d
    gate = tl.load(GATE_UP_TILE + gate_idx, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GATE_UP_TILE + up_idx, mask=mask, other=0.0).to(tl.float32)
    silu = gate / (1.0 + tl.exp(-gate))
    tl.store(OUT_TILE + offs, (silu * up).to(tl.float16), mask=mask)


@triton.jit
def _llama3_layer_token_to_head(
    SRC,
    DST,
    TOTAL,
    SEQ_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    d = offs % HEAD_DIM
    tmp = offs // HEAD_DIM
    s = tmp % SEQ_LEN
    h = (tmp // SEQ_LEN) % HEADS
    b = tmp // (SEQ_LEN * HEADS)
    src_idx = ((b * SEQ_LEN + s) * HEADS + h) * HEAD_DIM + d
    val = tl.load(SRC + src_idx, mask=mask)
    tl.store(DST + offs, val, mask=mask)


@triton.jit
def _llama3_layer_token_to_head_strided(
    SRC,
    DST,
    TOTAL,
    SRC_ROW_STRIDE: tl.constexpr,
    SRC_COL_OFFSET: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    d = offs % HEAD_DIM
    tmp = offs // HEAD_DIM
    s = tmp % SEQ_LEN
    h = (tmp // SEQ_LEN) % HEADS
    b = tmp // (SEQ_LEN * HEADS)
    src_idx = (b * SEQ_LEN + s) * SRC_ROW_STRIDE + SRC_COL_OFFSET + h * HEAD_DIM + d
    val = tl.load(SRC + src_idx, mask=mask)
    tl.store(DST + offs, val, mask=mask)


@triton.jit
def _llama3_layer_head_to_token(
    SRC,
    DST,
    TOTAL,
    SEQ_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    d = offs % HEAD_DIM
    tmp = offs // HEAD_DIM
    h = tmp % HEADS
    s = (tmp // HEADS) % SEQ_LEN
    b = tmp // (HEADS * SEQ_LEN)
    src_idx = ((b * HEADS + h) * SEQ_LEN + s) * HEAD_DIM + d
    val = tl.load(SRC + src_idx, mask=mask)
    tl.store(DST + offs, val, mask=mask)


@triton.jit
def _llama3_layer_silu_mul(GATE, UP, OUT, N_ELEMS, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_ELEMS
    gate = tl.load(GATE + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(UP + offs, mask=mask, other=0.0).to(tl.float32)
    silu = gate / (1.0 + tl.exp(-gate))
    tl.store(OUT + offs, (silu * up).to(tl.float16), mask=mask)


@triton.jit
def _llama3_layer_silu_mul_split(
    GATE_UP,
    OUT,
    ROWS: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = ROWS * INTERMEDIATE
    mask = offs < total
    j = offs % INTERMEDIATE
    row = offs // INTERMEDIATE
    row_base = row * INTERMEDIATE * 2
    gate = tl.load(GATE_UP + row_base + j, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GATE_UP + row_base + INTERMEDIATE + j, mask=mask, other=0.0).to(tl.float32)
    silu = gate / (1.0 + tl.exp(-gate))
    tl.store(OUT + offs, (silu * up).to(tl.float16), mask=mask)


@triton.jit
def _llama3_layer_add(A, B, OUT, N_ELEMS, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_ELEMS
    a = tl.load(A + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)
    tl.store(OUT + offs, a + b, mask=mask)


def _ceil_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def rmsnorm_triton(x, weight, eps=1e-5):
    rows, cols = x.shape
    y = torch.empty_like(x)
    block = _ceil_pow2(cols)
    _llama3_layer_rmsnorm[(rows,)](
        x, weight, y, N_COLS=cols, EPS=eps, BLOCK=block
    )
    return y


def matmul_triton(a, b_t):
    m, k = a.shape
    n, b_k = b_t.shape
    assert k == b_k
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)
    block_m = 64
    block_n = 128
    block_k = 64
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _llama3_layer_matmul[grid](
        a,
        b_t,
        c,
        M=m,
        N=n,
        K=k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
    )
    return c


def matmul_qkv_head_out_triton(a, wqkv_t, *, batch, seq_len, q_heads, kv_heads, head_dim):
    m, k = a.shape
    qkv_n, b_k = wqkv_t.shape
    assert k == b_k
    assert m == batch * seq_len
    assert qkv_n == (q_heads + 2 * kv_heads) * head_dim

    q = torch.empty((batch, q_heads, seq_len, head_dim), device=a.device, dtype=torch.float16)
    k_out = torch.empty((batch, kv_heads, seq_len, head_dim), device=a.device, dtype=torch.float16)
    v = torch.empty_like(k_out)
    block_m = 64
    block_n = 128
    block_k = 64
    grid = (triton.cdiv(m, block_m) * triton.cdiv(qkv_n, block_n),)
    _llama3_layer_matmul_qkv_head_out[grid](
        a,
        wqkv_t,
        q,
        k_out,
        v,
        M=m,
        QKV_N=qkv_n,
        K_IN=k,
        SEQ_LEN=seq_len,
        Q_HEADS=q_heads,
        KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
    )
    return q, k_out, v


def matmul_o_blocked_in_triton(
    attn_blk,
    wo_t,
    *,
    batch,
    seq_len,
    q_heads,
    head_dim,
    s_tile,
    h_tile,
):
    n, k = wo_t.shape
    hidden = q_heads * head_dim
    assert k == hidden
    m = batch * seq_len
    s_blk = triton.cdiv(seq_len, s_tile)
    h_blk = triton.cdiv(q_heads, h_tile)
    assert tuple(attn_blk.shape) == (batch, s_blk, h_blk, s_tile, h_tile, head_dim)

    c = torch.empty((m, n), device=attn_blk.device, dtype=torch.float16)
    block_m = 64
    block_n = 128
    block_k = 64
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _llama3_layer_matmul_o_blocked_in[grid](
        attn_blk,
        wo_t,
        c,
        M=m,
        N=n,
        K_IN=k,
        SEQ_LEN=seq_len,
        Q_HEADS=q_heads,
        HEAD_DIM=head_dim,
        S_TILE=s_tile,
        H_TILE=h_tile,
        S_BLK=s_blk,
        H_BLK=h_blk,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
    )
    return c


def token_to_head(src, batch, seq_len, heads, head_dim):
    dst = torch.empty((batch * heads, seq_len, head_dim), device=src.device, dtype=src.dtype)
    total = dst.numel()
    block = 256
    _llama3_layer_token_to_head[(triton.cdiv(total, block),)](
        src, dst, total, SEQ_LEN=seq_len, HEADS=heads, HEAD_DIM=head_dim, BLOCK=block
    )
    return dst.view(batch, heads, seq_len, head_dim)


def token_to_head_strided(src, batch, seq_len, heads, head_dim, row_stride, col_offset):
    dst = torch.empty((batch * heads, seq_len, head_dim), device=src.device, dtype=src.dtype)
    total = dst.numel()
    block = 256
    _llama3_layer_token_to_head_strided[(triton.cdiv(total, block),)](
        src,
        dst,
        total,
        SRC_ROW_STRIDE=row_stride,
        SRC_COL_OFFSET=col_offset,
        SEQ_LEN=seq_len,
        HEADS=heads,
        HEAD_DIM=head_dim,
        BLOCK=block,
    )
    return dst.view(batch, heads, seq_len, head_dim)


def head_to_token(src, batch, seq_len, heads, head_dim):
    dst = torch.empty((batch * seq_len, heads * head_dim), device=src.device, dtype=src.dtype)
    total = dst.numel()
    block = 256
    _llama3_layer_head_to_token[(triton.cdiv(total, block),)](
        src, dst, total, SEQ_LEN=seq_len, HEADS=heads, HEAD_DIM=head_dim, BLOCK=block
    )
    return dst


def silu_mul_triton(gate, up):
    out = torch.empty_like(gate)
    total = gate.numel()
    block = 256
    _llama3_layer_silu_mul[(triton.cdiv(total, block),)](
        gate, up, out, total, BLOCK=block
    )
    return out


def silu_mul_split_triton(gate_up, intermediate):
    rows, two_intermediate = gate_up.shape
    assert two_intermediate == 2 * intermediate
    out = torch.empty((rows, intermediate), device=gate_up.device, dtype=gate_up.dtype)
    total = out.numel()
    block = 256
    _llama3_layer_silu_mul_split[(triton.cdiv(total, block),)](
        gate_up, out, ROWS=rows, INTERMEDIATE=intermediate, BLOCK=block
    )
    return out


def add_triton(a, b):
    out = torch.empty_like(a)
    total = a.numel()
    block = 256
    _llama3_layer_add[(triton.cdiv(total, block),)](a, b, out, total, BLOCK=block)
    return out


FULL_TILE_M = 64
FULL_TILE_N = 128


def tile_matrix_host(mat, *, m_tile=FULL_TILE_M, n_tile=FULL_TILE_N):
    m, n = mat.shape
    assert m % m_tile == 0
    assert n % n_tile == 0
    return (
        mat.view(m // m_tile, m_tile, n // n_tile, n_tile)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def untile_matrix_host(tile, *, m, n, m_tile=FULL_TILE_M, n_tile=FULL_TILE_N):
    assert tile.shape == (m // m_tile, n // n_tile, m_tile, n_tile)
    return tile.permute(0, 2, 1, 3).contiguous().view(m, n)


def tile_weight_host(w_t, *, n_tile=FULL_TILE_N, k_tile=FULL_TILE_N):
    n, k = w_t.shape
    assert n % n_tile == 0
    assert k % k_tile == 0
    return (
        w_t.view(n // n_tile, n_tile, k // k_tile, k_tile)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def make_full_tiled_weights(weights):
    return {
        "rms_attn": weights["rms_attn"],
        "rms_mlp": weights["rms_mlp"],
        "wqkv_tile": tile_weight_host(weights["wqkv_t"]),
        "wo_tile": tile_weight_host(weights["wo_t"]),
        "w_gate_up_tile": tile_weight_host(weights["w_gate_up_t"]),
        "w_down_tile": tile_weight_host(weights["w_down_t"]),
    }


def matmul_tile_triton(a_tile, b_tile):
    assert a_tile.dim() == 4 and b_tile.dim() == 4
    m_blk, k_blk, m_tile, k_tile = a_tile.shape
    n_blk, b_k_blk, n_tile, b_k_tile = b_tile.shape
    assert k_blk == b_k_blk
    assert m_tile == FULL_TILE_M
    assert n_tile == FULL_TILE_N
    assert k_tile == FULL_TILE_N and b_k_tile == FULL_TILE_N

    c = torch.empty(
        (m_blk, n_blk, FULL_TILE_M, FULL_TILE_N),
        device=a_tile.device,
        dtype=torch.float16,
    )
    grid = (m_blk * n_blk,)
    _llama3_layer_matmul_tile[grid](
        a_tile,
        b_tile,
        c,
        M_BLK=m_blk,
        N_BLK=n_blk,
        K_BLK=k_blk,
        BLOCK_M=FULL_TILE_M,
        BLOCK_N=FULL_TILE_N,
        BLOCK_K=FULL_TILE_N,
        GROUP_M=8,
    )
    return c


def matmul_residual_tile_triton(a_tile, b_tile, residual_tile):
    assert a_tile.dim() == 4 and b_tile.dim() == 4 and residual_tile.dim() == 4
    m_blk, k_blk, m_tile, k_tile = a_tile.shape
    n_blk, b_k_blk, n_tile, b_k_tile = b_tile.shape
    assert residual_tile.shape == (m_blk, n_blk, FULL_TILE_M, FULL_TILE_N)
    assert k_blk == b_k_blk
    assert m_tile == FULL_TILE_M
    assert n_tile == FULL_TILE_N
    assert k_tile == FULL_TILE_N and b_k_tile == FULL_TILE_N

    c = torch.empty_like(residual_tile)
    grid = (m_blk * n_blk,)
    _llama3_layer_matmul_residual_tile[grid](
        a_tile,
        b_tile,
        residual_tile,
        c,
        M_BLK=m_blk,
        N_BLK=n_blk,
        K_BLK=k_blk,
        BLOCK_M=FULL_TILE_M,
        BLOCK_N=FULL_TILE_N,
        BLOCK_K=FULL_TILE_N,
        GROUP_M=8,
    )
    return c


def rmsnorm_tile_triton(x_tile, weight, eps=1e-5):
    assert x_tile.dim() == 4
    m_blk, n_blk, m_tile, n_tile = x_tile.shape
    assert m_tile == FULL_TILE_M and n_tile == FULL_TILE_N
    y = torch.empty_like(x_tile)
    _llama3_layer_rmsnorm_tile[(m_blk * m_tile,)](
        x_tile,
        weight,
        y,
        M=m_blk * m_tile,
        N_BLK=n_blk,
        M_TILE=m_tile,
        N_TILE=n_tile,
        EPS=eps,
    )
    return y


def add_tile_triton(a_tile, b_tile):
    assert a_tile.shape == b_tile.shape
    out = torch.empty_like(a_tile)
    total = a_tile.numel()
    block = 256
    _llama3_layer_add[(triton.cdiv(total, block),)](
        a_tile, b_tile, out, total, BLOCK=block
    )
    return out


def silu_mul_split_tile_triton(gate_up_tile):
    assert gate_up_tile.dim() == 4
    m_blk, two_i_blk, m_tile, n_tile = gate_up_tile.shape
    assert two_i_blk % 2 == 0
    i_blk = two_i_blk // 2
    out = torch.empty(
        (m_blk, i_blk, m_tile, n_tile),
        device=gate_up_tile.device,
        dtype=gate_up_tile.dtype,
    )
    total = out.numel()
    block = 256
    _llama3_layer_silu_mul_split_tile[(triton.cdiv(total, block),)](
        gate_up_tile,
        out,
        total,
        I_BLK=i_blk,
        M_TILE=m_tile,
        N_TILE=n_tile,
        BLOCK=block,
    )
    return out


def llama3_layer_full_tiled_triton(
    x_tile,
    weights,
    *,
    batch,
    seq_len,
    q_heads,
    kv_heads,
    head_dim,
    causal,
    s_tile=FULL_TILE_M,
):
    hidden = q_heads * head_dim
    assert s_tile == FULL_TILE_M
    assert head_dim == FULL_TILE_N
    assert hidden % FULL_TILE_N == 0
    assert seq_len % s_tile == 0
    m_blk = batch * seq_len // s_tile
    h_blk = hidden // FULL_TILE_N
    qkv_blk = q_heads + 2 * kv_heads
    intermediate_blk = weights["w_down_tile"].shape[1]
    assert tuple(x_tile.shape) == (m_blk, h_blk, FULL_TILE_M, FULL_TILE_N)
    assert weights["wqkv_tile"].shape == (qkv_blk, h_blk, FULL_TILE_N, FULL_TILE_N)
    assert weights["wo_tile"].shape == (h_blk, q_heads, FULL_TILE_N, FULL_TILE_N)

    norm1 = rmsnorm_tile_triton(x_tile, weights["rms_attn"])
    qkv_tile = matmul_tile_triton(norm1, weights["wqkv_tile"])
    attn_tile = llama3_gqa_attention_qkv_tiled(
        qkv_tile,
        batch=batch,
        seq_len=seq_len,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        causal=causal,
        s_tile=s_tile,
    )
    h1 = matmul_residual_tile_triton(attn_tile, weights["wo_tile"], x_tile)

    norm2 = rmsnorm_tile_triton(h1, weights["rms_mlp"])
    gate_up = matmul_tile_triton(norm2, weights["w_gate_up_tile"])
    assert gate_up.shape[1] == 2 * intermediate_blk
    act = silu_mul_split_tile_triton(gate_up)
    return matmul_residual_tile_triton(act, weights["w_down_tile"], h1)


def llama3_layer_triton(x, weights, *, q_heads, kv_heads, head_dim, causal):
    batch, seq_len, hidden = x.shape
    flat = x.contiguous().view(batch * seq_len, hidden)

    norm1 = rmsnorm_triton(flat, weights["rms_attn"])
    q_tok = matmul_triton(norm1, weights["wq_t"])
    k_tok = matmul_triton(norm1, weights["wk_t"])
    v_tok = matmul_triton(norm1, weights["wv_t"])

    q = token_to_head(q_tok, batch, seq_len, q_heads, head_dim)
    k = token_to_head(k_tok, batch, seq_len, kv_heads, head_dim)
    v = token_to_head(v_tok, batch, seq_len, kv_heads, head_dim)

    attn = llama3_gqa_attention(q, k, v, causal=causal)
    attn_tok = head_to_token(attn, batch, seq_len, q_heads, head_dim)
    attn_proj = matmul_triton(attn_tok, weights["wo_t"])
    h1 = add_triton(flat, attn_proj)

    norm2 = rmsnorm_triton(h1, weights["rms_mlp"])
    gate = matmul_triton(norm2, weights["w_gate_t"])
    up = matmul_triton(norm2, weights["w_up_t"])
    act = silu_mul_triton(gate, up)
    down = matmul_triton(act, weights["w_down_t"])
    out = add_triton(h1, down)
    return out.view(batch, seq_len, hidden)


def llama3_layer_packed_triton(x, weights, *, q_heads, kv_heads, head_dim, causal):
    batch, seq_len, hidden = x.shape
    flat = x.contiguous().view(batch * seq_len, hidden)
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    qkv_dim = q_dim + 2 * kv_dim
    intermediate = weights["w_down_t"].shape[1]

    norm1 = rmsnorm_triton(flat, weights["rms_attn"])
    qkv_tok = matmul_triton(norm1, weights["wqkv_t"])
    q = token_to_head_strided(qkv_tok, batch, seq_len, q_heads, head_dim, qkv_dim, 0)
    k = token_to_head_strided(qkv_tok, batch, seq_len, kv_heads, head_dim, qkv_dim, q_dim)
    v = token_to_head_strided(
        qkv_tok, batch, seq_len, kv_heads, head_dim, qkv_dim, q_dim + kv_dim
    )

    attn = llama3_gqa_attention(q, k, v, causal=causal)
    attn_tok = head_to_token(attn, batch, seq_len, q_heads, head_dim)
    attn_proj = matmul_triton(attn_tok, weights["wo_t"])
    h1 = add_triton(flat, attn_proj)

    norm2 = rmsnorm_triton(h1, weights["rms_mlp"])
    gate_up = matmul_triton(norm2, weights["w_gate_up_t"])
    act = silu_mul_split_triton(gate_up, intermediate)
    down = matmul_triton(act, weights["w_down_t"])
    out = add_triton(h1, down)
    return out.view(batch, seq_len, hidden)


def llama3_layer_tiled_triton(x, weights, *, q_heads, kv_heads, head_dim, causal):
    batch, seq_len, hidden = x.shape
    flat = x.contiguous().view(batch * seq_len, hidden)
    intermediate = weights["w_down_t"].shape[1]

    norm1 = rmsnorm_triton(flat, weights["rms_attn"])
    q, k, v = matmul_qkv_head_out_triton(
        norm1,
        weights["wqkv_t"],
        batch=batch,
        seq_len=seq_len,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )

    attn = llama3_gqa_attention(q, k, v, causal=causal)
    attn_tok = head_to_token(attn, batch, seq_len, q_heads, head_dim)
    attn_proj = matmul_triton(attn_tok, weights["wo_t"])
    h1 = add_triton(flat, attn_proj)

    norm2 = rmsnorm_triton(h1, weights["rms_mlp"])
    gate_up = matmul_triton(norm2, weights["w_gate_up_t"])
    act = silu_mul_split_triton(gate_up, intermediate)
    down = matmul_triton(act, weights["w_down_t"])
    out = add_triton(h1, down)
    return out.view(batch, seq_len, hidden)


def llama3_layer_token_out_triton(x, weights, *, q_heads, kv_heads, head_dim, causal):
    batch, seq_len, hidden = x.shape
    flat = x.contiguous().view(batch * seq_len, hidden)
    intermediate = weights["w_down_t"].shape[1]

    norm1 = rmsnorm_triton(flat, weights["rms_attn"])
    q, k, v = matmul_qkv_head_out_triton(
        norm1,
        weights["wqkv_t"],
        batch=batch,
        seq_len=seq_len,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )

    attn_tok = llama3_gqa_attention_token_out(q, k, v, causal=causal)
    attn_proj = matmul_triton(attn_tok, weights["wo_t"])
    h1 = add_triton(flat, attn_proj)

    norm2 = rmsnorm_triton(h1, weights["rms_mlp"])
    gate_up = matmul_triton(norm2, weights["w_gate_up_t"])
    act = silu_mul_split_triton(gate_up, intermediate)
    down = matmul_triton(act, weights["w_down_t"])
    out = add_triton(h1, down)
    return out.view(batch, seq_len, hidden)


def llama3_layer_blocked_triton(
    x,
    weights,
    *,
    q_heads,
    kv_heads,
    head_dim,
    causal,
    s_tile=64,
    h_tile=4,
):
    batch, seq_len, hidden = x.shape
    flat = x.contiguous().view(batch * seq_len, hidden)
    intermediate = weights["w_down_t"].shape[1]

    norm1 = rmsnorm_triton(flat, weights["rms_attn"])
    q, k, v = matmul_qkv_head_out_triton(
        norm1,
        weights["wqkv_t"],
        batch=batch,
        seq_len=seq_len,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )

    attn_blk = llama3_gqa_attention_blocked_out(
        q,
        k,
        v,
        causal=causal,
        s_tile=s_tile,
        h_tile=h_tile,
    )
    attn_proj = matmul_o_blocked_in_triton(
        attn_blk,
        weights["wo_t"],
        batch=batch,
        seq_len=seq_len,
        q_heads=q_heads,
        head_dim=head_dim,
        s_tile=s_tile,
        h_tile=h_tile,
    )
    h1 = add_triton(flat, attn_proj)

    norm2 = rmsnorm_triton(h1, weights["rms_mlp"])
    gate_up = matmul_triton(norm2, weights["w_gate_up_t"])
    act = silu_mul_split_triton(gate_up, intermediate)
    down = matmul_triton(act, weights["w_down_t"])
    out = add_triton(h1, down)
    return out.view(batch, seq_len, hidden)


def llama3_layer_variant_triton(
    x,
    weights,
    *,
    q_heads,
    kv_heads,
    head_dim,
    causal,
    variant,
    s_tile,
    h_tile,
):
    if variant == "baseline":
        return llama3_layer_triton(
            x, weights, q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim, causal=causal
        )
    if variant == "packed":
        return llama3_layer_packed_triton(
            x, weights, q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim, causal=causal
        )
    if variant == "tiled":
        return llama3_layer_tiled_triton(
            x, weights, q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim, causal=causal
        )
    if variant == "token_out":
        return llama3_layer_token_out_triton(
            x, weights, q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim, causal=causal
        )
    if variant == "blocked":
        return llama3_layer_blocked_triton(
            x,
            weights,
            q_heads=q_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            causal=causal,
            s_tile=s_tile,
            h_tile=h_tile,
        )
    if variant == "full_tiled":
        raise ValueError("full_tiled is called from main with pre-tiled inputs")
    raise ValueError(f"unknown variant: {variant}")


def llama3_layer_reference(x, weights, *, q_heads, kv_heads, head_dim, causal):
    batch, seq_len, hidden = x.shape
    flat = x.contiguous().view(batch * seq_len, hidden)

    def rmsnorm_ref(inp, weight, eps=1e-5):
        scale = torch.rsqrt(inp.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        return (inp.float() * scale * weight.float()).to(inp.dtype)

    norm1 = rmsnorm_ref(flat, weights["rms_attn"])
    q_tok = torch.matmul(norm1.float(), weights["wq_t"].float().T).to(torch.float16)
    k_tok = torch.matmul(norm1.float(), weights["wk_t"].float().T).to(torch.float16)
    v_tok = torch.matmul(norm1.float(), weights["wv_t"].float().T).to(torch.float16)
    q = q_tok.view(batch, seq_len, q_heads, head_dim).transpose(1, 2).contiguous()
    k = k_tok.view(batch, seq_len, kv_heads, head_dim).transpose(1, 2).contiguous()
    v = v_tok.view(batch, seq_len, kv_heads, head_dim).transpose(1, 2).contiguous()
    attn = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)
    attn_tok = attn.transpose(1, 2).contiguous().view(batch * seq_len, hidden)
    attn_proj = torch.matmul(attn_tok.float(), weights["wo_t"].float().T).to(torch.float16)
    h1 = (flat + attn_proj).to(torch.float16)

    norm2 = rmsnorm_ref(h1, weights["rms_mlp"])
    gate = torch.matmul(norm2.float(), weights["w_gate_t"].float().T)
    up = torch.matmul(norm2.float(), weights["w_up_t"].float().T)
    act = (F.silu(gate) * up).to(torch.float16)
    down = torch.matmul(act.float(), weights["w_down_t"].float().T).to(torch.float16)
    return (h1 + down).view(batch, seq_len, hidden).to(torch.float16)


def make_weights(hidden, q_heads, kv_heads, head_dim, intermediate, device):
    def rand_t(out_features, in_features):
        return (
            torch.randn(out_features, in_features, device=device, dtype=torch.float16)
            / math.sqrt(in_features)
        ).contiguous()

    wq_t = rand_t(q_heads * head_dim, hidden)
    wk_t = rand_t(kv_heads * head_dim, hidden)
    wv_t = rand_t(kv_heads * head_dim, hidden)
    w_gate_t = rand_t(intermediate, hidden)
    w_up_t = rand_t(intermediate, hidden)

    return {
        "rms_attn": torch.ones(hidden, device=device, dtype=torch.float16),
        "rms_mlp": torch.ones(hidden, device=device, dtype=torch.float16),
        "wq_t": wq_t,
        "wk_t": wk_t,
        "wv_t": wv_t,
        "wqkv_t": torch.cat((wq_t, wk_t, wv_t), dim=0).contiguous(),
        "wo_t": rand_t(hidden, q_heads * head_dim),
        "w_gate_t": w_gate_t,
        "w_up_t": w_up_t,
        "w_gate_up_t": torch.cat((w_gate_t, w_up_t), dim=0).contiguous(),
        "w_down_t": rand_t(hidden, intermediate),
    }


def validate_args(args):
    if args.q_heads % args.kv_heads != 0:
        raise ValueError("--q-heads must be divisible by --kv-heads")
    if args.hidden != args.q_heads * args.head_dim:
        raise ValueError("--hidden must equal --q-heads * --head-dim")
    for name in ("seq_len", "hidden", "head_dim", "intermediate"):
        value = getattr(args, name)
        if value % 64 != 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a multiple of 64")
    if args.s_tile % 64 != 0:
        raise ValueError("--s-tile must be a multiple of 64")
    if args.seq_len % args.s_tile != 0:
        raise ValueError("--seq-len must be divisible by --s-tile")
    if args.q_heads % args.h_tile != 0:
        raise ValueError("--q-heads must be divisible by --h-tile")
    if getattr(args, "variant", None) == "full_tiled":
        if args.s_tile != FULL_TILE_M:
            raise ValueError(f"--s-tile must be {FULL_TILE_M} for full_tiled")
        if args.head_dim != FULL_TILE_N:
            raise ValueError(f"--head-dim must be {FULL_TILE_N} for full_tiled")
        if args.hidden % FULL_TILE_N != 0:
            raise ValueError(f"--hidden must be a multiple of {FULL_TILE_N} for full_tiled")
        if args.intermediate % FULL_TILE_N != 0:
            raise ValueError(
                f"--intermediate must be a multiple of {FULL_TILE_N} for full_tiled"
            )


def tracking_subdir_for_variant(variant):
    if variant == "baseline":
        return "test_llama3_layer"
    return f"test_llama3_layer_{variant}"


def main():
    parser = argparse.ArgumentParser(description="Generate a Triton Llama3 layer trace")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=14336)
    parser.add_argument("--causal", action="store_true", default=False)
    parser.add_argument(
        "--variant",
        choices=("baseline", "packed", "tiled", "token_out", "blocked", "full_tiled"),
        default="baseline",
    )
    parser.add_argument("--s-tile", type=int, default=64)
    parser.add_argument("--h-tile", type=int, default=4)
    parser.add_argument("--no-trace", action="store_true", help="Run correctness only")
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    print("=" * 80)
    print("Triton Llama3 Layer Trace Generator")
    print(
        f"  batch={args.batch}, q_heads={args.q_heads}, kv_heads={args.kv_heads}, "
        f"seq_len={args.seq_len}, head_dim={args.head_dim}, hidden={args.hidden}, "
        f"intermediate={args.intermediate}, causal={args.causal}, "
        f"variant={args.variant}, s_tile={args.s_tile}, h_tile={args.h_tile}"
    )
    print("=" * 80)

    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    )

    causal_str = "_causal" if args.causal else ""
    subdir = (
        f"b{args.batch}_hq{args.q_heads}_hkv{args.kv_heads}_seq{args.seq_len}"
        f"_d{args.head_dim}_h{args.hidden}_i{args.intermediate}{causal_str}"
    )
    if args.variant == "blocked":
        subdir = f"{subdir}_st{args.s_tile}_ht{args.h_tile}"
    elif args.variant == "full_tiled":
        subdir = f"{subdir}_mt{FULL_TILE_M}_nt{FULL_TILE_N}"
    tracker = None
    output_dir = (TRACKING_ROOT / tracking_subdir_for_variant(args.variant) / subdir).resolve()
    if not args.no_trace:
        if output_dir.exists():
            shutil.rmtree(output_dir)

        tracker = TritonKernelTracker(output_dir, save_binaries=True, capture_args=True)
        tracker.disable()
        print(f"\nOutput directory: {output_dir}")

    torch.manual_seed(42)
    x = torch.randn(
        args.batch,
        args.seq_len,
        args.hidden,
        device="cuda",
        dtype=torch.float16,
    )
    weights = make_weights(
        args.hidden,
        args.q_heads,
        args.kv_heads,
        args.head_dim,
        args.intermediate,
        x.device,
    )
    tiled_x = None
    tiled_weights = None
    if args.variant == "full_tiled":
        flat = x.contiguous().view(args.batch * args.seq_len, args.hidden)
        tiled_x = tile_matrix_host(flat)
        tiled_weights = make_full_tiled_weights(weights)

    print("\nLaunching Triton layer kernels...")
    if args.variant == "full_tiled":
        tri_out_tile = llama3_layer_full_tiled_triton(
            tiled_x,
            tiled_weights,
            batch=args.batch,
            seq_len=args.seq_len,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            causal=args.causal,
            s_tile=args.s_tile,
        )
        tri_out = untile_matrix_host(
            tri_out_tile,
            m=args.batch * args.seq_len,
            n=args.hidden,
        ).view(args.batch, args.seq_len, args.hidden)
    else:
        tri_out = llama3_layer_variant_triton(
            x,
            weights,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            causal=args.causal,
            variant=args.variant,
            s_tile=args.s_tile,
            h_tile=args.h_tile,
        )
    ref_out = llama3_layer_reference(
        x,
        weights,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        causal=args.causal,
    )
    torch.cuda.synchronize()

    max_diff = (tri_out.float() - ref_out.float()).abs().max().item()
    rel_l2 = ((tri_out.float() - ref_out.float()).norm() / ref_out.float().norm()).item()
    if torch.allclose(tri_out.float(), ref_out.float(), atol=8e-2, rtol=5e-2):
        print(f"Layer output verified - PASSED (max_diff={max_diff:.5f}, rel_l2={rel_l2:.5f})")
    else:
        print(f"Layer output mismatch - FAILED (max_diff={max_diff:.5f}, rel_l2={rel_l2:.5f})")
        return 1

    if args.no_trace:
        print("\nTrace generation skipped (--no-trace)")
        return 0

    tracker.enable()
    if args.variant == "full_tiled":
        llama3_layer_full_tiled_triton(
            tiled_x,
            tiled_weights,
            batch=args.batch,
            seq_len=args.seq_len,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            causal=args.causal,
            s_tile=args.s_tile,
        )
    else:
        llama3_layer_variant_triton(
            x,
            weights,
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            causal=args.causal,
            variant=args.variant,
            s_tile=args.s_tile,
            h_tile=args.h_tile,
        )
    tracker.save_summary()

    print(f"\nTrace saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
