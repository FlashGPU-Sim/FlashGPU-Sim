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
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

TRITON_TRACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRITON_TRACE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_triton_kernels import TritonKernelTracker
from test_llama3_gqa_attn import llama3_gqa_attention


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


def token_to_head(src, batch, seq_len, heads, head_dim):
    dst = torch.empty((batch * heads, seq_len, head_dim), device=src.device, dtype=src.dtype)
    total = dst.numel()
    block = 256
    _llama3_layer_token_to_head[(triton.cdiv(total, block),)](
        src, dst, total, SEQ_LEN=seq_len, HEADS=heads, HEAD_DIM=head_dim, BLOCK=block
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


def add_triton(a, b):
    out = torch.empty_like(a)
    total = a.numel()
    block = 256
    _llama3_layer_add[(triton.cdiv(total, block),)](a, b, out, total, BLOCK=block)
    return out


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

    return {
        "rms_attn": torch.ones(hidden, device=device, dtype=torch.float16),
        "rms_mlp": torch.ones(hidden, device=device, dtype=torch.float16),
        "wq_t": rand_t(q_heads * head_dim, hidden),
        "wk_t": rand_t(kv_heads * head_dim, hidden),
        "wv_t": rand_t(kv_heads * head_dim, hidden),
        "wo_t": rand_t(hidden, q_heads * head_dim),
        "w_gate_t": rand_t(intermediate, hidden),
        "w_up_t": rand_t(intermediate, hidden),
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


def main():
    parser = argparse.ArgumentParser(description="Generate a Triton Llama3 layer trace")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--intermediate", type=int, default=2816)
    parser.add_argument("--causal", action="store_true", default=False)
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
        f"intermediate={args.intermediate}, causal={args.causal}"
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
    output_dir = (
        TRITON_TRACE_DIR / f"triton_kernel_tracking/test_llama3_layer/{subdir}"
    ).resolve()
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

    print("\nLaunching Triton layer kernels...")
    tri_out = llama3_layer_triton(
        x,
        weights,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        causal=args.causal,
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

    tracker.enable()
    llama3_layer_triton(
        x,
        weights,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        causal=args.causal,
    )
    tracker.save_summary()

    print(f"\nTrace saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
