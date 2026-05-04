#!/usr/bin/env python3
"""
GPT-2 Small — Full LLM Inference with Real Pretrained Weights (Pure Triton Kernels)
======================================================================================
Loads GPT-2 small weights from HuggingFace, tokenizes input, runs the complete
forward pass through custom Triton kernels, and tracks every kernel launch with
TritonKernelTracker for GPGPU-Sim simulation.

Architecture (GPT-2 small):
  vocab_size=50257, n_embd=768, n_heads=12, n_layers=12, ffn_dim=3072, max_seq=1024

Triton kernels (all use TMA via tl.make_tensor_descriptor):
  1. embedding_kernel    — token + positional embedding lookup & add
  2. layernorm_kernel    — standard LayerNorm
  3. linear_kernel       — GEMM for all projections (x @ W^T + b)
  4. flash_attn_kernel   — causal Flash-Attention v2
  5. gelu_kernel         — GELU activation (tanh approximation)
  6. residual_add_kernel — in-place residual addition

TMA rule: make_tensor_descriptor base must be a static kernel-arg pointer.
Row/tile selection goes in the offset list of load/store_tensor_descriptor.
"""

import math
import shutil
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from transformers import GPT2Tokenizer, GPT2Model

TRITON_TRACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRITON_TRACE_DIR))

from track_triton_kernels import TritonKernelTracker

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# ==============================================================================
# ★ USER CONFIGURATION ★
# ==============================================================================
INPUT_TEXT     = "The sun sets over the horizon, casting a warm"
N_TRACK_LAYERS = 1
# ==============================================================================

VOCAB_SIZE = 50257
N_EMBD     = 768
N_HEADS    = 12
HEAD_DIM   = N_EMBD // N_HEADS   # 64
N_LAYERS   = 12
FFN_DIM    = N_EMBD * 4          # 3072


# ==============================================================================
# Kernel 1 — Embedding (token + positional)
#   Descriptors: 2-D [V,C] / [T,C], block [1, BLOCK_C]; row in offset[0].
# ==============================================================================
@triton.jit
def embedding_kernel(
    tokens_ptr,   # [T]      int32
    tok_emb_ptr,  # [V, C]   fp16
    pos_emb_ptr,  # [T, C]   fp16
    out_ptr,      # [T, C]   fp16
    T, C, V,
    BLOCK_C: tl.constexpr,
):
    t   = tl.program_id(0)
    tok = tl.load(tokens_ptr + t)

    te_desc = tl.make_tensor_descriptor(
        base=tok_emb_ptr, shape=[V, C], strides=[C, 1], block_shape=[1, BLOCK_C])
    pe_desc = tl.make_tensor_descriptor(
        base=pos_emb_ptr, shape=[T, C], strides=[C, 1], block_shape=[1, BLOCK_C])
    out_desc = tl.make_tensor_descriptor(
        base=out_ptr, shape=[T, C], strides=[C, 1], block_shape=[1, BLOCK_C])

    te = tl.reshape(tl.load_tensor_descriptor(te_desc, [tok, 0]), [BLOCK_C])
    pe = tl.reshape(tl.load_tensor_descriptor(pe_desc, [t,   0]), [BLOCK_C])
    tl.store_tensor_descriptor(out_desc, [t, 0], tl.reshape(te + pe, [1, BLOCK_C]))


def embedding_fwd(tokens, tok_emb, pos_emb):
    T_tokens = tokens.shape[0]
    C        = tok_emb.shape[1]
    V        = tok_emb.shape[0]
    max_T_pos = pos_emb.shape[0]

    if T_tokens > max_T_pos:
        raise ValueError(
            f"Token sequence length {T_tokens} exceeds positional embedding length "
            f"{max_T_pos}. Truncate the input sequence or use a model/configuration "
            f"with a larger max positional embedding."
        )

    T   = T_tokens
    out = torch.empty(T, C, device=DEVICE, dtype=tok_emb.dtype)
    embedding_kernel[(T,)](
        tokens, tok_emb, pos_emb, out,
        T, C, V, BLOCK_C=triton.next_power_of_2(C)
    )
    return out


# ==============================================================================
# Kernel 2 — LayerNorm  (y = (x−mean)/sqrt(var+eps) * w + b)
#   Descriptors: 2-D [T,C] for x/out; 1-D [C] for w/b.
#   Padding: when BLOCK_C > C, TMA zero-pads lanes [C, BLOCK_C). Those lanes
#   are explicitly masked before variance computation and normalisation.
# ==============================================================================
@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    T, C,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    t = tl.program_id(0)

    x_desc = tl.make_tensor_descriptor(
        base=x_ptr,   shape=[T, C], strides=[C, 1], block_shape=[1, BLOCK_C])
    out_desc = tl.make_tensor_descriptor(
        base=out_ptr, shape=[T, C], strides=[C, 1], block_shape=[1, BLOCK_C])
    w_desc = tl.make_tensor_descriptor(
        base=w_ptr, shape=[C], strides=[1], block_shape=[BLOCK_C])
    b_desc = tl.make_tensor_descriptor(
        base=b_ptr, shape=[C], strides=[1], block_shape=[BLOCK_C])

    x    = tl.reshape(tl.load_tensor_descriptor(x_desc, [t, 0]), [BLOCK_C]).to(tl.float32)
    mean = tl.sum(x, axis=0) / C
    x_c  = x - mean

    offsets = tl.arange(0, BLOCK_C)
    x_c_m   = tl.where(offsets < C, x_c, tl.zeros_like(x_c))
    var     = tl.sum(x_c_m * x_c_m, axis=0) / C
    x_n     = x_c_m / tl.sqrt(var + eps)

    w = tl.load_tensor_descriptor(w_desc, [0]).to(tl.float32)
    b = tl.load_tensor_descriptor(b_desc, [0]).to(tl.float32)
    y = (x_n * w + b).to(tl.float16)
    tl.store_tensor_descriptor(out_desc, [t, 0], tl.reshape(y, [1, BLOCK_C]))


def layernorm_fwd(x, w, b, eps=1e-5):
    T, C    = x.shape
    out     = torch.empty_like(x)
    BLOCK_C = triton.next_power_of_2(C)
    layernorm_kernel[(T,)](x, w, b, out, T, C, eps=eps, BLOCK_C=BLOCK_C)
    return out


# ==============================================================================
# Kernel 3 — Linear projection:  out = x @ W^T + b
#   x:[M,K], W:[N,K] → out:[M,N]
#   Descriptors: 2-D [M,K]/[N,K]/[M,N_stride]; 1-D [N] for bias.
#   N_stride = ceil(N/BN)*BN ensures TMA row-stride is 16-byte aligned.
# ==============================================================================
BM_FIXED = 64
BN_FIXED = 64
BK_FIXED = 32

@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    N_stride,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    X_desc = tl.make_tensor_descriptor(
        base=x_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    W_desc = tl.make_tensor_descriptor(
        base=w_ptr, shape=[N, K], strides=[K, 1], block_shape=[BN, BK])

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        xt  = tl.load_tensor_descriptor(X_desc, [pid_m * BM, k * BK])
        wt  = tl.load_tensor_descriptor(W_desc, [pid_n * BN, k * BK])
        acc = tl.dot(xt, tl.trans(wt), acc)

    if HAS_BIAS:
        b_desc = tl.make_tensor_descriptor(
            base=b_ptr, shape=[N], strides=[1], block_shape=[BN])
        bias = tl.load_tensor_descriptor(b_desc, [pid_n * BN])
        acc  = acc + bias[None, :].to(tl.float32)

    Out_desc = tl.make_tensor_descriptor(
        base=out_ptr, shape=[M, N_stride], strides=[N_stride, 1], block_shape=[BM, BN])
    tl.store_tensor_descriptor(Out_desc, [pid_m * BM, pid_n * BN], acc.to(tl.float16))


def linear_fwd(x, w, b=None):
    M, K     = x.shape
    N        = w.shape[0]
    N_stride = triton.cdiv(N, BN_FIXED) * BN_FIXED
    out      = torch.empty(M, N_stride, device=DEVICE, dtype=x.dtype)
    grid     = (triton.cdiv(M, BM_FIXED), triton.cdiv(N, BN_FIXED))
    dummy    = x.new_empty(0)
    linear_kernel[grid](x, w, dummy if b is None else b,
                        out, M, N, K, N_stride,
                        HAS_BIAS=(b is not None),
                        BM=BM_FIXED, BN=BN_FIXED, BK=BK_FIXED)
    return out[:, :N].contiguous()


# ==============================================================================
# Kernel 4 — Causal Flash-Attention 2
#   q, k, v: [H, T, D] contiguous fp16 → out: [H, T, D]
#   Grid: (cdiv(T, BLOCK_T), H)
#   Descriptors: 3-D [H,T,D], block [1, BLOCK_T, BLOCK_D]; head+row in offset.
#   Loop A: fully-past KV tiles (no mask); Loop B: diagonal tile (causal mask).
# ==============================================================================
@triton.jit
def flash_attn_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    H, T, D,
    scale,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_blk   = tl.program_id(0)
    h       = tl.program_id(1)
    q_start = q_blk * BLOCK_T

    Q_desc = tl.make_tensor_descriptor(
        base=q_ptr, shape=[H, T, D], strides=[T*D, D, 1], block_shape=[1, BLOCK_T, BLOCK_D])
    K_desc = tl.make_tensor_descriptor(
        base=k_ptr, shape=[H, T, D], strides=[T*D, D, 1], block_shape=[1, BLOCK_T, BLOCK_D])
    V_desc = tl.make_tensor_descriptor(
        base=v_ptr, shape=[H, T, D], strides=[T*D, D, 1], block_shape=[1, BLOCK_T, BLOCK_D])
    Out_desc = tl.make_tensor_descriptor(
        base=out_ptr, shape=[H, T, D], strides=[T*D, D, 1], block_shape=[1, BLOCK_T, BLOCK_D])

    Q   = tl.reshape(tl.load_tensor_descriptor(Q_desc, [h, q_start, 0]),
                     [BLOCK_T, BLOCK_D]).to(tl.float32) * scale
    m_i = tl.full((BLOCK_T,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_T,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)

    # Sub-loop A: fully-past KV tiles
    for kv in range(0, q_blk):
        K = tl.reshape(tl.load_tensor_descriptor(K_desc, [h, kv*BLOCK_T, 0]),
                       [BLOCK_T, BLOCK_D]).to(tl.float32)
        V = tl.reshape(tl.load_tensor_descriptor(V_desc, [h, kv*BLOCK_T, 0]),
                       [BLOCK_T, BLOCK_D])
        S     = tl.dot(Q, tl.trans(K))
        m_new = tl.maximum(m_i, tl.max(S, axis=1))
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(S - m_new[:, None])
        l_i   = alpha * l_i + tl.sum(p, axis=1)
        acc   = acc * alpha[:, None] + tl.dot(p.to(V.dtype), V).to(tl.float32)
        m_i   = m_new

    # Sub-loop B: diagonal tile with causal mask
    q_range  = q_start + tl.arange(0, BLOCK_T)
    kv_range = q_blk * BLOCK_T + tl.arange(0, BLOCK_T)
    K = tl.reshape(tl.load_tensor_descriptor(K_desc, [h, q_blk*BLOCK_T, 0]),
                   [BLOCK_T, BLOCK_D]).to(tl.float32)
    V = tl.reshape(tl.load_tensor_descriptor(V_desc, [h, q_blk*BLOCK_T, 0]),
                   [BLOCK_T, BLOCK_D])
    S     = tl.dot(Q, tl.trans(K))
    S     = tl.where(kv_range[None, :] <= q_range[:, None], S, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(S, axis=1))
    alpha = tl.exp(m_i - m_new)
    p     = tl.exp(S - m_new[:, None])
    l_i   = alpha * l_i + tl.sum(p, axis=1)
    acc   = acc * alpha[:, None] + tl.dot(p.to(V.dtype), V).to(tl.float32)

    out = (acc / tl.maximum(l_i[:, None], 1e-6)).to(tl.float16)
    tl.store_tensor_descriptor(Out_desc, [h, q_start, 0],
                               tl.reshape(out, [1, BLOCK_T, BLOCK_D]))


def flash_attn_fwd(q, k, v):
    H, T, D = q.shape
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    out   = torch.empty_like(q)
    scale = 1.0 / math.sqrt(D)
    BT    = max(triton.next_power_of_2(min(T, 32)), 16)
    BD    = triton.next_power_of_2(D)
    flash_attn_kernel[(triton.cdiv(T, BT), H)](
        q, k, v, out, H, T, D, scale, BLOCK_T=BT, BLOCK_D=BD)
    return out


# ==============================================================================
# Kernel 5 — GELU activation (tanh approximation)
#   Descriptors: 1-D [N], block [BLOCK].
# ==============================================================================
@triton.jit
def gelu_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(
        base=x_ptr,   shape=[N], strides=[1], block_shape=[BLOCK])
    out_desc = tl.make_tensor_descriptor(
        base=out_ptr, shape=[N], strides=[1], block_shape=[BLOCK])

    x   = tl.load_tensor_descriptor(x_desc, [pid * BLOCK]).to(tl.float32)
    c   = 0.7978845608028654
    v   = c * (x + 0.044715 * x * x * x)
    v   = tl.minimum(tl.maximum(v, -20.0), 20.0)
    e2v = tl.exp(2.0 * v)
    g   = 0.5 * x * (1.0 + (e2v - 1.0) / (e2v + 1.0))
    tl.store_tensor_descriptor(out_desc, [pid * BLOCK], g.to(tl.float16))


def gelu_fwd(x):
    flat  = x.contiguous().view(-1)
    out   = torch.empty_like(flat)
    N     = flat.numel()
    BLOCK = min(triton.next_power_of_2(N), 1024)
    gelu_kernel[(triton.cdiv(N, BLOCK),)](flat, out, N, BLOCK=BLOCK)
    return out.view(x.shape)


# ==============================================================================
# Kernel 6 — Residual add (in-place: x += y)
#   Descriptors: 1-D [N], block [BLOCK].
#   x_desc and out_desc share base=x_ptr but are separate descriptor objects.
# ==============================================================================
@triton.jit
def residual_add_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    x_desc = tl.make_tensor_descriptor(
        base=x_ptr, shape=[N], strides=[1], block_shape=[BLOCK])
    y_desc = tl.make_tensor_descriptor(
        base=y_ptr, shape=[N], strides=[1], block_shape=[BLOCK])
    out_desc = tl.make_tensor_descriptor(
        base=x_ptr, shape=[N], strides=[1], block_shape=[BLOCK])
    x = tl.load_tensor_descriptor(x_desc,   [pid * BLOCK])
    y = tl.load_tensor_descriptor(y_desc,   [pid * BLOCK])
    tl.store_tensor_descriptor(out_desc, [pid * BLOCK], x + y)


def residual_add(x, y):
    N     = x.numel()
    BLOCK = min(triton.next_power_of_2(N), 1024)
    residual_add_kernel[(triton.cdiv(N, BLOCK),)](x, y, N, BLOCK=BLOCK)
    return x


# ==============================================================================
# GPT-2 weight container — loads from HuggingFace
# HuggingFace Conv1D weight is [in, out]; transposed here to [out, in].
# ==============================================================================
class GPT2Weights:
    def __init__(self, model_name: str = "gpt2"):
        print(f"\n[GPT2Weights] Loading '{model_name}' from HuggingFace …")
        hf = GPT2Model.from_pretrained(model_name)
        hf.eval()
        sd = hf.state_dict()

        def get(key):
            return sd[key].to(torch.float16).to(DEVICE)

        def get_T(key):
            return sd[key].T.contiguous().to(torch.float16).to(DEVICE)

        self.tok_emb = get("wte.weight")   # [50257, 768]
        self.pos_emb = get("wpe.weight")   # [1024,  768]

        self.ln1_w, self.ln1_b = [], []
        self.ln2_w, self.ln2_b = [], []
        self.c_attn_w, self.c_attn_b = [], []
        self.c_proj_w, self.c_proj_b = [], []
        self.ffn_w1,   self.ffn_b1   = [], []
        self.ffn_w2,   self.ffn_b2   = [], []

        for i in range(N_LAYERS):
            p = f"h.{i}"
            self.ln1_w.append(get(f"{p}.ln_1.weight"))
            self.ln1_b.append(get(f"{p}.ln_1.bias"))
            self.ln2_w.append(get(f"{p}.ln_2.weight"))
            self.ln2_b.append(get(f"{p}.ln_2.bias"))
            self.c_attn_w.append(get_T(f"{p}.attn.c_attn.weight"))  # [2304, 768]
            self.c_attn_b.append(get(f"{p}.attn.c_attn.bias"))      # [2304]
            self.c_proj_w.append(get_T(f"{p}.attn.c_proj.weight"))  # [768,  768]
            self.c_proj_b.append(get(f"{p}.attn.c_proj.bias"))      # [768]
            self.ffn_w1.append(get_T(f"{p}.mlp.c_fc.weight"))       # [3072, 768]
            self.ffn_b1.append(get(f"{p}.mlp.c_fc.bias"))           # [3072]
            self.ffn_w2.append(get_T(f"{p}.mlp.c_proj.weight"))     # [768,  3072]
            self.ffn_b2.append(get(f"{p}.mlp.c_proj.bias"))         # [768]

        self.ln_f_w    = get("ln_f.weight")
        self.ln_f_b    = get("ln_f.bias")
        self.lm_head_w = self.tok_emb   # weight tying

        print(f"[GPT2Weights] Loaded {N_LAYERS} layers, embd={N_EMBD}, heads={N_HEADS}")
        param_bytes = sum(p.nbytes for p in sd.values())
        print(f"[GPT2Weights] Total parameters: ~{param_bytes / 1e6:.1f} MB")


# ==============================================================================
# GPT-2 forward pass — pure Triton
# ==============================================================================
def gpt2_forward_triton(tokens: torch.Tensor, w: GPT2Weights,
                        n_layers: int = N_LAYERS) -> torch.Tensor:
    """tokens: [T] int32 → logits: [T, VOCAB_SIZE] fp16"""
    T = tokens.shape[0]
    C = N_EMBD

    x = embedding_fwd(tokens, w.tok_emb, w.pos_emb)

    for layer in range(n_layers):
        # Attention sub-layer
        ln1  = layernorm_fwd(x, w.ln1_w[layer], w.ln1_b[layer])
        qkv  = linear_fwd(ln1, w.c_attn_w[layer], w.c_attn_b[layer])
        Q, K, V = (qkv[:, :C].contiguous(),
                   qkv[:, C:2*C].contiguous(),
                   qkv[:, 2*C:].contiguous())
        Q = Q.view(T, N_HEADS, HEAD_DIM).permute(1, 0, 2).contiguous()
        K = K.view(T, N_HEADS, HEAD_DIM).permute(1, 0, 2).contiguous()
        V = V.view(T, N_HEADS, HEAD_DIM).permute(1, 0, 2).contiguous()
        attn_out  = flash_attn_fwd(Q, K, V).permute(1, 0, 2).contiguous().view(T, C)
        attn_proj = linear_fwd(attn_out, w.c_proj_w[layer], w.c_proj_b[layer])
        residual_add(x, attn_proj)

        # FFN sub-layer
        ln2     = layernorm_fwd(x, w.ln2_w[layer], w.ln2_b[layer])
        ffn_h   = gelu_fwd(linear_fwd(ln2, w.ffn_w1[layer], w.ffn_b1[layer]))
        ffn_out = linear_fwd(ffn_h, w.ffn_w2[layer], w.ffn_b2[layer])
        residual_add(x, ffn_out)

    x = layernorm_fwd(x, w.ln_f_w, w.ln_f_b)
    return linear_fwd(x, w.lm_head_w)


# ==============================================================================
# HuggingFace reference forward (float32, for validation)
# ==============================================================================
def gpt2_forward_hf(tokens: torch.Tensor, n_layers: int = N_LAYERS) -> torch.Tensor:
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    model.eval()
    model.config.n_layer = n_layers
    model.transformer.h  = model.transformer.h[:n_layers]
    with torch.no_grad():
        out = model(tokens.long().unsqueeze(0))
    return out.logits.squeeze(0).half()


# ==============================================================================
# Helpers
# ==============================================================================
def validate(triton_out, ref_out, name, atol=0.01, rtol=0.1):
    ok     = torch.allclose(triton_out.float(), ref_out.float(), atol=atol, rtol=rtol)
    maxd   = (triton_out.float() - ref_out.float()).abs().max().item()
    status = "✅ PASSED" if ok else f"❌ FAILED  (max_diff={maxd:.4f})"
    print(f"  {status}  — {name}")
    return ok


def show_predictions(logits, tokenizer, top_k=5):
    probs = torch.softmax(logits[-1].float(), dim=-1)
    topk  = torch.topk(probs, top_k)
    print(f"\n  Top-{top_k} next token predictions:")
    for rank, (tok_id, prob) in enumerate(
            zip(topk.indices.tolist(), topk.values.tolist())):
        word = tokenizer.decode([tok_id]).strip()
        print(f"    #{rank+1}  '{word}'  (id={tok_id:5d}, prob={prob:.4f})")


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 80)
    print("GPT-2 Small — Full LLM Inference (Pure Triton, make_tensor_descriptor TMA)")
    print("=" * 80)
    print(f"\nInput text  : \"{INPUT_TEXT}\"")
    print(f"Track layers: {N_TRACK_LAYERS} / {N_LAYERS}")
    print(f"\nGPU    : {torch.cuda.get_device_name()}")
    print(f"Triton : {triton.__version__}")

    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(
            size, device="cuda", dtype=torch.int8))

    print("\n[1/4] Tokenizing input …")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    token_ids = tokenizer.encode(INPUT_TEXT)
    tokens    = torch.tensor(token_ids, dtype=torch.int32, device=DEVICE)
    T         = tokens.shape[0]
    print(f"  Tokens ({T}): {token_ids}")
    print(f"  Decoded: {[tokenizer.decode([t]) for t in token_ids]}")

    print("\n[2/4] Loading GPT-2 small pretrained weights …")
    w = GPT2Weights("gpt2")

    print("\n[3/4] Tracked forward pass (JIT + capture, tracker ON) …")
    output_dir = (TRITON_TRACE_DIR / "triton_kernel_tracking/gpt2_small").resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    tracker = TritonKernelTracker(output_dir, save_binaries=True, capture_args=True)
    print(f"Output directory: {output_dir}")

    with torch.no_grad():
        gpt2_forward_triton(tokens, w, n_layers=N_TRACK_LAYERS)
    torch.cuda.synchronize()
    tracker.disable()

    n_compiled = len(getattr(tracker, 'compiled_kernels', {}))
    n_launches  = len(getattr(tracker, 'kernel_launches',  []))
    print(f"  Captured: {n_compiled} kernel variants, {n_launches} launches.")
    tracker.save_summary()

    print(f"\n[4/4] Full {N_LAYERS}-layer forward pass (validation + predictions) …")
    with torch.no_grad():
        full_logits = gpt2_forward_triton(tokens, w, n_layers=N_LAYERS)
    torch.cuda.synchronize()

    with torch.no_grad():
        ref_logits = gpt2_forward_hf(tokens, n_layers=N_LAYERS)
    validate(full_logits, ref_logits, f"GPT-2 small ({N_LAYERS} layers)")

    print(f"\n  Next-token predictions (full {N_LAYERS}-layer GPT-2):")
    print(f"  Input: \"{INPUT_TEXT}\"")
    show_predictions(full_logits, tokenizer)

    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)
    print(f"\nInput          : \"{INPUT_TEXT}\"")
    print(f"Tokens         : {T}")
    print(f"Tracked layers : {N_TRACK_LAYERS}  → GPGPU-Sim harnesses in {output_dir}/launchers/")
    print(f"Validated      : {N_LAYERS} layers  (full GPT-2 small)")
    print(f"\nArtifacts saved to: {output_dir}")
    print("  binaries/   — PTX / CUBIN per kernel")
    print("  launchers/  — Standalone C++ harness per launch")
    print("  data/       — Serialized tensor inputs & outputs (.bin)")
    print("  tracking_summary.json")
    print("  tracking_report.txt")
    return 0


if __name__ == "__main__":
    exit(main())
