#!/usr/bin/env python3
"""
Example: Track Flash Attention kernel (Simplified Version)

This is a simplified Flash Attention implementation that avoids
the argument count mismatch issue by not passing stride parameters explicitly.

Triton will automatically infer strides for contiguous tensors.
"""

from pathlib import Path
import torch
import triton
import triton.language as tl

from track_triton_kernels import TritonKernelTracker


# ============================================================================
# Device and Backend Detection
# ============================================================================

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


# ============================================================================
# Set up Triton memory allocator for TMA descriptors
# ============================================================================

def init_triton_allocator():
    """Initialize Triton's memory allocator using PyTorch's CUDA allocator."""
    def alloc_fn(size, align, stream):
        # Allocate memory using PyTorch
        return torch.empty(size, dtype=torch.uint8, device=DEVICE).data_ptr()
    
    triton.set_allocator(alloc_fn)

# Initialize allocator at module load time
init_triton_allocator()


# ============================================================================
# Simplified Flash Attention Forward Kernel
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
    TMA provides hardware-accelerated async memory transfers on Hopper+ GPUs.
    """
    # Get program IDs
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Compute base pointer for this batch/head
    qkv_offset = off_hz.to(tl.int64) * SEQ_LEN * HEAD_DIM

    # Create tensor descriptor for Q using TMA
    # Shape: (SEQ_LEN, HEAD_DIM), we load a (BLOCK_M, HEAD_DIM) tile
    Q_desc = tl.make_tensor_descriptor(
        Q + qkv_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

    # Create tensor descriptor for K (load normally, transpose in registers)
    # TMA requires last stride = 1, so we load K normally and transpose after
    K_desc = tl.make_tensor_descriptor(
        K + qkv_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )

    # Create tensor descriptor for V
    V_desc = tl.make_tensor_descriptor(
        V + qkv_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )

    # Initialize accumulator and running statistics
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Load Q block via TMA descriptor
    q = tl.load_tensor_descriptor(Q_desc, [start_m * BLOCK_M, 0])

    # Determine loop bounds for causal masking
    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, SEQ_LEN)
    else:
        hi = SEQ_LEN

    # Offset arrays for causal masking
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Loop over K, V blocks
    for start_n in range(0, hi, BLOCK_N):
        # Load K block via TMA descriptor, then transpose for Q @ K^T
        k = tl.load_tensor_descriptor(K_desc, [start_n, 0])
        k = tl.trans(k)  # Transpose from (BLOCK_N, HEAD_DIM) to (HEAD_DIM, BLOCK_N)

        # Compute Q @ K^T
        qk = tl.dot(q, k)
        qk = qk * sm_scale

        # Apply causal mask if needed
        if CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        # Compute running max (for numerical stability)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        # Subtract max for numerical stability
        qk_normalized = tl.where(
            m_ij[:, None] == float("-inf"),
            float("-inf"),
            qk - m_ij[:, None]
        )

        # Compute attention weights
        p = tl.math.exp(qk_normalized)
        l_ij = tl.sum(p, 1)

        # Correction factor for running max
        alpha = tl.math.exp(m_i - m_ij)

        # Update accumulator with rescaling
        acc = acc * alpha[:, None]

        # Load V block via TMA descriptor
        v = tl.load_tensor_descriptor(V_desc, [start_n, 0])

        # Accumulate P @ V
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)

        # Update running statistics
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    # Normalize by softmax denominator
    acc = acc / l_i[:, None]

    # Create output tensor descriptor
    Out_desc = tl.make_tensor_descriptor(
        Out + qkv_offset,
        shape=[SEQ_LEN, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

    # Store output via TMA descriptor
    tl.store_tensor_descriptor(Out_desc, [start_m * BLOCK_M, 0], acc.to(Out.type.element_ty))


# ============================================================================
# Python Wrapper Functions
# ============================================================================

def flash_attention_forward(q, k, v, causal=False, sm_scale=None):
    """
    Flash Attention forward pass (simplified version).

    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim)
        k: Key tensor of shape (batch, heads, seq_len, head_dim)
        v: Value tensor of shape (batch, heads, seq_len, head_dim)
        causal: Whether to apply causal masking
        sm_scale: Softmax scale factor (default: 1/sqrt(head_dim))

    Returns:
        Output tensor of shape (batch, heads, seq_len, head_dim)
    """
    # Validate shapes
    assert q.dim() == 4, f"Expected 4D tensor, got {q.dim()}D"
    assert q.shape == k.shape == v.shape, "Q, K, V must have same shape"

    batch, heads, seq_len, head_dim = q.shape
    assert head_dim in {16, 32, 64, 128, 256}, f"Head dim must be power of 2, got {head_dim}"

    # Ensure tensors are contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Default scale
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    # Reshape to (batch * heads, seq_len, head_dim) for simpler kernel
    q_reshaped = q.view(batch * heads, seq_len, head_dim)
    k_reshaped = k.view(batch * heads, seq_len, head_dim)
    v_reshaped = v.view(batch * heads, seq_len, head_dim)

    # Allocate output
    o_reshaped = torch.empty_like(q_reshaped)

    # Launch kernel with fixed block size
    BLOCK_M = 64
    BLOCK_N = 64
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * heads)

    _flash_attn_fwd[grid](
        q_reshaped, k_reshaped, v_reshaped, o_reshaped,
        sm_scale,
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        CAUSAL=causal,
    )

    # Reshape output back
    o = o_reshaped.view(batch, heads, seq_len, head_dim)

    return o


def reference_attention(q, k, v, causal=False, sm_scale=None):
    """
    Reference attention implementation using PyTorch.

    Used for validation against the Triton kernel.
    """
    batch, heads, seq_len, head_dim = q.shape

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    # Compute attention scores: Q @ K^T
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sm_scale

    # Apply causal mask
    if causal:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float('-inf'))

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Compute output: attn_weights @ V
    output = torch.matmul(attn_weights.to(v.dtype), v)

    return output


# ============================================================================
# Test Functions
# ============================================================================

def validate_flash_attention(batch=2, heads=4, seq_len=512, head_dim=64, causal=False):
    """Validate Triton kernel against reference implementation."""
    print(f"\nValidating Flash Attention:")
    print(f"  Batch={batch}, Heads={heads}, SeqLen={seq_len}, HeadDim={head_dim}")
    print(f"  Causal={causal}")

    # Create random inputs
    torch.manual_seed(42)
    q = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)

    # Reference output
    ref_out = reference_attention(q, k, v, causal=causal)

    # Triton output
    tri_out = flash_attention_forward(q, k, v, causal=causal)

    # Debug: print statistics
    print(f"  Reference: min={ref_out.min():.6f}, max={ref_out.max():.6f}, mean={ref_out.mean():.6f}")
    print(f"  Triton:    min={tri_out.min():.6f}, max={tri_out.max():.6f}, mean={tri_out.mean():.6f}")

    # Compare
    if torch.allclose(tri_out.float(), ref_out.float(), atol=1e-2, rtol=1e-2):
        print("  Result: ✓ PASSED")
        return True
    else:
        max_diff = (tri_out.float() - ref_out.float()).abs().max().item()
        mean_diff = (tri_out.float() - ref_out.float()).abs().mean().item()
        print(f"  Result: ✗ FAILED (max diff: {max_diff:.6f}, mean diff: {mean_diff:.6f})")

        # Check for NaN
        if torch.isnan(tri_out).any():
            print(f"  WARNING: Triton output contains NaN!")
        if torch.isnan(ref_out).any():
            print(f"  WARNING: Reference output contains NaN!")

        return False


def benchmark_flash_attention(batch=4, heads=32, seq_len=1024, head_dim=64, causal=True):
    """Benchmark Flash Attention performance."""
    print(f"\nBenchmarking Flash Attention:")
    print(f"  Batch={batch}, Heads={heads}, SeqLen={seq_len}, HeadDim={head_dim}")
    print(f"  Causal={causal}")

    q = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)

    # Warmup
    for _ in range(3):
        _ = flash_attention_forward(q, k, v, causal=causal)
    torch.cuda.synchronize()

    # Benchmark
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: flash_attention_forward(q, k, v, causal=causal),
        quantiles=[0.5, 0.2, 0.8],
    )

    # Calculate FLOPS
    # Forward: 2 * batch * heads * seq_len^2 * head_dim (for Q@K^T and attn@V)
    flops = 2 * 2 * batch * heads * seq_len * seq_len * head_dim
    if causal:
        flops *= 0.5  # Only half the computation for causal

    tflops = flops / (ms * 1e-3) / 1e12

    print(f"  Execution time: {ms:.3f} ms (min: {min_ms:.3f}, max: {max_ms:.3f})")
    print(f"  Throughput: {tflops:.2f} TFLOPS")

    return ms, tflops


# ============================================================================
# Main Function with Kernel Tracking
# ============================================================================

def main():
    print("=" * 80)
    print("Triton Kernel Tracker - Flash Attention Example (Simplified)")
    print("=" * 80)

    # Initialize tracker
    output_dir = (Path(__file__).parent / "triton_kernel_tracking/example_flash_attention").resolve()
    tracker = TritonKernelTracker(output_dir, save_binaries=True, capture_args=True)
    tracker.disable()  # Disable during warmup
    print(f"\nOutput directory: {output_dir}")

    # Test configurations
    test_configs = [
        # (batch, heads, seq_len, head_dim, causal)
        (1, 4, 256, 64, False),   # Small non-causal
        (1, 4, 256, 64, True),    # Small causal
        (2, 8, 512, 64, False),   # Medium non-causal
        (2, 8, 512, 64, True),    # Medium causal
    ]

    # Validate all configurations
    print("\n" + "=" * 80)
    print("Validation Tests")
    print("=" * 80)

    all_passed = True
    for batch, heads, seq_len, head_dim, causal in test_configs:
        passed = validate_flash_attention(batch, heads, seq_len, head_dim, causal)
        all_passed = all_passed and passed

    if all_passed:
        print("\n✓ All validation tests PASSED!")
    else:
        print("\n✗ Some validation tests FAILED!")
        return 1

    # Benchmark
    print("\n" + "=" * 80)
    print("Performance Benchmark")
    print("=" * 80)

    benchmark_flash_attention(batch=4, heads=32, seq_len=1024, head_dim=64, causal=True)

    # Enable tracking and run a traced execution
    print("\n" + "=" * 80)
    print("Tracked Kernel Execution")
    print("=" * 80)

    tracker.enable()

    # Run with tracking enabled
    batch, heads, seq_len, head_dim = 2, 4, 256, 64
    print(f"\nRunning tracked execution: batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}")

    torch.manual_seed(42)
    q = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, head_dim, device=DEVICE, dtype=torch.float16)

    # Non-causal attention (tracked)
    print("\n→ Executing non-causal attention (tracked)...")
    output_non_causal = flash_attention_forward(q, k, v, causal=False)

    # Causal attention (tracked)
    print("→ Executing causal attention (tracked)...")
    output_causal = flash_attention_forward(q, k, v, causal=True)

    tracker.disable()

    # Save tracking data
    tracker.save_summary()

    # Print summary
    print("\n" + "=" * 80)
    print("✓ Tracking Complete!")
    print("=" * 80)
    print(f"\nGenerated files in: {output_dir}")
    print("  - binaries/       : Kernel binaries (CUBIN/PTX)")
    print("  - launchers/      : Standalone C++ harnesses")
    print("  - data/           : Serialized tensor arguments")
    print("  - tracking_summary.json")
    print("  - tracking_report.txt")
    print("\nTo build and run standalone harness:")
    print(f"  cd {output_dir}/launchers")
    print("  make -f <kernel>_launch<N>_Makefile")
    print("  ./<kernel>_launch<N>")

    return 0


if __name__ == '__main__':
    exit(main())