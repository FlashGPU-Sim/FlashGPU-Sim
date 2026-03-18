#!/usr/bin/env python3
"""
Example: Track a TMA GEMM kernel with device-side tensor descriptors and AUTOTUNE

This example demonstrates how to track a matrix multiplication kernel that uses
Triton's tensor descriptor API (TMA operations on Hopper+ GPUs). Tensor descriptors
are built inside the kernel, no host side preparation is needed.

C = A @ B where:
- A has shape (M, K)
- B has shape (K, N) but stored as (N, K) transposed for coalesced TMA access
- C has shape (M, N)
"""

from pathlib import Path
import torch
import triton
import triton.language as tl

from track_triton_kernels import TritonKernelTracker

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def get_autotune_config():
    """Get autotune configurations for TMA GEMM"""
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3,
            num_warps=8,
            num_ctas=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
            num_ctas=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
            num_ctas=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
            num_ctas=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4,
            num_warps=4,
            num_ctas=1,
        ),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def kernel_tma_gemm(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Matrix multiplication using TMA descriptor API (device-side descriptors).
    
    C = A @ B where:
    - A has shape (M, K), row-major
    - B has shape (N, K), transposed from (K, N) for coalesced access
    - C has shape (M, N), row-major
    """
    # Program ID processing with L2 cache optimization grouping
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create TMA tensor descriptors (device-side)
    desc_A = tl.make_tensor_descriptor(
        base=A_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K]
    )

    desc_B = tl.make_tensor_descriptor(
        base=B_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K]
    )

    desc_C = tl.make_tensor_descriptor(
        base=C_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N]
    )

    # Calculate block offsets
    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Main computation loop
    for k in range(k_tiles):
        offs_k = k * BLOCK_SIZE_K
        
        # Load tiles using TMA
        a = tl.load_tensor_descriptor(desc_A, [offs_am, offs_k])
        b = tl.load_tensor_descriptor(desc_B, [offs_bn, offs_k])
        
        # Accumulate: a is (BLOCK_SIZE_M, BLOCK_SIZE_K), b is (BLOCK_SIZE_N, BLOCK_SIZE_K)
        # We need to transpose b for the dot product
        accumulator = tl.dot(a, b.T, accumulator)

    # Convert to output dtype and store using TMA
    c = accumulator.to(tl.float16)
    tl.store_tensor_descriptor(desc_C, [offs_am, offs_bn], c)


def tma_gemm(a, b, print_config=False, print_perf=False):
    """
    Matrix multiplication using TMA with device-side tensor descriptors.
    
    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (N, K) - NOTE: transposed from standard (K, N)
        print_config: Whether to print the best configuration
        print_perf: Whether to print performance metrics
    
    Returns:
        Output matrix C of shape (M, N)
    """
    # Check constraints
    assert len(a.shape) == 2 and len(b.shape) == 2, "Inputs must be 2D tensors"
    assert a.shape[1] == b.shape[1], f"K dimension mismatch: A.shape[1]={a.shape[1]}, B.shape[1]={b.shape[1]}"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    
    M, K = a.shape
    N, K = b.shape
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    
    # 1D launch kernel with L2 cache optimization
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    
    # Benchmark if requested
    if print_perf:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: kernel_tma_gemm[grid](
                a, b, c, M, N, K,
            ),
            quantiles=[0.5, 0.2, 0.8],
        )
    else:
        kernel_tma_gemm[grid](
            a, b, c, M, N, K,
        )
    
    # Print performance metrics
    if print_perf:
        flops = 2.0 * M * N * K
        tflops = flops / (ms * 1e-3) / 1e12
        print(f"\n{'='*60}")
        print(f"Performance for M={M}, N={N}, K={K}:")
        print(f"{'='*60}")
        print(f"  Execution time (median): {ms:.4f} ms")
        print(f"  TFLOPS (median): {tflops:.2f}")
        print(f"{'='*60}\n")
    
    # Print the best configuration after autotuning
    if print_config:
        key = [M, N, K]
        for arg in [a, b, c]:
            if hasattr(arg, "dtype"):
                key.append(str(arg.dtype))
        key = tuple(key)
        
        best_config = None
        if hasattr(kernel_tma_gemm, "cache") and key in kernel_tma_gemm.cache:
            best_config = kernel_tma_gemm.cache[key]
        elif hasattr(kernel_tma_gemm, "best_config"):
            best_config = kernel_tma_gemm.best_config
        
        if best_config is not None:
            print(f"\n{'='*60}")
            print(f"Best Configuration for M={M}, N={N}, K={K}:")
            print(f"{'='*60}")
            print(f"  BLOCK_SIZE_M: {best_config.kwargs.get('BLOCK_SIZE_M')}")
            print(f"  BLOCK_SIZE_N: {best_config.kwargs.get('BLOCK_SIZE_N')}")
            print(f"  BLOCK_SIZE_K: {best_config.kwargs.get('BLOCK_SIZE_K')}")
            print(f"  GROUP_SIZE_M: {best_config.kwargs.get('GROUP_SIZE_M')}")
            print(f"  num_stages: {best_config.num_stages}")
            print(f"  num_warps: {best_config.num_warps}")
            print(f"{'='*60}\n")
        else:
            print(f"\n⚠️  No best configuration found (key={key})")
    
    return c


def main():
    print("=" * 80)
    print("Triton Kernel Tracker - TMA GEMM Example with AUTOTUNE")
    print("=" * 80)
    
    # Set TMA allocator for device-side descriptor creation
    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    )
    
    # Initialize tracker - clear output directory for fresh tracking
    output_dir = Path("./triton_kernel_tracking/example_tma_gemm").resolve()
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    tracker = TritonKernelTracker(output_dir, save_binaries=True, capture_args=True)
    tracker.disable()  # Disable during warmup/perf runs
    print(f"\nOutput directory: {output_dir}")
    
    # Prepare data
    M, N, K = 1024, 1024, 1024
    torch.manual_seed(0)
    
    # A is (M, K), B is transposed to (N, K) for coalesced TMA access
    a = torch.randn((M, K), dtype=torch.float16, device="cuda")
    b_original = torch.randn((K, N), dtype=torch.float16, device="cuda")
    b = b_original.T.contiguous()  # Transpose to (N, K)
    
    # Launch kernel (this also triggers autotune compilation)
    print(f"\nLaunching TMA GEMM kernel with M={M}, N={N}, K={K}")
    print("Autotuning to find optimal block sizes...")
    c = tma_gemm(a, b, print_config=True, print_perf=True)
    
    # Verify result against torch.matmul
    expected = torch.matmul(a.float(), b_original.float()).half()
    
    if torch.allclose(c, expected, atol=1e-1, rtol=1e-1):
        print("✅ Kernel output verified - PASSED")
    else:
        max_diff = (c - expected).abs().max().item()
        print(f"❌ Kernel output mismatch - FAILED (max diff: {max_diff})")
        return 1
    
    # Track kernel execution with fresh output tensor
    c_track = torch.empty((M, N), device="cuda", dtype=torch.float16)
    tracker.enable()
    tma_gemm(a, b)  # Tracked launch - should use correct kernel hash!
    tracker.save_summary()
    
    print("\n" + "=" * 80)
    print("Tracking Complete!")
    print("=" * 80)
    print(f"\nGenerated files in: {output_dir}")
    print("  - binaries/       : Kernel binaries (CUBIN/PTX)")
    print("  - launchers/      : Standalone C++ harnesses")
    print("  - data/           : Serialized tensor arguments")
    print("  - tracking_summary.json")
    print("  - tracking_report.txt")
    print("\nTo build and run standalone harness:")
    print(f"  cd {output_dir}/launchers")
    print("  make -f kernel_tma_gemm_launch1_Makefile")
    print("  ./kernel_tma_gemm_launch1")
    print("\nNote: TMA requires SM 90+ GPU (Hopper or newer)")
    
    return 0


if __name__ == '__main__':
    exit(main())
