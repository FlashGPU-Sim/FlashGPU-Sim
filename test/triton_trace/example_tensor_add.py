#!/usr/bin/env python3
"""
Example: Track a TMA tensor descriptor kernel with autotune

This example demonstrates how to track kernels that use Triton's tensor
descriptor API (TMA operations on Hopper+ GPUs). Tensor descriptors are 
built inside the kernel, no host side preparation is needed.
"""

from pathlib import Path
import torch
import triton
import triton.language as tl

from track_triton_kernels import TritonKernelTracker

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def get_autotune_config():
    """Get autotune configurations for CUDA"""
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128},
            num_stages=3,
            num_warps=8,
            num_ctas=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128},
            num_stages=4,
            num_warps=4,
            num_ctas=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64},
            num_stages=4,
            num_warps=4,
            num_ctas=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64},
            num_stages=4,
            num_warps=4,
            num_ctas=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128},
            num_stages=5,
            num_warps=2,
            num_ctas=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32},
            num_stages=5,
            num_warps=2,
            num_ctas=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32},
            num_stages=5,
            num_warps=2,
            num_ctas=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128},
            num_stages=3,
            num_warps=8,
            num_ctas=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256},
            num_stages=3,
            num_warps=8,
            num_ctas=1,
        ),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N"],
)
@triton.jit
def kernel_add(
    A_ptr, B_ptr,
    M, N,
    STRIDE_AM, STRIDE_BM,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    """Tensor addition using TMA descriptor API"""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Create tensor descriptors
    desc_A = tl.make_tensor_descriptor(
        base=A_ptr,
        shape=[M, N],
        strides=[STRIDE_AM, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N]
    )

    desc_B = tl.make_tensor_descriptor(
        base=B_ptr,
        shape=[M, N],
        strides=[STRIDE_BM, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N]
    )

    # Calculate offsets
    offs_m = pid_m * BLOCK_SIZE_M
    offs_n = pid_n * BLOCK_SIZE_N
    offsets = [offs_m, offs_n]

    # Load, compute, store
    a = tl.load_tensor_descriptor(desc_A, offsets)
    output = a + 1.0
    tl.store_tensor_descriptor(desc_B, offsets, output)


def tensor_add(a, b, print_config=False, print_perf=False):
    """
    Wrapper function for tensor addition with autotune support.
    
    Args:
        a: Input tensor
        b: Output tensor
        print_config: Whether to print the best configuration
        print_perf: Whether to print performance metrics
    
    Returns:
        Output tensor b
    """
    # Check constraints
    assert a.shape == b.shape, "Input and output shapes must match"
    assert a.is_contiguous(), "Input tensor must be contiguous"
    assert b.is_contiguous(), "Output tensor must be contiguous"
    
    M, N = a.shape
    
    # Define grid as a lambda function that takes META parameters
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    
    # Benchmark the kernel execution if requested
    if print_perf:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: kernel_add[grid](
                a, b, M, N,
                a.stride(0), b.stride(0),
            ),
            quantiles=[0.5, 0.2, 0.8],
        )
    else:
        kernel_add[grid](
            a, b, M, N,
            a.stride(0), b.stride(0),
        )
    
    # Print performance metrics
    if print_perf:
        # Calculate operations: M*N additions
        ops = M * N
        gops = ops / (ms * 1e-3) / 1e9
        gops_min = ops / (max_ms * 1e-3) / 1e9
        gops_max = ops / (min_ms * 1e-3) / 1e9
        print(f"\n{'='*60}")
        print(f"Performance for M={M}, N={N}:")
        print(f"{'='*60}")
        print(f"  Execution time (median): {ms:.4f} ms")
        print(f"  Execution time (min): {min_ms:.4f} ms")
        print(f"  Execution time (max): {max_ms:.4f} ms")
        print(f"  GOPS (median): {gops:.2f}")
        print(f"  GOPS (min): {gops_min:.2f}")
        print(f"  GOPS (max): {gops_max:.2f}")
        print(f"{'='*60}\n")
    
    # Print the best configuration after autotuning
    if print_config:
        # Construct the key the same way the autotuner does
        key = [M, N]
        # Add dtype information
        for arg in [a, b]:
            if hasattr(arg, "dtype"):
                key.append(str(arg.dtype))
        key = tuple(key)
        
        # Try to get from cache first, otherwise use best_config
        best_config = None
        if hasattr(kernel_add, "cache") and key in kernel_add.cache:
            best_config = kernel_add.cache[key]
        elif hasattr(kernel_add, "best_config"):
            best_config = kernel_add.best_config
        
        if best_config is not None:
            print(f"\n{'='*60}")
            print(f"Best Configuration for M={M}, N={N}:")
            print(f"{'='*60}")
            print(f"  BLOCK_SIZE_M: {best_config.kwargs.get('BLOCK_SIZE_M')}")
            print(f"  BLOCK_SIZE_N: {best_config.kwargs.get('BLOCK_SIZE_N')}")
            print(f"  num_stages: {best_config.num_stages}")
            print(f"  num_warps: {best_config.num_warps}")
            print(f"{'='*60}\n")
        else:
            print(f"\n⚠️  No best configuration found (key={key})")
    
    return b


def main():
    print("=" * 80)
    print("Triton Kernel Tracker - TMA Tensor Descriptor Example with Autotune")
    print("=" * 80)
    
    # Set TMA allocator
    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    )
    
    # Initialize tracker
    output_dir = Path("./triton_kernel_tracking/example_tensor_add").resolve()
    tracker = TritonKernelTracker(output_dir, save_binaries=True, capture_args=True)
    tracker.disable()
    print(f"\nOutput directory: {output_dir}")
    
    # Prepare data
    M, N = 4096, 4096
    a = torch.randn((M, N), dtype=torch.float32, device="cuda")
    b = torch.empty((M, N), dtype=torch.float32, device="cuda")
    
    # Launch kernel with autotune
    print(f"\nLaunching kernel with M={M}, N={N}")
    print("Autotuning to find optimal block sizes...")
    tensor_add(a, b, print_config=True, print_perf=True)
    
    # Verify result
    expected = a + 1.0
    if torch.allclose(b, expected, atol=1e-2, rtol=1e-2):
        print("✅ Kernel output verified - PASSED")
    else:
        print("❌ Kernel output mismatch - FAILED")
        return 1
    
    # Save tracking data
    # Important: Clear output tensor before tracking to ensure it's detected as modified
    b.zero_()  # Clear b so tracker can detect it as an output
    tracker.enable()
    tensor_add(a, b)
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
    print("  make -f kernel_add_launch1_Makefile")
    print("  ./kernel_add_launch1")
    print("\nNote: TMA requires SM 90+ GPU (Hopper or newer)")
    
    return 0


if __name__ == '__main__':
    exit(main())
