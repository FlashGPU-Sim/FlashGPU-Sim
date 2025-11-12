#!/usr/bin/env python3
"""
Example: Track a TMA tensor descriptor kernel

This example demonstrates how to track kernels that use Triton's tensor
descriptor API (TMA operations on Hopper+ GPUs). Tensor descriptors are 
built inside the kernel, no host side preparation is needed.
"""

from pathlib import Path
import torch
import triton
import triton.language as tl

from track_triton_kernels import TritonKernelTracker


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


def main():
    print("=" * 80)
    print("Triton Kernel Tracker - TMA Tensor Descriptor Example")
    print("=" * 80)
    
    # Set TMA allocator
    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    )
    
    # Initialize tracker
    output_dir = Path("./triton_kernel_tracking/example_tensor_add").resolve()
    tracker = TritonKernelTracker(output_dir, save_binaries=True, capture_args=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Prepare data
    M, N = 4096, 4096
    a = torch.randn((M, N), dtype=torch.float32, device="cuda")
    b = torch.empty((M, N), dtype=torch.float32, device="cuda")
    
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    
    # Launch kernel
    print(f"\nLaunching kernel with M={M}, N={N}, grid={grid}")
    kernel_add[grid](
        a, b, M, N,
        a.stride(0), b.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    # Verify result
    expected = a + 1.0
    if torch.allclose(b, expected, atol=1e-2, rtol=1e-2):
        print("Kernel output verified - PASSED")
    else:
        print("Kernel output mismatch - FAILED")
        return 1
    
    # Save tracking data
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
