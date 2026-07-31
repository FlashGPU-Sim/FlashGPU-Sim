#!/usr/bin/env python3
"""
Example: Track a simple vector addition kernel

This example demonstrates how to use TritonTrace to capture
kernel compilation and launch information.
"""

from pathlib import Path
import torch
import triton
import triton.language as tl

TRITON_TRACE_DIR = Path(__file__).resolve().parent.parent

import tritontrace


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Simple vector addition kernel"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def main():
    print("=" * 80)
    print("Triton Kernel Tracker - Vector Addition Example")
    print("=" * 80)
    
    # Initialize tracker
    output_dir = (TRITON_TRACE_DIR / "triton_kernel_tracking/example_vector_add").resolve()
    tracker = tritontrace.Tracker(output_dir, save_binaries=True, capture_args=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Prepare data
    size = 1024
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    output = torch.empty_like(x)
    
    # Define grid
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    print(f"\nLaunching kernel with size={size}, BLOCK_SIZE=256")
    add_kernel[grid](x, y, output, size, BLOCK_SIZE=256)
    
    # Verify result
    expected = x + y
    if torch.allclose(output, expected):
        print("Kernel output verified - PASSED")
    else:
        print("Kernel output mismatch - FAILED")
        return 1
    
    # Launch again to demonstrate multiple launches
    print("\nLaunching kernel again (second launch)")
    add_kernel[grid](x, y, output, size, BLOCK_SIZE=256)
    
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
    print("  make -f add_kernel_launch1_Makefile")
    print("  ./add_kernel_launch1")
    
    return 0


if __name__ == '__main__':
    exit(main())
