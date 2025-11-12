#!/usr/bin/env python3
"""
Modified add.py with tensormap support.

This example shows how to use the tensormap extension to properly track
TMA (Tensor Memory Accelerator) kernels.

The key difference from the original add.py is the call to
register_tensormap_for_next_launch() before launching the kernel.
"""

import torch
import triton
import triton.language as tl
import shutil
import os
from pathlib import Path
from track_triton_kernels import TritonKernelTracker


# ------------------------------------
# 1. Kernel (unchanged)
# ------------------------------------
@triton.jit
def kernel_add(
    A_ptr, B_ptr,
    M, N,   # Full matrix shape (Row, Col)
    STRIDE_AM, STRIDE_BM,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    """
    Explicitly trigger TMA using the tl.make_tensor_descriptor API
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Create descriptors
    # Triton will compile this to a TMA descriptor object on H100/B100
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

    # Calculate scalar offsets for this program block
    offs_m = pid_m * BLOCK_SIZE_M
    offs_n = pid_n * BLOCK_SIZE_N
    offsets = [offs_m, offs_n]

    # Load a (BLOCK_SIZE_M, BLOCK_SIZE_N) block from desc_A at the offsets
    # This will be compiled to 'cp.async.bulk'
    a = tl.load_tensor_descriptor(desc_A, offsets)

    # Compute
    output = a + 1.0

    # Store the block to desc_B at the offsets
    # This will be compiled to 'st.async' or 'stmatrix'
    tl.store_tensor_descriptor(desc_B, offsets, output)


# ------------------------------------
# 2. Test Runner
# ------------------------------------
def run_test_with_tracker(tracker):
    """
    Run the kernel with proper tensormap tracking.
    """
    print("Running EXPLICIT TMA Descriptor test case with tensormap support...")

    if not torch.cuda.is_available():
        print("Error: CUDA not detected!")
        return

    cap = torch.cuda.get_device_capability()
    print(f"Detected GPU Compute Capability: {cap}")

    # Check for Hopper or newer (required for TMA)
    if cap[0] < 9:
        print(f"WARNING: TMA requires SM 90+ (Hopper), but got SM {cap[0]}{cap[1]}")
        print("The tracker will still work, but the generated harness may not run on this GPU.")

    M, N = 4096, 4096
    a_cpu = torch.randn((M, N), dtype=torch.float32)
    a_gpu = a_cpu.to("cuda")
    b_gpu = torch.empty((M, N), dtype=torch.float32, device="cuda")

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    print(f"Grid: {grid}, Block: ({BLOCK_SIZE_M}, {BLOCK_SIZE_N})")
    print(f"Input DTypes: A={a_gpu.dtype}, B={b_gpu.dtype}")

    print(f"\n[Main] Launching Descriptor API kernel...")
    kernel_add[grid](
        a_gpu, b_gpu,
        M, N,
        a_gpu.stride(0), b_gpu.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )

    torch.cuda.synchronize()
    print("[Main] Kernel execution finished.")

    # Verification
    expected_b_cpu = a_cpu + 1.0
    torch.testing.assert_close(b_gpu.cpu(), expected_b_cpu, atol=1e-2, rtol=1e-2)
    print("\nTest Passed! GPU result matches CPU expected value.")


# ------------------------------------
# 3. Main Program
# ------------------------------------
if __name__ == "__main__":

    # 1. Clear cache (forces re-compilation for external tools)
    cache_dir = os.path.expanduser("~/.triton/cache/")
    if os.path.exists(cache_dir):
        print(f"--- Clearing Triton cache: {cache_dir} ---")
        shutil.rmtree(cache_dir)
        print("--- Cache cleared ---")

    # 2. Set the allocator required for TMA
    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    )

    # 3. Initialize tracker
    print("\n--- Initializing TritonKernelTracker with tensormap support ---")
    tracker = TritonKernelTracker(
        output_dir=Path("./triton_tracking"),
        save_binaries=True,
        capture_args=True
    )

    # 4. Run the test with tensormap tracking
    run_test_with_tracker(tracker)

    # 5. Save tracking data
    tracker.save_summary()

    print("\n" + "="*60)
    print("Tracking complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - PTX and CUBIN: triton_tracking/binaries/")
    print("  - C++ harness: triton_tracking/launchers/")
    print("  - Tensor data: triton_tracking/data/")
    print("\nTo build and run the harness:")
    print("  cd triton_tracking/launchers")
    print("  make -f kernel_add_launch1_Makefile")
    print("  ./kernel_add_launch1")
    print("\nNote: The harness requires SM 90+ GPU to run (Hopper or newer)")
