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
def kernel_add_2d(
    A_ptr, B_ptr,
    M, N,
    STRIDE_AM, STRIDE_BM,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    """2D Tensor addition using TMA descriptor API"""
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Create tensor descriptors (device-side)
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


@triton.jit
def kernel_add_1d(
    A_ptr, B_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    """1D Tensor addition using TMA descriptor API"""
    pid = tl.program_id(axis=0)

    # Create tensor descriptors (device-side)
    desc_A = tl.make_tensor_descriptor(
        base=A_ptr,
        shape=[N],
        strides=[1],
        block_shape=[BLOCK_SIZE]
    )

    desc_B = tl.make_tensor_descriptor(
        base=B_ptr,
        shape=[N],
        strides=[1],
        block_shape=[BLOCK_SIZE]
    )

    # Calculate offset
    offs = pid * BLOCK_SIZE
    offsets = [offs]

    # Load, compute, store
    a = tl.load_tensor_descriptor(desc_A, offsets)
    output = a + 1.0
    tl.store_tensor_descriptor(desc_B, offsets, output)


@triton.jit
def kernel_add_3d(
    A_ptr, B_ptr,
    D0, D1, D2,
    BLOCK_D0: tl.constexpr,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr
):
    """3D Tensor addition using TMA descriptor API"""
    pid_0 = tl.program_id(axis=0)
    pid_1 = tl.program_id(axis=1)
    pid_2 = tl.program_id(axis=2)

    # Create tensor descriptors (device-side)
    desc_A = tl.make_tensor_descriptor(
        base=A_ptr,
        shape=[D0, D1, D2],
        strides=[D1 * D2, D2, 1],
        block_shape=[BLOCK_D0, BLOCK_D1, BLOCK_D2]
    )

    desc_B = tl.make_tensor_descriptor(
        base=B_ptr,
        shape=[D0, D1, D2],
        strides=[D1 * D2, D2, 1],
        block_shape=[BLOCK_D0, BLOCK_D1, BLOCK_D2]
    )

    # Calculate offsets
    offs_0 = pid_0 * BLOCK_D0
    offs_1 = pid_1 * BLOCK_D1
    offs_2 = pid_2 * BLOCK_D2
    offsets = [offs_0, offs_1, offs_2]

    # Load, compute, store
    a = tl.load_tensor_descriptor(desc_A, offsets)
    output = a + 1.0
    tl.store_tensor_descriptor(desc_B, offsets, output)


@triton.jit
def kernel_add_4d(
    A_ptr, B_ptr,
    D0, D1, D2, D3,
    GRID_D1: tl.constexpr,
    BLOCK_D0: tl.constexpr,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    BLOCK_D3: tl.constexpr
):
    """4D Tensor addition using TMA descriptor API"""
    pid_0 = tl.program_id(axis=0)
    pid_1 = tl.program_id(axis=1)
    pid_2 = tl.program_id(axis=2)

    # Create tensor descriptors (device-side)
    desc_A = tl.make_tensor_descriptor(
        base=A_ptr,
        shape=[D0, D1, D2, D3],
        strides=[D1 * D2 * D3, D2 * D3, D3, 1],
        block_shape=[BLOCK_D0, BLOCK_D1, BLOCK_D2, BLOCK_D3]
    )

    desc_B = tl.make_tensor_descriptor(
        base=B_ptr,
        shape=[D0, D1, D2, D3],
        strides=[D1 * D2 * D3, D2 * D3, D3, 1],
        block_shape=[BLOCK_D0, BLOCK_D1, BLOCK_D2, BLOCK_D3]
    )

    # Calculate offsets (4D uses 3 program IDs max, linearize first two dims)
    pid_01 = pid_0
    offs_0 = (pid_01 // GRID_D1) * BLOCK_D0
    offs_1 = (pid_01 % GRID_D1) * BLOCK_D1
    offs_2 = pid_1 * BLOCK_D2
    offs_3 = pid_2 * BLOCK_D3
    offsets = [offs_0, offs_1, offs_2, offs_3]

    # Load, compute, store
    a = tl.load_tensor_descriptor(desc_A, offsets)
    output = a + 1.0
    tl.store_tensor_descriptor(desc_B, offsets, output)


@triton.jit
def kernel_add_5d(
    A_ptr, B_ptr,
    D0, D1, D2, D3, D4,
    GRID_D1: tl.constexpr,
    GRID_D2: tl.constexpr,
    BLOCK_D0: tl.constexpr,
    BLOCK_D1: tl.constexpr,
    BLOCK_D2: tl.constexpr,
    BLOCK_D3: tl.constexpr,
    BLOCK_D4: tl.constexpr
):
    """5D Tensor addition using TMA descriptor API"""
    pid_0 = tl.program_id(axis=0)
    pid_1 = tl.program_id(axis=1)
    pid_2 = tl.program_id(axis=2)

    # Create tensor descriptors (device-side)
    desc_A = tl.make_tensor_descriptor(
        base=A_ptr,
        shape=[D0, D1, D2, D3, D4],
        strides=[D1 * D2 * D3 * D4, D2 * D3 * D4, D3 * D4, D4, 1],
        block_shape=[BLOCK_D0, BLOCK_D1, BLOCK_D2, BLOCK_D3, BLOCK_D4]
    )

    desc_B = tl.make_tensor_descriptor(
        base=B_ptr,
        shape=[D0, D1, D2, D3, D4],
        strides=[D1 * D2 * D3 * D4, D2 * D3 * D4, D3 * D4, D4, 1],
        block_shape=[BLOCK_D0, BLOCK_D1, BLOCK_D2, BLOCK_D3, BLOCK_D4]
    )

    # Calculate offsets (5D uses 3 program IDs max, linearize first three dims)
    pid_012 = pid_0
    offs_0 = (pid_012 // (GRID_D1 * GRID_D2)) * BLOCK_D0
    offs_1 = ((pid_012 // GRID_D2) % GRID_D1) * BLOCK_D1
    offs_2 = (pid_012 % GRID_D2) * BLOCK_D2
    offs_3 = pid_1 * BLOCK_D3
    offs_4 = pid_2 * BLOCK_D4
    offsets = [offs_0, offs_1, offs_2, offs_3, offs_4]

    # Load, compute, store
    a = tl.load_tensor_descriptor(desc_A, offsets)
    output = a + 1.0
    tl.store_tensor_descriptor(desc_B, offsets, output)


# Legacy alias for backward compatibility
kernel_add = kernel_add_2d


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


def test_1d_tensor(tracker, shapes_and_blocks):
    """Test 1D TMA operations with corner cases"""
    print("\n" + "=" * 80)
    print("Testing 1D TMA Operations")
    print("=" * 80)

    for N, BLOCK_SIZE in shapes_and_blocks:
        print(f"\n1D Test: N={N}, BLOCK_SIZE={BLOCK_SIZE}")
        a = torch.randn(N, dtype=torch.float32, device="cuda")
        b = torch.empty(N, dtype=torch.float32, device="cuda")

        grid = (triton.cdiv(N, BLOCK_SIZE),)
        kernel_add_1d[grid](a, b, N, BLOCK_SIZE=BLOCK_SIZE)

        expected = a + 1.0
        if torch.allclose(b, expected, atol=1e-2, rtol=1e-2):
            print(f"✅ 1D kernel (N={N}) - PASSED")
        else:
            print(f"❌ 1D kernel (N={N}) - FAILED")
            return False

    # Track one configuration
    N, BLOCK_SIZE = shapes_and_blocks[0]
    a = torch.randn(N, dtype=torch.float32, device="cuda")
    b = torch.empty(N, dtype=torch.float32, device="cuda")
    b.zero_()
    tracker.enable()
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    kernel_add_1d[grid](a, b, N, BLOCK_SIZE=BLOCK_SIZE)
    tracker.disable()

    return True


def test_3d_tensor(tracker, shapes_and_blocks):
    """Test 3D TMA operations"""
    print("\n" + "=" * 80)
    print("Testing 3D TMA Operations")
    print("=" * 80)

    for (D0, D1, D2), (B0, B1, B2) in shapes_and_blocks:
        print(f"\n3D Test: shape=({D0}, {D1}, {D2}), blocks=({B0}, {B1}, {B2})")
        a = torch.randn(D0, D1, D2, dtype=torch.float32, device="cuda")
        b = torch.empty(D0, D1, D2, dtype=torch.float32, device="cuda")

        grid = (triton.cdiv(D0, B0), triton.cdiv(D1, B1), triton.cdiv(D2, B2))
        kernel_add_3d[grid](a, b, D0, D1, D2, BLOCK_D0=B0, BLOCK_D1=B1, BLOCK_D2=B2)

        expected = a + 1.0
        if torch.allclose(b, expected, atol=1e-2, rtol=1e-2):
            print(f"✅ 3D kernel ({D0}x{D1}x{D2}) - PASSED")
        else:
            print(f"❌ 3D kernel ({D0}x{D1}x{D2}) - FAILED")
            return False

    # Track one configuration
    (D0, D1, D2), (B0, B1, B2) = shapes_and_blocks[0]
    a = torch.randn(D0, D1, D2, dtype=torch.float32, device="cuda")
    b = torch.empty(D0, D1, D2, dtype=torch.float32, device="cuda")
    b.zero_()
    tracker.enable()
    grid = (triton.cdiv(D0, B0), triton.cdiv(D1, B1), triton.cdiv(D2, B2))
    kernel_add_3d[grid](a, b, D0, D1, D2, BLOCK_D0=B0, BLOCK_D1=B1, BLOCK_D2=B2)
    tracker.disable()

    return True


def test_4d_tensor(tracker, shapes_and_blocks):
    """Test 4D TMA operations with degenerate cases (some dims=1)"""
    print("\n" + "=" * 80)
    print("Testing 4D TMA Operations")
    print("=" * 80)

    for (D0, D1, D2, D3), (B0, B1, B2, B3) in shapes_and_blocks:
        print(f"\n4D Test: shape=({D0}, {D1}, {D2}, {D3}), blocks=({B0}, {B1}, {B2}, {B3})")
        a = torch.randn(D0, D1, D2, D3, dtype=torch.float32, device="cuda")
        b = torch.empty(D0, D1, D2, D3, dtype=torch.float32, device="cuda")

        grid_d1 = triton.cdiv(D1, B1)
        grid_01 = triton.cdiv(D0, B0) * grid_d1
        grid = (grid_01, triton.cdiv(D2, B2), triton.cdiv(D3, B3))
        kernel_add_4d[grid](a, b, D0, D1, D2, D3, GRID_D1=grid_d1, BLOCK_D0=B0, BLOCK_D1=B1, BLOCK_D2=B2, BLOCK_D3=B3)

        expected = a + 1.0
        if torch.allclose(b, expected, atol=1e-2, rtol=1e-2):
            print(f"✅ 4D kernel ({D0}x{D1}x{D2}x{D3}) - PASSED")
        else:
            print(f"❌ 4D kernel ({D0}x{D1}x{D2}x{D3}) - FAILED")
            return False

    # Track one configuration
    (D0, D1, D2, D3), (B0, B1, B2, B3) = shapes_and_blocks[0]
    a = torch.randn(D0, D1, D2, D3, dtype=torch.float32, device="cuda")
    b = torch.empty(D0, D1, D2, D3, dtype=torch.float32, device="cuda")
    b.zero_()
    tracker.enable()
    grid_d1 = triton.cdiv(D1, B1)
    grid_01 = triton.cdiv(D0, B0) * grid_d1
    grid = (grid_01, triton.cdiv(D2, B2), triton.cdiv(D3, B3))
    kernel_add_4d[grid](a, b, D0, D1, D2, D3, GRID_D1=grid_d1, BLOCK_D0=B0, BLOCK_D1=B1, BLOCK_D2=B2, BLOCK_D3=B3)
    tracker.disable()

    return True


def test_5d_tensor(tracker, shapes_and_blocks):
    """Test 5D TMA operations with non-uniform sizes"""
    print("\n" + "=" * 80)
    print("Testing 5D TMA Operations")
    print("=" * 80)

    for (D0, D1, D2, D3, D4), (B0, B1, B2, B3, B4) in shapes_and_blocks:
        print(f"\n5D Test: shape=({D0}, {D1}, {D2}, {D3}, {D4}), blocks=({B0}, {B1}, {B2}, {B3}, {B4})")
        a = torch.randn(D0, D1, D2, D3, D4, dtype=torch.float32, device="cuda")
        b = torch.empty(D0, D1, D2, D3, D4, dtype=torch.float32, device="cuda")

        grid_d1 = triton.cdiv(D1, B1)
        grid_d2 = triton.cdiv(D2, B2)
        grid_012 = triton.cdiv(D0, B0) * grid_d1 * grid_d2
        grid = (grid_012, triton.cdiv(D3, B3), triton.cdiv(D4, B4))
        kernel_add_5d[grid](a, b, D0, D1, D2, D3, D4, GRID_D1=grid_d1, GRID_D2=grid_d2, BLOCK_D0=B0, BLOCK_D1=B1, BLOCK_D2=B2, BLOCK_D3=B3, BLOCK_D4=B4)

        expected = a + 1.0
        if torch.allclose(b, expected, atol=1e-2, rtol=1e-2):
            print(f"✅ 5D kernel ({D0}x{D1}x{D2}x{D3}x{D4}) - PASSED")
        else:
            print(f"❌ 5D kernel ({D0}x{D1}x{D2}x{D3}x{D4}) - FAILED")
            return False

    # Track one configuration
    (D0, D1, D2, D3, D4), (B0, B1, B2, B3, B4) = shapes_and_blocks[0]
    a = torch.randn(D0, D1, D2, D3, D4, dtype=torch.float32, device="cuda")
    b = torch.empty(D0, D1, D2, D3, D4, dtype=torch.float32, device="cuda")
    b.zero_()
    tracker.enable()
    grid_d1 = triton.cdiv(D1, B1)
    grid_d2 = triton.cdiv(D2, B2)
    grid_012 = triton.cdiv(D0, B0) * grid_d1 * grid_d2
    grid = (grid_012, triton.cdiv(D3, B3), triton.cdiv(D4, B4))
    kernel_add_5d[grid](a, b, D0, D1, D2, D3, D4, GRID_D1=grid_d1, GRID_D2=grid_d2, BLOCK_D0=B0, BLOCK_D1=B1, BLOCK_D2=B2, BLOCK_D3=B3, BLOCK_D4=B4)
    tracker.disable()

    return True


def main():
    print("=" * 80)
    print("Triton Kernel Tracker - TMA Multi-Dimensional Testing (1D, 2D, 3D-5D)")
    print("=" * 80)

    # Set TMA allocator
    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    )

    # Initialize tracker
    output_dir = (Path(__file__).parent / "triton_kernel_tracking/example_tensor_add").resolve()
    tracker = TritonKernelTracker(output_dir, save_binaries=True, capture_args=True)
    tracker.disable()
    print(f"\nOutput directory: {output_dir}")

    # Test 2D (existing, for regression)
    print("\n" + "=" * 80)
    print("Testing 2D TMA Operations (Regression)")
    print("=" * 80)
    M, N = 4096, 4096
    a = torch.randn((M, N), dtype=torch.float32, device="cuda")
    b = torch.empty((M, N), dtype=torch.float32, device="cuda")
    print(f"\n2D Test: M={M}, N={N}")
    print("Autotuning to find optimal block sizes...")
    tensor_add(a, b, print_config=True, print_perf=True)
    expected = a + 1.0
    if torch.allclose(b, expected, atol=1e-2, rtol=1e-2):
        print("✅ 2D kernel (regression) - PASSED")
    else:
        print("❌ 2D kernel (regression) - FAILED")
        return 1

    # Track 2D
    b.zero_()
    tracker.enable()
    tensor_add(a, b)
    tracker.disable()

    # Test 1D with corner case: N not divisible by BLOCK_SIZE (remainder handling)
    test_1d_shapes = [
        (8192, 128),     # Regular case
        (8195, 128),     # Remainder case: 8195 = 64*128 + 3
    ]
    if not test_1d_tensor(tracker, test_1d_shapes):
        return 1

    # Test 3D with mixed sizes
    test_3d_shapes = [
        ((64, 64, 64), (16, 16, 16)),        # Regular case
        ((100, 50, 80), (16, 16, 16)),       # Mixed sizes
    ]
    if not test_3d_tensor(tracker, test_3d_shapes):
        return 1

    # Test 4D with minimal dims (degenerate cases)
    test_4d_shapes = [
        ((32, 32, 32, 32), (8, 8, 8, 8)),    # Regular case
        ((1, 64, 64, 64), (1, 16, 16, 16)),  # Degenerate: D0=1
    ]
    if not test_4d_tensor(tracker, test_4d_shapes):
        return 1

    # Test 5D with small, non-uniform sizes
    test_5d_shapes = [
        ((16, 16, 16, 16, 16), (4, 4, 4, 4, 4)),    # Regular case
        ((8, 12, 10, 14, 16), (4, 4, 4, 4, 4)),     # Non-uniform sizes
    ]
    if not test_5d_tensor(tracker, test_5d_shapes):
        return 1

    # Save tracking summary
    tracker.save_summary()

    print("\n" + "=" * 80)
    print("All Tests PASSED - Tracking Complete!")
    print("=" * 80)
    print(f"\nGenerated files in: {output_dir}")
    print("  - binaries/       : Kernel binaries (CUBIN/PTX)")
    print("  - launchers/      : Standalone C++ harnesses")
    print("  - data/           : Serialized tensor arguments")
    print("  - tracking_summary.json")
    print("  - tracking_report.txt")
    print("\nNext steps:")
    print("1. Inspect PTX for stubbed instructions (allowed):")
    print(f"   cd {output_dir}/launchers")
    print("   grep -E 'cp\\.async\\.bulk\\.(commit_group|wait_group)|tensormap\\.cp_fence' *.ptx")
    print("2. Build and run standalone harnesses:")
    print("   make")
    print("   ./kernel_add_*_launch*")
    print("\nNote: TMA requires SM 90+ GPU (Hopper or newer)")

    return 0


if __name__ == '__main__':
    exit(main())
