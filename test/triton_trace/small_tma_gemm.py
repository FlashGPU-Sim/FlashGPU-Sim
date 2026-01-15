#!/usr/bin/env python3
"""
Small TMA GEMM test case for debugging.

This is a simplified version of example_tma_gemm.py with:
- Smaller matrix sizes (64x64x64)
- Deterministic, easy-to-trace input data
- Fixed autotuning config (no autotune to simplify)

C = A @ B where:
- A has shape (M, K) = (64, 64)
- B has shape (N, K) = (64, 64), transposed from (K, N) for coalesced TMA access
- C has shape (M, N) = (64, 64)

Input data patterns:
- Matrix A: Row index pattern. A[i][j] = i (row number as float16)
  This makes it easy to trace which row is being accessed.
  
- Matrix B (before transpose): Column index pattern. B[i][j] = j (column number as float16)
  After transpose to (N, K): B_t[j][i] = j
  
Expected result:
- C[i][j] = sum_{k=0}^{K-1} A[i][k] * B[k][j]
         = sum_{k=0}^{K-1} i * j
         = i * j * K
  For K=64: C[i][j] = i * j * 64

How to observe/trace data during PTX execution:
1. Matrix A values are row indices (0, 1, 2, ..., 63)
   - When you see ldmatrix loading from A, values will be the row number
   - All elements in the same row have the same value
   
2. Matrix B values (after transpose) are original column indices
   - When you see ldmatrix loading from B, values will be column indices
   
3. After mma/dot operations, partial results will show recognizable patterns:
   - Intermediate: i * j * (number of K iterations completed)
   
4. Final results: C[i][j] = i * j * 64
   - C[0][*] = 0 (first row all zeros)
   - C[*][0] = 0 (first column all zeros)
   - C[1][1] = 64, C[1][2] = 128, C[2][2] = 256, etc.
"""

from pathlib import Path
import torch
import triton
import triton.language as tl

from track_triton_kernels import TritonKernelTracker

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# Fixed configuration - no autotuning for simpler debugging
# Using smallest reasonable block sizes for easier tracing
@triton.jit
def kernel_tma_gemm_small(
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


def tma_gemm_small(a, b, block_m=64, block_n=64, block_k=64):
    """
    Matrix multiplication using TMA with device-side tensor descriptors.
    Fixed block sizes for predictable debugging.
    
    Args:
        a: Input matrix A of shape (M, K)
        b: Input matrix B of shape (N, K) - NOTE: transposed from standard (K, N)
        block_m, block_n, block_k: Block sizes (default 64 for small test)
    
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
    
    # 1D launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    
    kernel_tma_gemm_small[grid](
        a, b, c, M, N, K,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=8,
    )
    
    return c


def create_debug_matrices(M, N, K, device="cuda"):
    """
    Create matrices with easy-to-trace patterns.
    
    Matrix A (M x K): Each element A[i][j] = i (row index)
    Matrix B_original (K x N): Each element B[i][j] = j (column index)
    Matrix B (N x K): Transposed version of B_original
    
    Expected C = A @ B_original:
    C[i][j] = sum_{k=0}^{K-1} A[i][k] * B_original[k][j]
            = sum_{k=0}^{K-1} i * j
            = i * j * K
    """
    # Matrix A: row index pattern
    # A[i][j] = i for all j
    a = torch.zeros((M, K), dtype=torch.float16, device=device)
    for i in range(M):
        a[i, :] = float(i)
    
    # Matrix B_original: column index pattern  
    # B_original[i][j] = j for all i
    b_original = torch.zeros((K, N), dtype=torch.float16, device=device)
    for j in range(N):
        b_original[:, j] = float(j)
    
    # Transpose B for TMA coalesced access: (N, K)
    b = b_original.T.contiguous()
    
    return a, b, b_original


def print_matrix_sample(mat, name, rows=8, cols=8):
    """Print a sample of matrix values for verification."""
    print(f"\n{name} (showing {rows}x{cols} corner):")
    print("-" * 50)
    for i in range(min(rows, mat.shape[0])):
        row_vals = [f"{mat[i, j].item():8.1f}" for j in range(min(cols, mat.shape[1]))]
        print(f"  Row {i:2d}: " + " ".join(row_vals))


def main():
    print("=" * 80)
    print("Small TMA GEMM Test Case for Debugging")
    print("=" * 80)
    
    # Set TMA allocator for device-side descriptor creation
    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    )
    
    # Initialize tracker
    output_dir = Path("./triton_kernel_tracking/small_tma_gemm").resolve()
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    tracker = TritonKernelTracker(output_dir, save_binaries=True, capture_args=True)
    tracker.disable()
    print(f"\nOutput directory: {output_dir}")
    
    # Small matrix sizes - fits in one block for simplest case
    M, N, K = 64, 64, 64
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    
    print(f"\nMatrix dimensions: M={M}, N={N}, K={K}")
    print(f"Block sizes: BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")
    print(f"Number of blocks: {M // BLOCK_M} x {N // BLOCK_N} = 1 (single block!)")
    print(f"K iterations: {K // BLOCK_K} = 1 (single iteration!)")
    
    # Create debug matrices
    print("\n" + "=" * 80)
    print("Creating Debug Matrices")
    print("=" * 80)
    a, b, b_original = create_debug_matrices(M, N, K)
    
    print("\nInput data patterns:")
    print("  - Matrix A[i][j] = i (row index)")
    print("  - Matrix B_original[i][j] = j (column index)")
    print("  - Matrix B = B_original.T (for TMA)")
    
    print_matrix_sample(a, "Matrix A (row indices)")
    print_matrix_sample(b_original, "Matrix B_original (column indices)")
    print_matrix_sample(b, "Matrix B transposed (N x K)")
    
    # Expected result
    print("\n" + "=" * 80)
    print("Expected Result: C[i][j] = i * j * K = i * j * 64")
    print("=" * 80)
    expected = torch.matmul(a.float(), b_original.float()).half()
    print_matrix_sample(expected, "Expected C")
    
    # Run kernel
    print("\n" + "=" * 80)
    print("Running TMA GEMM Kernel")
    print("=" * 80)
    c = tma_gemm_small(a, b, BLOCK_M, BLOCK_N, BLOCK_K)
    
    # Verify result
    print_matrix_sample(c, "Actual C (kernel output)")
    
    if torch.allclose(c, expected, atol=1e-1, rtol=1e-1):
        print("\n✅ Kernel output verified - PASSED")
    else:
        max_diff = (c - expected).abs().max().item()
        print(f"\n❌ Kernel output mismatch - FAILED (max diff: {max_diff})")
        
        # Show where differences occur
        diff = (c - expected).abs()
        max_idx = torch.argmax(diff)
        i, j = max_idx // N, max_idx % N
        print(f"   Max diff at [{i}, {j}]: expected={expected[i,j].item()}, got={c[i,j].item()}")
    
    # Track kernel execution
    print("\n" + "=" * 80)
    print("Tracking Kernel Execution")
    print("=" * 80)
    tracker.enable()
    c_track = tma_gemm_small(a, b, BLOCK_M, BLOCK_N, BLOCK_K)
    tracker.save_summary()
    
    # Print debugging guide
    print("\n" + "=" * 80)
    print("HOW TO DEBUG / TRACE DATA IN PTX EXECUTION")
    print("=" * 80)
    print("""
Data Patterns to Look For:
--------------------------
1. When ldmatrix loads from Matrix A:
   - Values will be: 0, 1, 2, 3, ... (row indices as fp16)
   - All 64 elements in row 0 = 0.0
   - All 64 elements in row 1 = 1.0
   - All 64 elements in row 5 = 5.0
   - In hex fp16: 0=0x0000, 1=0x3C00, 2=0x4000, 3=0x4200...

2. When ldmatrix loads from Matrix B (transposed):
   - Values will be: 0, 1, 2, 3, ... (original column indices)
   - B[row j][k] = j for all k
   - Row 0 of B_transposed = all 0s
   - Row 1 of B_transposed = all 1s
   
3. After MMA/dot operations:
   - Intermediate accumulator values follow: i * j * (k_iters * BLOCK_K)
   - Since K=64 and BLOCK_K=64, only 1 iteration
   - Final values before store: i * j * 64
   
4. When stmatrix stores to Matrix C:
   - C[0][*] = 0 (row 0 is all zeros)
   - C[*][0] = 0 (column 0 is all zeros)
   - C[1][1] = 64.0 (0x5400 in fp16)
   - C[2][3] = 2*3*64 = 384.0 (0x5E00 in fp16)
   - C[7][7] = 7*7*64 = 3136.0

Useful grep patterns for log analysis:
--------------------------------------
  grep "ldmatrix" your_log.txt          # Find all ldmatrix operations
  grep "stmatrix" your_log.txt          # Find all stmatrix operations
  grep "mma\\|wgmma" your_log.txt       # Find MMA operations
  grep "cvt.*f16" your_log.txt          # Find f32->f16 conversions (before store)
  
FP16 Reference Values:
----------------------
  0.0  = 0x0000     16.0 = 0x4C00     128.0 = 0x5800
  1.0  = 0x3C00     32.0 = 0x5000     256.0 = 0x5C00
  2.0  = 0x4000     64.0 = 0x5400     512.0 = 0x6000
  4.0  = 0x4400     
  8.0  = 0x4800
""")
    
    print(f"\nGenerated files in: {output_dir}")
    print("  - binaries/       : Kernel binaries (CUBIN/PTX)")
    print("  - launchers/      : Standalone C++ harnesses")
    print("  - data/           : Serialized tensor arguments")
    
    return 0


if __name__ == '__main__':
    exit(main())
