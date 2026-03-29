#!/usr/bin/env python3
"""
Parameterized TMA GEMM trace generator for shape sweep testing.

Usage:
    python3 test_tma_gemm.py --size 1024
    python3 test_tma_gemm.py --m 64 --n 128 --k 256

Generates trace data in triton_kernel_tracking/test_tma_gemm/size_<N>/
or triton_kernel_tracking/test_tma_gemm/m<M>_n<N>_k<K>/
"""

import argparse
import shutil
from pathlib import Path

import torch
import triton
import triton.language as tl

from track_triton_kernels import TritonKernelTracker

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def get_autotune_config():
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=3, num_warps=8, num_ctas=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4, num_ctas=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4, num_ctas=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4, num_ctas=1,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
            num_stages=4, num_warps=4, num_ctas=1,
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
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    desc_A = tl.make_tensor_descriptor(
        base=A_ptr, shape=[M, K], strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    desc_B = tl.make_tensor_descriptor(
        base=B_ptr, shape=[N, K], strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    desc_C = tl.make_tensor_descriptor(
        base=C_ptr, shape=[M, N], strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(k_tiles):
        offs_k = k * BLOCK_SIZE_K
        a = tl.load_tensor_descriptor(desc_A, [offs_am, offs_k])
        b = tl.load_tensor_descriptor(desc_B, [offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(tl.float16)
    tl.store_tensor_descriptor(desc_C, [offs_am, offs_bn], c)


def tma_gemm(a, b):
    assert len(a.shape) == 2 and len(b.shape) == 2
    assert a.shape[1] == b.shape[1]
    assert a.is_contiguous() and b.is_contiguous()

    M, K = a.shape
    N, K = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    kernel_tma_gemm[grid](a, b, c, M, N, K)
    return c


def main():
    parser = argparse.ArgumentParser(description="Generate TMA GEMM trace for a given size")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--size", type=int, help="Square GEMM dimension (M=N=K=size)")
    group.add_argument("--m", type=int, help="M dimension (requires --n and --k)")
    parser.add_argument("--n", type=int, help="N dimension")
    parser.add_argument("--k", type=int, help="K dimension")
    args = parser.parse_args()

    if args.size is not None:
        M = N = K = args.size
        label = f"size={args.size} (M=N=K={args.size})"
        subdir = f"size_{args.size}"
    else:
        if args.n is None or args.k is None:
            parser.error("--m requires --n and --k")
        M, N, K = args.m, args.n, args.k
        label = f"M={M}, N={N}, K={K}"
        subdir = f"m{M}_n{N}_k{K}"

    print(f"{'='*80}")
    print(f"TMA GEMM Trace Generator — {label}")
    print(f"{'='*80}")

    triton.set_allocator(
        lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8)
    )

    output_dir = (Path(__file__).parent / f"triton_kernel_tracking/test_tma_gemm/{subdir}").resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)

    tracker = TritonKernelTracker(output_dir, save_binaries=True, capture_args=True)
    tracker.disable()
    print(f"\nOutput directory: {output_dir}")

    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.float16, device="cuda")
    b_original = torch.randn((K, N), dtype=torch.float16, device="cuda")
    b = b_original.T.contiguous()

    print(f"\nLaunching TMA GEMM kernel with M={M}, N={N}, K={K}")
    print("Autotuning...")
    c = tma_gemm(a, b)

    expected = torch.matmul(a.float(), b_original.float()).half()
    if torch.allclose(c, expected, atol=1e-1, rtol=1e-1):
        print("✅ Kernel output verified — PASSED")
    else:
        max_diff = (c - expected).abs().max().item()
        print(f"❌ Kernel output mismatch — FAILED (max diff: {max_diff})")
        return 1

    tracker.enable()
    tma_gemm(a, b)
    tracker.save_summary()

    print(f"\nTrace saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
