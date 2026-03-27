"""
Adopted from Triton tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py
"""

import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=5,
            num_warps=2,
        ),
        # Good config for fp8 inputs.
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
    ]


def get_hip_autotune_config():
    sizes = [
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 6},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4},
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 6},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 6},
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 4,
        },
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 4,
        },
        {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 4,
        },
        {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 6,
        },
    ]
    return [
        triton.Config(s | {"matrix_instr_nonkdim": 16}, num_warps=8, num_stages=2)
        for s in sizes
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Here we assume A is row-major, B is column-major, C is row-major.
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_bk,
    stride_cm,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    ACTIVATION: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # Add some integer bound assumptions.
    # This helps to guide integer analysis in the backend to optimize
    # load/store offset address calculation
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :])

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def matmul(a, b, activation="", print_config=False, print_perf=False):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    # Benchmark the kernel execution if requested
    if print_perf:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul_kernel[grid](
                a,
                b,
                c,
                M,
                N,
                K,
                a.stride(0),
                b.stride(0),
                c.stride(0),
                ACTIVATION=activation,
            ),
            quantiles=[0.5, 0.2, 0.8],
        )
    else:
        matmul_kernel[grid](
            a,
            b,
            c,  #
            M,
            N,
            K,  #
            a.stride(0),
            b.stride(0),
            c.stride(0),
            ACTIVATION=activation,  #
        )

    # Print performance metrics
    if print_perf:
        # Calculate FLOPS: 2*M*N*K operations for matrix multiplication
        flops = 2.0 * M * N * K
        tflops = flops / (ms * 1e-3) / 1e12
        tflops_min = flops / (max_ms * 1e-3) / 1e12
        tflops_max = flops / (min_ms * 1e-3) / 1e12
        print(f"\n{'='*60}")
        print(f"Performance for M={M}, N={N}, K={K}:")
        print(f"{'='*60}")
        print(f"  Execution time (median): {ms:.4f} ms")
        print(f"  Execution time (min): {min_ms:.4f} ms")
        print(f"  Execution time (max): {max_ms:.4f} ms")
        print(f"  TFLOPS (median): {tflops:.2f}")
        print(f"  TFLOPS (min): {tflops_min:.2f}")
        print(f"  TFLOPS (max): {tflops_max:.2f}")
        print(f"{'='*60}\n")

    # Print the best configuration after autotuning
    if print_config:
        # Construct the key the same way the autotuner does
        # The key includes the autotune key values (M, N, K) plus dtypes
        key = [M, N, K]
        # Add dtype information from tensors
        for arg in [a, b, c]:
            if hasattr(arg, "dtype"):
                key.append(str(arg.dtype))
        key = tuple(key)

        # Try to get from cache first, otherwise use best_config
        best_config = None
        if hasattr(matmul_kernel, "cache") and key in matmul_kernel.cache:
            best_config = matmul_kernel.cache[key]
        elif hasattr(matmul_kernel, "best_config"):
            best_config = matmul_kernel.best_config

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


# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).

torch.manual_seed(0)


def validate():
    a = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
    b = torch.rand((512, 512), device=DEVICE, dtype=torch.float16) - 0.5
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    print(f"triton_output_with_fp16_inputs={triton_output}")
    print(f"torch_output_with_fp16_inputs={torch_output}")

    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

    TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
    if TORCH_HAS_FP8 and is_cuda():
        torch.manual_seed(0)
        a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
        b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
        a = a.to(torch.float8_e5m2)
        # pre-transpose b for efficiency.
        b = b.T
        b = b.to(torch.float8_e5m2)
        triton_output = matmul(a, b)
        torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
        print(f"triton_output_with_fp8_inputs={triton_output}")
        print(f"torch_output_with_fp8_inputs={torch_output}")
        if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch differ")


# %%
# Benchmark
# ---------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now compare the performance of our kernel against that of cuBLAS or rocBLAS. Here we focus on square matrices,
# but feel free to arrange this script as you wish to benchmark any other matrix shape.


def benchmark_square():
    ref_lib = "cuBLAS" if is_cuda() else "rocBLAS"
    TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")

    configs = []
    for fp8_inputs in [False, True]:
        if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=[
                    "M",
                    "N",
                    "K",
                ],  # Argument names to use as an x-axis for the plot
                x_vals=[
                    128 * i for i in range(2, 33)
                ],  # Different possible values for `x_name`
                line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
                # Possible values for `line_arg`
                # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
                line_vals=(
                    ["triton"] if fp8_inputs else [ref_lib.lower(), "triton"]
                ),  # Label name for the lines
                line_names=(
                    ["Triton"] if fp8_inputs else [ref_lib, "Triton"]
                ),  # Line styles
                styles=[("green", "-"), ("blue", "-")],
                ylabel="TFLOPS",  # Label name for the y-axis
                plot_name="matmul-performance-"
                + (
                    "fp16" if not fp8_inputs else "fp8"
                ),  # Name for the plot, used also as a file name for saving the plot.
                args={"fp8_inputs": fp8_inputs},
            )
        )

    @triton.testing.perf_report(configs)
    def benchmark(M, N, K, provider, fp8_inputs):
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
        if TORCH_HAS_FP8 and fp8_inputs:
            a = a.to(torch.float8_e5m2)
            b = b.T
            b = b.to(torch.float8_e5m2)
        quantiles = [0.5, 0.2, 0.8]
        if provider == ref_lib.lower():
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.matmul(a, b), quantiles=quantiles
            )
        if provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: matmul(a, b), quantiles=quantiles
            )
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=True, print_data=True)


def simple_test():

    from pathlib import Path
    from track_triton_kernels import TritonKernelTracker

    output_dir = (Path(__file__).parent / "triton_kernel_tracking/example_gemm").resolve()
    tracker = TritonKernelTracker(output_dir, save_binaries=True, capture_args=True)
    tracker.disable()
    print(f"\nOutput directory: {output_dir}")

    M = 512
    K = 512
    N = 512
    torch.manual_seed(0)
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    triton_output = matmul(a, b, print_config=True, print_perf=True)
    torch_output = torch.matmul(a, b)
    if not torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
        print("❌ Triton and Torch differ")
    else:
        print("✅ Triton and Torch match")

    # Call once for tracking
    tracker.enable()
    torch_output = matmul(a, b)


def main():
    # validate()
    # benchmark_square()
    simple_test()


if __name__ == "__main__":
    main()
