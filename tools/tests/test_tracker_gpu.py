import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import tritontrace


TMA_GEMM_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
        },
        num_stages=3,
        num_warps=8,
        num_ctas=1,
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
        num_ctas=1,
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
        num_ctas=1,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
        },
        num_stages=4,
        num_warps=4,
        num_ctas=1,
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
        num_ctas=1,
    ),
]


@triton.autotune(configs=TMA_GEMM_CONFIGS, key=["M", "N", "K"])
@triton.jit
def kernel_tma_gemm(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
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

    desc_a = tl.make_tensor_descriptor(
        base=A_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    desc_b = tl.make_tensor_descriptor(
        base=B_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    desc_c = tl.make_tensor_descriptor(
        base=C_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    offs_am = pid_m * BLOCK_SIZE_M
    offs_bn = pid_n * BLOCK_SIZE_N
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(k_tiles):
        offs_k = k * BLOCK_SIZE_K
        a = tl.load_tensor_descriptor(desc_a, [offs_am, offs_k])
        b = tl.load_tensor_descriptor(desc_b, [offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

    c = accumulator.to(tl.float16)
    tl.store_tensor_descriptor(desc_c, [offs_am, offs_bn], c)


def load_tracked_fp16_tensor(data_file, shape):
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    return torch.from_file(
        data_file,
        shared=False,
        size=element_count,
        dtype=torch.float16,
    ).clone().reshape(shape)


def check_gpu_tracking(tracking_dir, cache_dir):
    from triton.runtime._allocation import _allocator
    from triton.runtime.jit import JITFunction

    original_cache_dir = os.environ.get("TRITON_CACHE_DIR")
    original_allocator = _allocator.get()
    original_jit_run = JITFunction.run
    runtime = triton.knobs.runtime
    hook_snapshots = {
        runtime.kernel_load_end_hook: list(runtime.kernel_load_end_hook.calls),
        runtime.launch_enter_hook: list(runtime.launch_enter_hook.calls),
        runtime.launch_exit_hook: list(runtime.launch_exit_hook.calls),
    }
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)

    try:
        device = torch.device("cuda")
        triton.set_allocator(
            lambda size, alignment, stream: torch.empty(
                size, device=device, dtype=torch.uint8
            )
        )
        tracker = tritontrace.Tracker(tracking_dir, enabled=False)
        assert tracker.mode == "online"
        assert tracker.target_name is None

        M = N = K = 128
        torch.manual_seed(0)
        a_cpu = torch.randn((M, K), dtype=torch.float16)
        b_original_cpu = torch.randn((K, N), dtype=torch.float16)
        b_cpu = b_original_cpu.T.contiguous()
        cpu_reference = torch.matmul(a_cpu.float(), b_original_cpu.float()).half()

        a = a_cpu.to(device)
        b = b_cpu.to(device)
        c = torch.zeros((M, N), device=device, dtype=torch.float16)
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_SIZE_M"])
            * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )

        kernel_tma_gemm[grid](a, b, c, M, N, K)
        torch.cuda.synchronize()
        best_config = kernel_tma_gemm.best_config
        assert best_config in TMA_GEMM_CONFIGS

        c.zero_()
        tracker.enable()
        kernel_tma_gemm[grid](a, b, c, M, N, K)
        torch.cuda.synchronize()

        gpu_result = c.cpu()
        torch.testing.assert_close(gpu_result, cpu_reference, rtol=0.1, atol=0.1)
        assert len(tracker.compiled_kernels) == len(TMA_GEMM_CONFIGS)
        assert len(tracker.kernel_launches) == 1

        launch_info = tracker.kernel_launches[0]
        expected_grid = (
            triton.cdiv(M, best_config.kwargs["BLOCK_SIZE_M"])
            * triton.cdiv(N, best_config.kwargs["BLOCK_SIZE_N"]),
            1,
            1,
        )
        assert launch_info.grid == expected_grid
        assert launch_info.block == (best_config.num_warps * 32, 1, 1)
        assert launch_info.num_warps == best_config.num_warps
        assert launch_info.num_ctas == best_config.num_ctas
        assert launch_info.shared_memory > 0
        assert launch_info.global_scratch_size > 0
        assert launch_info.kernel_hash in tracker.compiled_kernels
        assert not (tracker.output_dir / "metadata").exists()

        tracked_a = load_tracked_fp16_tensor(
            launch_info.args_info[0].data_file, launch_info.args_info[0].shape
        )
        tracked_b = load_tracked_fp16_tensor(
            launch_info.args_info[1].data_file, launch_info.args_info[1].shape
        )
        tracked_c = load_tracked_fp16_tensor(
            launch_info.args_info[2].data_file, launch_info.args_info[2].shape
        )
        tracked_output = load_tracked_fp16_tensor(
            launch_info.args_info[2].output_file, launch_info.args_info[2].shape
        )

        torch.testing.assert_close(tracked_a, a_cpu, rtol=0, atol=0)
        torch.testing.assert_close(tracked_b, b_cpu, rtol=0, atol=0)
        torch.testing.assert_close(tracked_c, torch.zeros_like(tracked_c), rtol=0, atol=0)
        torch.testing.assert_close(tracked_output, gpu_result, rtol=0, atol=0)
        torch.testing.assert_close(tracked_output, cpu_reference, rtol=0.1, atol=0.1)
        assert launch_info.args_info[0].output_file is None
        assert launch_info.args_info[1].output_file is None

        kernel_info = tracker.compiled_kernels[launch_info.kernel_hash]
        assert Path(kernel_info.binary_path).is_file()
        assert Path(kernel_info.ptx_path).is_file()
        capability = torch.cuda.get_device_capability()
        assert f".target sm_{capability[0]}{capability[1]}" in Path(
            kernel_info.ptx_path
        ).read_text()

        tracker.save_summary()
        summary = json.loads((tracking_dir / "tracking_summary.json").read_text())
        assert summary["tracking_session"]["mode"] == "online"
        assert summary["tracking_session"]["target"] is None
        assert summary["tracking_session"]["total_launches"] == 1

        launcher_dir = tracking_dir / "launchers"
        harness_path = launcher_dir / "kernel_tma_gemm_launch1_harness.cu"
        harness = harness_path.read_text()
        assert "validate_tensor_output" in harness
        assert "Validating outputs" in harness
        assert "math.h" in harness

        subprocess.run(
            ["make", "-f", "kernel_tma_gemm_launch1_Makefile"],
            cwd=launcher_dir,
            check=True,
        )
        launcher_result = subprocess.run(
            [str(launcher_dir / "kernel_tma_gemm_launch1")],
            cwd=launcher_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        print(launcher_result.stdout, end="")
        assert "Validation PASSED for arg[2]" in launcher_result.stdout
    finally:
        JITFunction.run = original_jit_run
        for hook, calls in hook_snapshots.items():
            hook.calls[:] = calls
        triton.set_allocator(original_allocator)
        if original_cache_dir is None:
            os.environ.pop("TRITON_CACHE_DIR", None)
        else:
            os.environ["TRITON_CACHE_DIR"] = original_cache_dir
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_tracker_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("test_tracker_gpu.py requires a CUDA GPU")

    run_root = Path(__file__).resolve().parent / "run"
    tracking_dir = run_root / "gpu-tracking"
    cache_dir = run_root / "gpu-triton-cache"
    shutil.rmtree(tracking_dir, ignore_errors=True)
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"Test output: {tracking_dir}")
    check_gpu_tracking(tracking_dir, cache_dir)
    assert not cache_dir.exists()
    print("TritonTrace GPU test PASSED")


def main():
    test_tracker_gpu()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
