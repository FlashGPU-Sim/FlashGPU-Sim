import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

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


single_config_tma_gemm_kernel = triton.autotune(
    configs=[TMA_GEMM_CONFIGS[3]],
    key=["M", "N", "K"],
)(kernel_tma_gemm)
multi_config_tma_gemm_kernel = triton.autotune(
    configs=TMA_GEMM_CONFIGS,
    key=["M", "N", "K"],
)(kernel_tma_gemm)


class DriverAccessError:
    def __getattr__(self, name):
        raise AssertionError(f"Offline tracking accessed the GPU driver through '{name}'")


def assert_raises(error_type, message, callback):
    try:
        callback()
    except error_type as error:
        assert message in str(error)
        return
    raise AssertionError(f"Expected {error_type.__name__}: {message}")


def load_tracked_fp16_tensor(arg_info):
    assert arg_info.arg_type == "tensor"
    assert arg_info.dtype == "float16"
    element_count = arg_info.size_bytes // torch.empty((), dtype=torch.float16).element_size()
    return torch.from_file(
        arg_info.data_file,
        shared=False,
        size=element_count,
        dtype=torch.float16,
    ).clone().reshape(arg_info.shape)


def check_online_default(output_dir):
    from triton.runtime.jit import JITFunction

    original_jit_run = JITFunction.run
    runtime = triton.knobs.runtime
    hook_snapshots = {
        runtime.kernel_load_end_hook: list(runtime.kernel_load_end_hook.calls),
        runtime.launch_enter_hook: list(runtime.launch_enter_hook.calls),
        runtime.launch_exit_hook: list(runtime.launch_exit_hook.calls),
    }
    sentinel = object()

    def fake_online_run(jit_function, *args, **kwargs):
        return sentinel

    JITFunction.run = fake_online_run
    try:
        tracker = tritontrace.Tracker(
            output_dir,
            save_binaries=False,
            capture_args=False,
        )
        result = kernel_tma_gemm[(1,)](
            None,
            None,
            None,
            1,
            1,
            1,
            **TMA_GEMM_CONFIGS[0].all_kwargs(),
        )
        assert tracker.mode == "online"
        assert tracker.target_name is None
        assert result is sentinel
    finally:
        JITFunction.run = original_jit_run
        for hook, calls in hook_snapshots.items():
            hook.calls[:] = calls
        shutil.rmtree(output_dir, ignore_errors=True)


def check_target_validation(output_dir):
    assert_raises(
        ValueError,
        "requires an explicit CUDA target",
        lambda: tritontrace.Tracker(output_dir, mode="offline"),
    )
    assert_raises(
        ValueError,
        "Invalid CUDA target",
        lambda: tritontrace.Tracker(output_dir, mode="offline", target="gfx942"),
    )


def check_offline_compilation(tracking_dir, cache_dir):
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.driver import driver
    from triton.runtime.jit import JITFunction

    original_cache_dir = os.environ.get("TRITON_CACHE_DIR")
    os.environ["TRITON_CACHE_DIR"] = str(cache_dir)
    original_driver_active = driver._active
    original_driver_default = driver._default
    original_jit_run = JITFunction.run
    original_autotuner_run = Autotuner.run
    driver._active = DriverAccessError()
    driver._default = DriverAccessError()

    try:
        tracker = tritontrace.Tracker(
            tracking_dir,
            mode="offline",
            target="sm120",
        )
        M = N = K = 128
        torch.manual_seed(0)
        a = torch.randn((M, K), dtype=torch.float16)
        b_original = torch.randn((K, N), dtype=torch.float16)
        b = b_original.T.contiguous()
        c = torch.zeros((M, N), dtype=torch.float16)
        cpu_reference = torch.matmul(a.float(), b_original.float()).half()
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_SIZE_M"])
            * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )
        plain_config = TMA_GEMM_CONFIGS[0]

        tracker.disable()
        skipped = kernel_tma_gemm[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            **plain_config.all_kwargs(),
        )
        assert skipped is None
        assert not tracker.compiled_kernels
        assert not tracker.kernel_launches

        tracker.enable()
        compiled = kernel_tma_gemm[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            **plain_config.all_kwargs(),
        )
        single_config_compiled = single_config_tma_gemm_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
        )

        assert compiled.module is None
        assert compiled.function is None
        assert single_config_compiled.module is None
        assert torch.count_nonzero(c).item() == 0
        assert len(tracker.compiled_kernels) == 2
        assert len(tracker.kernel_launches) == 2
        assert not (tracker.output_dir / "metadata").exists()
        assert tracker.kernel_launches[0].grid == (1, 1, 1)
        assert tracker.kernel_launches[0].block == (256, 1, 1)

        for launch_info in tracker.kernel_launches:
            tracked_a = load_tracked_fp16_tensor(launch_info.args_info[0])
            tracked_b = load_tracked_fp16_tensor(launch_info.args_info[1])
            tracked_c = load_tracked_fp16_tensor(launch_info.args_info[2])
            torch.testing.assert_close(tracked_a, a, rtol=0, atol=0)
            torch.testing.assert_close(tracked_b, b, rtol=0, atol=0)
            torch.testing.assert_close(tracked_c, c, rtol=0, atol=0)

            tracked_reference = torch.matmul(tracked_a.float(), tracked_b.float().T).half()
            torch.testing.assert_close(tracked_reference, cpu_reference, rtol=0, atol=0)

        binary_info = tracker.compiled_kernels[compiled.hash]
        assert Path(binary_info.binary_path).is_file()
        assert Path(binary_info.ptx_path).is_file()
        assert ".target sm_120" in Path(binary_info.ptx_path).read_text()
        launcher_dir = tracker.output_dir / "launchers"
        harness_path = launcher_dir / "kernel_tma_gemm_launch1_harness.cu"
        assert harness_path.is_file()
        harness = harness_path.read_text()
        assert "validate_tensor_output" not in harness
        assert "Validating outputs" not in harness
        assert "math.h" not in harness
        assert (launcher_dir / "kernel_tma_gemm_launch1_Makefile").is_file()
        assert (launcher_dir / "kernel_tma_gemm_launch1_kernel.ptx").is_file()
        assert (launcher_dir / "kernel_tma_gemm_launch1_kernel.cubin").is_file()
        subprocess.run(
            ["make", "-f", "kernel_tma_gemm_launch1_Makefile"],
            cwd=launcher_dir,
            check=True,
        )
        assert (launcher_dir / "kernel_tma_gemm_launch1").is_file()
        assert (launcher_dir / "kernel_tma_gemm_launch1_kernel.fatbin").is_file()

        assert_raises(
            RuntimeError,
            "cannot benchmark multiple",
            lambda: multi_config_tma_gemm_kernel[grid](
                a,
                b,
                c,
                M,
                N,
                K,
            ),
        )

        tracker.save_summary()
        summary = json.loads((tracker.output_dir / "tracking_summary.json").read_text())
        assert summary["tracking_session"]["mode"] == "offline"
        assert summary["tracking_session"]["target"] == "sm120"
    finally:
        driver._active = original_driver_active
        driver._default = original_driver_default
        JITFunction.run = original_jit_run
        Autotuner.run = original_autotuner_run
        if original_cache_dir is None:
            os.environ.pop("TRITON_CACHE_DIR", None)
        else:
            os.environ["TRITON_CACHE_DIR"] = original_cache_dir
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_tracker_no_gpu():
    run_root = Path(__file__).resolve().parent / "run"
    tracking_dir = run_root / "no-gpu-tracking"
    cache_dir = run_root / "no-gpu-triton-cache"
    shutil.rmtree(tracking_dir, ignore_errors=True)
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"Test output: {tracking_dir}")
    check_online_default(run_root / "no-gpu-online-default")
    check_target_validation(run_root / "no-gpu-target-validation")
    check_offline_compilation(tracking_dir, cache_dir)
    assert not cache_dir.exists()
    print("TritonTrace no-GPU test PASSED")


def main():
    test_tracker_no_gpu()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
