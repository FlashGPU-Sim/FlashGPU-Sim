"""Capture fixed non-TMA GEMM cases for RTX 4060 calibration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.language as tl
import TritonTrace


REPO_ROOT = Path(__file__).resolve().parents[3]
CASES_CSV = Path(os.environ.get("GEMM_CASES_CSV", Path(__file__).with_name("gemm_cases.csv")))
OUTPUT_ROOT = REPO_ROOT / "reference" / "SM89_RTX4060" / "gemm"
SEED = 20261024
INTEGER_CASE_FIELDS = {
    "M", "N", "K", "BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_SIZE_M",
    "num_warps", "num_stages",
}


@triton.jit
def fixed_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + rows[:, None] * K + ks[None, :]
    b_ptrs = b_ptr + ks[:, None] * N + cols[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            a_ptrs,
            mask=(rows[:, None] < M) & (ks[None, :] + k_block * BLOCK_K < K),
            other=0,
        )
        b = tl.load(
            b_ptrs,
            mask=(ks[:, None] + k_block * BLOCK_K < K) & (cols[None, :] < N),
            other=0,
        )
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    output = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + rows[:, None] * N + cols[None, :]
    tl.store(c_ptrs, output, mask=(rows[:, None] < M) & (cols[None, :] < N))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_case(case_id: str, dtype_name: str) -> dict[str, str | int]:
    with CASES_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if row["case_id"] == case_id and row["dtype"] == dtype_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one CSV row for case={case_id}, dtype={dtype_name}; found {len(matches)}")
    result: dict[str, str | int] = {}
    for key, value in matches[0].items():
        result[key] = int(value) if key in INTEGER_CASE_FIELDS else value
    return result


def gpu_info() -> dict[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
        names = ["name", "compute_cap", "driver_version", "sm_clock_mhz", "mem_clock_mhz", "temperature_c", "power_w", "power_limit_w"]
        return dict(zip(names, [part.strip() for part in output.split(",")]))
    except (OSError, subprocess.CalledProcessError):
        return {"error": "nvidia-smi query unavailable"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--dtype", choices=("fp16",), required=True)
    args = parser.parse_args()
    case = load_case(args.case, args.dtype)
    output_dir = OUTPUT_ROOT / str(case["split"]) / args.dtype / args.case
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite captured case: {output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is unavailable")
    capability = torch.cuda.get_device_capability(0)
    if capability != (8, 9):
        raise RuntimeError(f"Expected RTX 4060 SM89, got {torch.cuda.get_device_name(0)} with capability {capability}")

    dtype = torch.float16
    M, N, K = (int(case[key]) for key in ("M", "N", "K"))
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    a = (torch.rand((M, K), generator=generator, dtype=torch.float32) - 0.5).to(dtype).to("cuda")
    b = (torch.rand((K, N), generator=generator, dtype=torch.float32) - 0.5).to(dtype).to("cuda")
    c = torch.empty((M, N), device="cuda", dtype=dtype)
    bm, bn, bk = (int(case[key]) for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
    group = int(case["GROUP_SIZE_M"])
    warps, stages = int(case["num_warps"]), int(case["num_stages"])
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)

    def launch():
        return fixed_gemm_kernel[grid](
            a, b, c, M, N, K,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_SIZE_M=group,
            num_warps=warps, num_stages=stages,
        )

    output_dir.mkdir(parents=True)
    # Install load hooks before the first JIT compile; otherwise the online
    # tracker cannot associate the compiled function pointer with its PTX.
    tracker = TritonTrace.Tracker(output_dir, save_binaries=True, capture_args=True, enabled=False)
    reference = (a.float() @ b.float()).to(dtype)
    for _ in range(10):
        launch()
    torch.cuda.synchronize()
    tolerances = {"rtol": 1e-2, "atol": 1e-2} if args.dtype == "fp16" else {"rtol": 2e-2, "atol": 5e-2}
    torch.testing.assert_close(c, reference, **tolerances)

    c.zero_()
    torch.cuda.synchronize()
    tracker.enable()
    launch()
    torch.cuda.synchronize()
    tracker.disable()
    tracker.save_summary()

    makefiles = sorted(output_dir.rglob("*_launch1_Makefile"))
    if len(makefiles) != 1:
        raise RuntimeError(f"Expected exactly one launch Makefile, found {len(makefiles)}")
    subprocess.run(
        ["make", "--no-print-directory", "-C", str(makefiles[0].parent), "-f", makefiles[0].name],
        check=True,
    )

    ptx_files = sorted(output_dir.rglob("*.ptx"))
    cubin_files = sorted(output_dir.rglob("*.cubin"))
    if not ptx_files:
        raise RuntimeError("TritonTrace did not save a PTX artifact")
    manifest = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": case,
        "input": {"seed": SEED, "generator": "CPU torch.float32 uniform(-0.5,0.5), cast to dtype then CUDA"},
        "math": {
            "equation": "C[M,N] = A[M,K] @ B[K,N]",
            "layout": "A, B, C contiguous row-major",
            "input_dtype": args.dtype,
            "accumulator_dtype": "fp32",
            "output_dtype": args.dtype,
            "activation": "none",
            "alpha": 1.0,
            "beta": 0.0,
            "tolerance": tolerances,
            "correctness": "PASS",
        },
        "kernel": {
            "name": "fixed_gemm_kernel",
            "grid": grid,
            "block": [warps * 32, 1, 1],
            "BLOCK_M": bm,
            "BLOCK_N": bn,
            "BLOCK_K": bk,
            "GROUP_SIZE_M": group,
            "num_warps": warps,
            "num_stages": stages,
            "autotune": False,
            "tensor_descriptor": False,
            "target": f"sm_{capability[0]}{capability[1]}",
        },
        "software": {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "python": os.sys.version,
            "cuda_runtime": torch.version.cuda,
        },
        "gpu_before_capture": gpu_info(),
        "gpu_after_capture": gpu_info(),
        "artifacts": {
            str(path.relative_to(output_dir)): sha256(path)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file() and path.name != "experiment-manifest.json"
        },
        "source_sha256": sha256(Path(__file__)),
        "cases_csv_sha256": sha256(CASES_CSV),
    }
    (output_dir / "experiment-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CASE={args.case} DTYPE={args.dtype} SPLIT={case['split']}")
    print(f"GRID={grid} BLOCK={[warps * 32, 1, 1]} TILE={bm}x{bn}x{bk} WARPS={warps} STAGES={stages}")
    print(f"CORRECTNESS=PASS OUTPUT_DIR={output_dir}")
    for rel, digest in manifest["artifacts"].items():
        print(f"SHA256 {rel} {digest}")


if __name__ == "__main__":
    main()
