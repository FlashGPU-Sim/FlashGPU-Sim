#!/usr/bin/env python3
"""Export the real FA4/CuTeDSL B200 host launcher as C artifacts.

The generated .h/.o pair comes from CuTeDSL's export_to_c() path.  This keeps
the FA4 host wrapper responsible for TMA descriptors, launch shape, dynamic
shared memory, and kernel argument packing.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--export-name", default="fa4_b200_launcher")
    parser.add_argument("--direction", choices=("fwd", "bwd"), default="fwd")
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--arch", default="sm_100a")
    parser.add_argument("--flash-attn-arch", default="sm_100")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen-q", type=int, default=128)
    parser.add_argument("--seqlen-k", type=int, default=128)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--head-dim-v", type=int, default=None)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--non-causal", action="store_true")
    parser.add_argument(
        "--disable-2cta",
        action="store_true",
        help="Force FA4 to avoid cta_group::2 kernels during export",
    )
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def tool_version(path: str) -> str | None:
    try:
        result = subprocess.run(
            [path, "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def configure_environment(
    args: argparse.Namespace, dump_dir: Path
) -> dict[str, object]:
    os.environ["CUTE_DSL_ARCH"] = args.arch
    os.environ["FLASH_ATTENTION_ARCH"] = args.flash_attn_arch
    os.environ["CUTE_DSL_KEEP"] = "ptx,cubin,ir"
    os.environ["CUTE_DSL_DUMP_DIR"] = str(dump_dir)
    os.environ.setdefault("QUACK_CUTE_DSL_SHIM", "0")
    os.environ.pop("CUTE_DSL_PTXAS_PATH", None)
    if args.disable_2cta:
        os.environ["FA_DISABLE_2CTA"] = "1"

    default_cu13_root = Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "cu13"
    cu13_root = Path(os.environ.get("FA4_CU13_ROOT", default_cu13_root))
    if cu13_root.exists():
        # This is local to the exporter child process.  The simulator build
        # keeps using CUDA_INSTALL_PATH (normally CUDA 12.8).
        os.environ["CUDA_HOME"] = str(cu13_root)
        os.environ["CUDA_PATH"] = str(cu13_root)
        os.environ["PATH"] = (
            str(cu13_root / "bin") + os.pathsep + os.environ.get("PATH", "")
        )
        os.environ["LD_LIBRARY_PATH"] = (
            str(cu13_root / "lib")
            + os.pathsep
            + os.environ.get("LD_LIBRARY_PATH", "")
        )
    ptxas = shutil.which("ptxas")
    return {
        "cuda_home": os.environ.get("CUDA_HOME"),
        "cuda_path": os.environ.get("CUDA_PATH"),
        "ptxas": ptxas,
        "ptxas_version": tool_version(ptxas) if ptxas else None,
    }


def first_artifact(dump_dir: Path, suffix: str) -> Path | None:
    matches = sorted(dump_dir.glob(f"*.{suffix}"))
    return matches[0] if matches else None


def function_artifact(dump_dir: Path, suffix: str, function_name: str) -> Path | None:
    matches = sorted(
        path for path in dump_dir.glob(f"*.{suffix}")
        if function_name in path.name
    )
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple .{suffix} artifacts matched {function_name}: "
            + ", ".join(str(path) for path in matches)
        )
    if matches:
        return matches[0]
    return first_artifact(dump_dir, suffix)


def distribution_info(
    module_name: str, fallback_distributions: tuple[str, ...] = ()
) -> dict[str, object]:
    """Return package provenance without importing another runtime module."""
    distributions = importlib.metadata.packages_distributions().get(module_name, [])
    distributions = [*distributions, *fallback_distributions]
    if not distributions:
        return {}
    distribution_name = distributions[0]
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return {"distribution": distribution_name}
    info: dict[str, object] = {
        "distribution": distribution_name,
        "version": distribution.version,
    }
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        try:
            info["direct_url"] = json.loads(direct_url)
        except json.JSONDecodeError:
            info["direct_url"] = direct_url
    return info


def ptx_contract(ptx_path: Path) -> dict[str, object]:
    text = ptx_path.read_text(encoding="utf-8", errors="replace")
    entries = re.findall(r"\.visible\s+\.entry\s+([^\s(]+)", text)
    version = re.search(r"^\.version\s+([^\s]+)", text, re.MULTILINE)
    target = re.search(r"^\.target\s+(.+)$", text, re.MULTILINE)
    reqntid = re.search(
        r"^\.reqntid\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",
        text,
        re.MULTILINE,
    )
    minnctapersm = re.search(r"^\.minnctapersm\s+(\d+)", text, re.MULTILINE)
    return {
        "version": version.group(1) if version else None,
        "target": target.group(1).strip() if target else None,
        "entries": entries,
        "reqntid": [int(value) for value in reqntid.groups()] if reqntid else None,
        "minnctapersm": int(minnctapersm.group(1)) if minnctapersm else None,
    }


def launch_contract(mlir_path: Path | None) -> dict[str, object]:
    if mlir_path is None:
        return {}
    text = mlir_path.read_text(encoding="utf-8", errors="replace")
    line = next(
        (line.strip() for line in text.splitlines() if "cuda.launch_cfg.create" in line),
        None,
    )
    if line is None:
        return {"source_mlir": str(mlir_path)}
    block = re.search(
        r"blockDim\s*=\s*\(%c(\d+)_i32,\s*%c(\d+)_i32,\s*%c(\d+)_i32\)",
        line,
    )
    dynamic_smem = re.search(r"dynamicSmemBytes\s*=\s*%c(\d+)_i64", line)
    return {
        "block": [int(value) for value in block.groups()] if block else None,
        "dynamic_smem_bytes": int(dynamic_smem.group(1)) if dynamic_smem else None,
        "grid_formula": "computed by the generated wrapper from runtime tensor shapes",
        "source_mlir": str(mlir_path),
        "source_operation": line,
    }


def wrapper_abi(direction: str) -> dict[str, object]:
    if direction == "fwd":
        parameters = [
            "module",
            "mQ",
            "mK",
            "mV",
            "mO",
            "softmax_scale",
            "stream",
        ]
    else:
        parameters = [
            "module",
            "mQ",
            "mK",
            "mV",
            "mdO",
            "mLSE",
            "mdPsum",
            "mdQaccum",
            "mdK",
            "mdV",
            "softmax_scale",
            "stream",
        ]
    return {
        "parameters": parameters,
        "return": "int32 CUDA status",
        "authority": "CuTe DSL export_to_c generated header",
    }


def select_compiled_function(compiled_functions, direction: str):
    if direction == "fwd":
        allowed = (
            "FlashAttentionForwardSm100",
            "BlackwellFusedMultiHeadAttentionForward",
        )
    else:
        allowed = (
            "FlashAttentionBackwardSm100",
            "BlackwellFusedMultiHeadAttentionBackward",
        )
    candidates = [
        fn for cls_name, fn in compiled_functions
        if cls_name in allowed
    ]
    if not candidates:
        seen = ", ".join(cls_name for cls_name, _ in compiled_functions)
        raise RuntimeError(
            f"did not capture an FA4 {direction} main kernel compile; "
            f"captured=[{seen}]"
        )
    return candidates[-1]


def main() -> int:
    args = parse_args()
    export_dir = Path(args.export_dir).resolve()
    dump_dir = Path(args.dump_dir).resolve()
    if args.clean:
        shutil.rmtree(export_dir, ignore_errors=True)
        shutil.rmtree(dump_dir, ignore_errors=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)
    compiler_toolchain = configure_environment(args, dump_dir)

    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    import cutlass.cute as cute

    compiled_functions = []
    original_compile = cute.compile

    def compile_for_c_export(*compile_args, **compile_kwargs):
        compile_kwargs.setdefault("no_jit_engine", True)
        # FA4 asks for TVM-FFI during normal execution.  For this runner we want
        # the generated C ABI export instead.
        compile_kwargs.pop("options", None)
        fn = original_compile(*compile_args, **compile_kwargs)
        obj = compile_args[0] if compile_args else None
        compiled_functions.append((obj.__class__.__name__, fn))
        return fn

    cute.compile = compile_for_c_export

    from flash_attn.cute import interface

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    head_dim_v = args.head_dim if args.head_dim_v is None else args.head_dim_v
    causal = not args.non_causal

    with FakeTensorMode():
        q = torch.empty(
            (args.batch, args.seqlen_q, args.heads, args.head_dim),
            device="cuda",
            dtype=dtype,
        )
        k = torch.empty(
            (args.batch, args.seqlen_k, args.heads, args.head_dim),
            device="cuda",
            dtype=dtype,
        )
        v = torch.empty(
            (args.batch, args.seqlen_k, args.heads, head_dim_v),
            device="cuda",
            dtype=dtype,
        )
        if args.direction == "fwd":
            out, lse = interface._flash_attn_fwd(
                q,
                k,
                v,
                softmax_scale=1.0 / math.sqrt(args.head_dim),
                causal=causal,
                return_lse=False,
            )
        else:
            out = torch.empty(
                (args.batch, args.seqlen_q, args.heads, head_dim_v),
                device="cuda",
                dtype=dtype,
            )
            dout = torch.empty_like(out)
            lse = torch.empty(
                (args.batch, args.heads, args.seqlen_q),
                device="cuda",
                dtype=torch.float32,
            )
            interface._flash_attn_bwd(
                q,
                k,
                v,
                out,
                dout,
                lse,
                softmax_scale=1.0 / math.sqrt(args.head_dim),
                causal=causal,
            )

    if not compiled_functions:
        raise RuntimeError("FA4 did not compile a CuTeDSL function")

    fn = select_compiled_function(compiled_functions, args.direction)
    fn.export_to_c(str(export_dir), args.export_name, args.export_name)
    header_path = export_dir / f"{args.export_name}.h"
    object_path = export_dir / f"{args.export_name}.o"
    if not header_path.exists():
        raise RuntimeError(f"CuTeDSL export did not create {header_path}")
    if not object_path.exists():
        raise RuntimeError(f"CuTeDSL export did not create {object_path}")

    ptx_path = function_artifact(dump_dir, "ptx", fn.function_name)
    cubin_path = function_artifact(dump_dir, "cubin", fn.function_name)
    mlir_path = function_artifact(dump_dir, "mlir", fn.function_name)
    canonical_ptx = export_dir / f"{args.export_name}.ptx"
    canonical_cubin = export_dir / f"{args.export_name}.cubin"
    if ptx_path is not None:
        shutil.copy2(ptx_path, canonical_ptx)
    if cubin_path is not None:
        shutil.copy2(cubin_path, canonical_cubin)
    if not canonical_ptx.exists():
        raise RuntimeError(
            f"CuTeDSL export did not dump a PTX sidecar under {dump_dir}"
        )

    metadata = {
        "direction": args.direction,
        "arch": args.arch,
        "flash_attention_arch": args.flash_attn_arch,
        "export_name": args.export_name,
        "function_name": fn.function_name,
        "captured_compiles": [cls_name for cls_name, _ in compiled_functions],
        "compiler_toolchain": compiler_toolchain,
        "software": {
            "python": {
                "version": platform.python_version(),
                "executable": sys.executable,
            },
            "torch": {
                **distribution_info("torch"),
                "version": torch.__version__,
            },
            "cutlass_dsl": distribution_info(
                "cutlass", ("nvidia-cutlass-dsl",)
            ),
            "flash_attn": distribution_info("flash_attn"),
        },
        "wrapper_abi": wrapper_abi(args.direction),
        "ptx_contract": ptx_contract(canonical_ptx),
        "launch": launch_contract(mlir_path),
        "shape": {
            "batch": args.batch,
            "seqlen_q": args.seqlen_q,
            "seqlen_k": args.seqlen_k,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "head_dim_v": head_dim_v,
            "dtype": args.dtype,
            "causal": causal,
        },
        "fake_output": {
            "shape": list(out.shape),
            "dtype": str(out.dtype),
            "device": str(out.device),
            "has_lse": lse is not None,
        },
        "artifacts": {
            "header": str(header_path),
            "object": str(object_path),
            "ptx": str(canonical_ptx) if canonical_ptx.exists() else None,
            "cubin": str(canonical_cubin) if canonical_cubin.exists() else None,
            "source_mlir": str(mlir_path) if mlir_path is not None else None,
            "dump_dir": str(dump_dir),
        },
    }
    metadata_path = export_dir / f"{args.export_name}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"exported {args.export_name} to {export_dir}")
    print(f"metadata {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
