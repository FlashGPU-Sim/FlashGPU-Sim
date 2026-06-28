#!/usr/bin/env python3
"""Export the real FA4/CuTeDSL B200 forward host launcher as C artifacts.

The generated .h/.o pair comes from CuTeDSL's export_to_c() path.  This keeps
the FA4 host wrapper responsible for TMA descriptors, launch shape, dynamic
shared memory, and kernel argument packing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sysconfig
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--export-name", default="fa4_b200_launcher")
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
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace, dump_dir: Path) -> None:
    os.environ["CUTE_DSL_ARCH"] = args.arch
    os.environ["FLASH_ATTENTION_ARCH"] = args.flash_attn_arch
    os.environ["CUTE_DSL_KEEP"] = "ptx,cubin,ir"
    os.environ["CUTE_DSL_DUMP_DIR"] = str(dump_dir)
    os.environ.setdefault("QUACK_CUTE_DSL_SHIM", "0")
    os.environ.pop("CUTE_DSL_PTXAS_PATH", None)

    cu13_root = (
        Path(sysconfig.get_paths()["purelib"])
        / "nvidia"
        / "cu13"
    )
    if cu13_root.exists():
        os.environ.setdefault("CUDA_HOME", str(cu13_root))
        os.environ.setdefault("CUDA_PATH", str(cu13_root))
        os.environ["PATH"] = (
            str(cu13_root / "bin") + os.pathsep + os.environ.get("PATH", "")
        )
        os.environ["LD_LIBRARY_PATH"] = (
            str(cu13_root / "lib")
            + os.pathsep
            + os.environ.get("LD_LIBRARY_PATH", "")
        )


def first_artifact(dump_dir: Path, suffix: str) -> Path | None:
    matches = sorted(dump_dir.glob(f"*.{suffix}"))
    return matches[0] if matches else None


def main() -> int:
    args = parse_args()
    export_dir = Path(args.export_dir).resolve()
    dump_dir = Path(args.dump_dir).resolve()
    if args.clean:
        shutil.rmtree(export_dir, ignore_errors=True)
        shutil.rmtree(dump_dir, ignore_errors=True)
    export_dir.mkdir(parents=True, exist_ok=True)
    dump_dir.mkdir(parents=True, exist_ok=True)
    configure_environment(args, dump_dir)

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
        compiled_functions.append(fn)
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
        out, lse = interface._flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=1.0 / math.sqrt(args.head_dim),
            causal=causal,
            return_lse=False,
        )

    if not compiled_functions:
        raise RuntimeError("FA4 did not compile a CuTeDSL function")

    fn = compiled_functions[-1]
    fn.export_to_c(str(export_dir), args.export_name, args.export_name)
    header_path = export_dir / f"{args.export_name}.h"
    object_path = export_dir / f"{args.export_name}.o"
    if not header_path.exists():
        raise RuntimeError(f"CuTeDSL export did not create {header_path}")
    if not object_path.exists():
        raise RuntimeError(f"CuTeDSL export did not create {object_path}")

    ptx_path = first_artifact(dump_dir, "ptx")
    cubin_path = first_artifact(dump_dir, "cubin")
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
        "arch": args.arch,
        "flash_attention_arch": args.flash_attn_arch,
        "export_name": args.export_name,
        "function_name": fn.function_name,
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
