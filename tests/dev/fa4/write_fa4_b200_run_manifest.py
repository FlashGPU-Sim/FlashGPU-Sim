#!/usr/bin/env python3
"""Create a developer-facing record of an FA4 B200 run.

This helper deliberately uses only the Python standard library.  The run
driver remains responsible for executing CuTe DSL and GPGPU-Sim; this file
turns their artifacts and logs into a stable, readable contract.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


CASE_FIELDS = (
    "suite",
    "name",
    "batch",
    "seqlen_q",
    "seqlen_k",
    "heads",
    "head_dim",
    "head_dim_v",
    "dtype",
    "causal",
)

WARNING_PATTERNS = (
    ("ptx_reqntid_ignored", re.compile(r"\.reqntid ignored", re.IGNORECASE)),
    (
        "ptx_minnctapersm_ignored",
        re.compile(r"\.minnctapersm ignored", re.IGNORECASE),
    ),
    ("ptx_nounroll_ignored", re.compile(r"ignoring pragma ['\"]nounroll", re.IGNORECASE)),
    (
        "cuFuncGetAttribute_unimplemented",
        re.compile(
            r"(?:cuFuncGetAttribute.*not been implemented|"
            r"not been implemented.*cuFuncGetAttribute)",
            re.IGNORECASE,
        ),
    ),
    (
        "cudaFuncSetAttribute_ignored",
        re.compile(r"ignoring call to .*cudaFuncSetAttribute", re.IGNORECASE),
    ),
    (
        "python_or_compiler_deprecation",
        re.compile(r"deprecated|deprecationwarning", re.IGNORECASE),
    ),
)


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean: {value}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def repository_info(root: Path) -> dict[str, Any]:
    commit = command_output(["git", "rev-parse", "HEAD"], root)
    branch = command_output(["git", "branch", "--show-current"], root)
    status = command_output(["git", "status", "--porcelain"], root)
    return {
        "root": str(root),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
    }


def file_record(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return record
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    record.update({"bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return record


def load_cases(path: Path, suite: str, case_name: str | None) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as source:
        rows = csv.reader(line for line in source if not line.lstrip().startswith("#"))
        for values in rows:
            if not values:
                continue
            if len(values) != len(CASE_FIELDS):
                raise ValueError(f"invalid case row with {len(values)} columns: {values}")
            row = dict(zip(CASE_FIELDS, values))
            if case_name:
                if row["name"] != case_name:
                    continue
            elif suite != "all" and row["suite"] != suite:
                continue
            for field in ("batch", "seqlen_q", "seqlen_k", "heads", "head_dim", "head_dim_v"):
                row[field] = int(row[field])
            row["causal"] = parse_bool(str(row["causal"]))
            row["grid_x"] = (
                (int(row["seqlen_q"]) + 127) // 128
                * int(row["heads"])
                * int(row["batch"])
            )
            selected.append(row)
    return selected


def ptx_contract(ptx_path: Path) -> dict[str, Any]:
    if not ptx_path.is_file():
        return {}
    text = ptx_path.read_text(encoding="utf-8", errors="replace")
    version = re.search(r"^\.version\s+([^\s]+)", text, re.MULTILINE)
    target = re.search(r"^\.target\s+(.+)$", text, re.MULTILINE)
    reqntid = re.search(
        r"^\.reqntid\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", text, re.MULTILINE
    )
    minnctapersm = re.search(r"^\.minnctapersm\s+(\d+)", text, re.MULTILINE)
    return {
        "version": version.group(1) if version else None,
        "target": target.group(1).strip() if target else None,
        "entries": re.findall(r"\.visible\s+\.entry\s+([^\s(]+)", text),
        "reqntid": [int(value) for value in reqntid.groups()] if reqntid else None,
        "minnctapersm": int(minnctapersm.group(1)) if minnctapersm else None,
    }


def ptx_entries(ptx_path: Path) -> list[str]:
    return list(ptx_contract(ptx_path).get("entries", []))


def wrapper_abi(direction: str) -> dict[str, Any]:
    if direction == "fwd":
        parameters = ["module", "mQ", "mK", "mV", "mO", "softmax_scale", "stream"]
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


def manual_kernel_abi() -> dict[str, Any]:
    return {
        "parameters": [
            "coord_q",
            "coord_k",
            "coord_v",
            "coord_o",
            "tma_q",
            "tma_k",
            "tma_v",
            "tma_o",
            "softmax_scale",
            "tiny0",
            "tiny1",
            "seqlen_q",
            "seqlen_k",
            "head_dim",
            "heads",
            "batch",
            "stride",
            "vec12_0",
            "vec12_1",
            "vec12_2",
            "vec12_3",
            "grid_x",
            "vec12_4",
            "zero",
        ],
        "return": "cuLaunchKernel status",
        "authority": "manually reconstructed generated-MLIR kernel ABI",
    }


def fallback_launch_contract(run_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    artifacts = metadata.get("artifacts", {})
    candidates: list[Path] = []
    source_mlir = artifacts.get("source_mlir")
    if source_mlir:
        candidates.append(Path(source_mlir))
    dump_dir = artifacts.get("dump_dir")
    if dump_dir:
        candidates.extend(sorted(Path(dump_dir).glob("*.mlir")))
    candidates.extend(sorted((run_dir / "fa4_b200_launcher_dump").glob("*.mlir")))
    mlir_path = next((path for path in candidates if path.is_file()), None)
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


def log_warning_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    categories = {
        name: len(pattern.findall(text))
        for name, pattern in WARNING_PATTERNS
        if pattern.search(text)
    }
    warning_lines = sum(
        1
        for line in text.splitlines()
        if not re.match(r"^\s*-", line)
        and re.search(r"warning:|warning --|deprecated", line, re.IGNORECASE)
    )
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "warning_line_count": warning_lines,
        "warning_categories": categories,
    }


def simulator_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"log": str(path), "exists": False}
    text = path.read_text(encoding="utf-8", errors="replace")

    def last_value(pattern: str, cast: type[int] | type[float]) -> int | float | None:
        matches = re.findall(pattern, text, re.MULTILINE)
        if not matches:
            return None
        try:
            return cast(matches[-1])
        except ValueError:
            return None

    warning_summary = log_warning_summary(path)
    if "synchronized and numeric check passed" in text:
        check = "full_output_numeric"
    elif "synchronized backward smoke check passed" in text:
        check = "backward_input_integrity_smoke"
    elif re.search(r"^synchronized$", text, re.MULTILINE):
        check = "manual_input_canary"
    else:
        check = "not_observed"
    return {
        "log": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "warnings": {
            "line_count": warning_summary["warning_line_count"],
            "categories": warning_summary["warning_categories"],
        },
        "check": check,
        "statistics": {
            "cycles": last_value(r"^gpu_tot_sim_cycle\s*=\s*(\d+)", int),
            "instructions": last_value(r"^gpu_tot_sim_insn\s*=\s*(\d+)", int),
            "ipc": last_value(r"^gpu_ipc\s*=\s*([0-9.eE+-]+)", float),
            "wall_seconds": last_value(
                r"^gpgpu_simulation_time\s*=.*\((\d+) sec\)", int
            ),
        },
    }


def parse_results(values: list[str]) -> dict[str, dict[str, str]]:
    results: dict[str, dict[str, str]] = {}
    for value in values:
        parts = value.split("|", 2)
        if len(parts) != 3:
            raise ValueError(f"invalid --result value: {value}")
        name, status, log = parts
        results[name] = {"status": status, "log": log}
    return results


def verification_contract(launcher: str, direction: str) -> dict[str, str]:
    if launcher == "manual":
        return {
            "inputs": "manual harness canary patterns",
            "assertion": "K/V prefixes remain unchanged after synchronization",
            "strength": "launch/input-integrity only; O is not numerically checked",
        }
    if direction == "bwd":
        return {
            "inputs": "Q=K=0, V=dO=1 with analytical LSE",
            "assertion": "Q/K/V/dO inputs remain unchanged",
            "strength": "backward bring-up only; gradients are not numerically checked",
        }
    return {
        "inputs": "Q=K=0 and V=1",
        "assertion": "every element of O is exactly 1 in the selected 16-bit dtype",
        "strength": "full O tensor numeric check for this deterministic fixture",
    }


def write_manifest(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    run_dir = Path(args.run_dir).resolve()
    metadata_path = Path(args.metadata).resolve() if args.metadata else None
    metadata = read_json(metadata_path) if metadata_path else {}
    cases = load_cases(Path(args.cases_file), args.suite, args.case_name)
    results = parse_results(args.result)
    artifact_shape = {
        "direction": args.direction,
        "head_dim": args.head_dim,
        "head_dim_v": args.head_dim_v,
        "dtype": args.dtype,
        "causal": parse_bool(args.causal),
    }

    for case in cases:
        compatible = (
            case["head_dim"] == args.head_dim
            and case["head_dim_v"] == args.head_dim_v
            and case["dtype"] == args.dtype
            and case["causal"] == artifact_shape["causal"]
        )
        case["artifact_compatible"] = compatible
        result = results.get(str(case["name"]))
        if result:
            case["result"] = {"status": result["status"]}
            if result["log"]:
                case["result"].update(simulator_summary(Path(result["log"])))
        else:
            case["result"] = {"status": "not_run"}

    ptx_path = Path(args.ptx).resolve() if args.ptx else None
    artifact_paths: dict[str, Path | None] = {
        "metadata": metadata_path,
        "header": Path(args.header).resolve() if args.header else None,
        "object": Path(args.object).resolve() if args.object else None,
        "ptx": ptx_path,
        "cubin": Path(args.cubin).resolve() if args.cubin else None,
        "runner": Path(args.runner).resolve() if args.runner else None,
    }
    if args.gpgpusim_config:
        artifact_paths["gpgpusim_config"] = Path(args.gpgpusim_config).resolve()
    for index, path in enumerate(args.interconnect_config):
        artifact_paths[f"interconnect_config_{index}"] = Path(path).resolve()
    if args.fatbin:
        artifact_paths["fatbin"] = Path(args.fatbin).resolve()

    ptxas_version = None
    if args.ptxas_root:
        ptxas_version = command_output([str(Path(args.ptxas_root) / "bin" / "ptxas"), "--version"])

    log_dir = Path(args.log_dir).resolve()
    stage_logs = {
        name: log_warning_summary(log_dir / filename)
        for name, filename in (
            ("export", "export.log"),
            ("environment", "environment.log"),
            ("build", "build.log"),
        )
    }
    if args.launcher == "manual":
        recorded_wrapper_abi = manual_kernel_abi()
        recorded_launch = {
            "block": [args.manual_block_x, 1, 1],
            "dynamic_smem_bytes": args.dynamic_smem,
            "grid_formula": "ceil(seqlen_q / 128) * heads * batch, 1, 1",
            "authority": "manual harness command line",
        }
    else:
        recorded_wrapper_abi = metadata.get("wrapper_abi") or wrapper_abi(
            args.direction
        )
        recorded_launch = metadata.get("launch") or fallback_launch_contract(
            run_dir, metadata
        )
    recorded_ptx_contract = metadata.get("ptx_contract")
    if not recorded_ptx_contract and ptx_path:
        recorded_ptx_contract = ptx_contract(ptx_path)
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "updated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "invocation": {"phase": args.phase, "argv": args.invocation_arg},
        "repository": repository_info(Path(args.repo_root).resolve()),
        "selection": {
            "config": args.config,
            "suite": args.suite,
            "case": args.case_name,
            "launcher": args.launcher,
            "run_dir": str(run_dir),
        },
        "environment": {
            "fa4_python": args.fa4_python or None,
            "cuda_install_path": args.cuda_root or None,
            "ptxas_cuda_install_path": args.ptxas_root or None,
            "ptxas_version": ptxas_version,
        },
        "execution": {
            "affinity_mode": args.affinity_mode,
            "cpu_set": args.cpu_set or None,
            "cpus_per_job": args.cpus_per_job,
            "threads_per_job": args.threads_per_job,
            "queue_job_id": args.queue_job_id or None,
            "queue_slot": args.queue_slot,
        },
        "artifact": {
            "specialization": artifact_shape,
            "function_name": metadata.get("function_name"),
            "software": metadata.get("software", {}),
            "compiler_toolchain": metadata.get("compiler_toolchain", {}),
            "wrapper_abi": recorded_wrapper_abi,
            "launch": recorded_launch,
            "ptx_contract": recorded_ptx_contract or {},
            "ptx_entries": (
                recorded_ptx_contract.get("entries", [])
                if recorded_ptx_contract
                else []
            ),
            "files": {
                name: file_record(path) for name, path in artifact_paths.items()
            },
        },
        "verification": verification_contract(args.launcher, args.direction),
        "cases": cases,
        "logs": {
            "directory": str(log_dir),
            "stages": stage_logs,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--case", dest="case_name")
    parser.add_argument("--launcher", choices=("generated", "manual"), required=True)
    parser.add_argument("--direction", choices=("fwd", "bwd"), required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--head-dim-v", type=int, required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--causal", required=True)
    parser.add_argument("--dynamic-smem", type=int, required=True)
    parser.add_argument("--manual-block-x", type=int, default=512)
    parser.add_argument("--fa4-python")
    parser.add_argument("--cuda-root")
    parser.add_argument("--ptxas-root")
    parser.add_argument("--affinity-mode", required=True)
    parser.add_argument("--cpu-set")
    parser.add_argument("--cpus-per-job", type=int, required=True)
    parser.add_argument("--threads-per-job", type=int, required=True)
    parser.add_argument("--queue-job-id")
    parser.add_argument("--queue-slot", type=int)
    parser.add_argument("--cases-file", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--header")
    parser.add_argument("--object")
    parser.add_argument("--ptx")
    parser.add_argument("--cubin")
    parser.add_argument("--runner")
    parser.add_argument("--gpgpusim-config")
    parser.add_argument("--interconnect-config", action="append", default=[])
    parser.add_argument("--fatbin")
    parser.add_argument("--result", action="append", default=[])
    parser.add_argument("--invocation-arg", action="append", default=[])

    return parser.parse_args()


def main() -> int:
    return write_manifest(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
