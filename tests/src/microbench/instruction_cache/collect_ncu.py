#!/usr/bin/env python3
"""Collect aggregate ICC/GCC counters for static instruction footprints."""

from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_STEPS = (
    64,
    128,
    256,
    320,
    336,
    352,
    368,
    512,
    1024,
    2048,
    3584,
    3840,
    3968,
    4032,
    4096,
    4160,
    4352,
    4608,
    5120,
    8192,
    9216,
    9728,
    10240,
    10496,
    10624,
    10752,
    10880,
    11008,
    11264,
    11520,
    12288,
    16384,
)

METRICS = (
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg",
    "smsp__inst_executed.sum",
    "smsp__warps_issue_stalled_no_instruction.sum",
    "sm__icc_requests.sum",
    "sm__icc_requests_lookup_hit.sum",
    "sm__icc_requests_lookup_miss.sum",
    "sm__icc_requests_lookup_miss_tag_hit.sum",
    "sm__icc_requests_lookup_miss_tag_miss.sum",
    "sm__icc_requests_lookup_miss_tag_unavailable.sum",
    "gcc__cache_requests_type_instruction.sum",
    "gcc__cache_requests_type_instruction_lookup_hit.sum",
    "gcc__cache_requests_type_instruction_lookup_miss.sum",
    "gcc__cache_requests_type_instruction_lookup_miss_tag_hit.sum",
    "gcc__cache_requests_type_instruction_lookup_miss_tag_miss.sum",
    "gcc__gcc2xbar_requests_type_instruction.sum",
    "gcc__cycles_elapsed.sum",
    "gcc__cycles_elapsed.avg",
    "lts__t_requests_srcunit_gcc.sum",
    "lts__t_sectors_srcunit_gcc.sum",
)

REQUIRED_BASE_METRICS = {
    "sm__icc_requests",
    "sm__icc_requests_lookup_hit",
    "sm__icc_requests_lookup_miss",
    "gcc__cache_requests_type_instruction",
    "gcc__cache_requests_type_instruction_lookup_hit",
    "gcc__cache_requests_type_instruction_lookup_miss",
}

IDENTITY_COLUMNS = (
    "mode",
    "steps",
    "text_bytes",
    "blocks",
    "repetitions",
    "warmup_launches",
    "device",
    "kernel",
)

DERIVED_COLUMNS = (
    "icc_miss_rate",
    "gcc_miss_rate",
    "inst_per_gcc_request",
    "lts_sectors_per_gcc2xbar_request",
    "gcc_instances",
    "icc_identity_error",
    "gcc_identity_error",
)


def metric_base(metric: str) -> str:
    return re.sub(r"\.(sum|avg|max|min)$", "", metric)


def parse_steps(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 or item >= 32768 for item in result):
        raise argparse.ArgumentTypeError(
            "steps must be comma-separated values in [1, 32767]"
        )
    return result


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, capture_output=True)


def query_metrics(ncu: str) -> set[str]:
    result = run([ncu, "--query-metrics"])
    if result.returncode != 0:
        raise RuntimeError(f"failed to query NCU metrics:\n{result.stderr}")
    metrics: set[str] = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if fields and re.fullmatch(r"[a-z][a-z0-9_]+", fields[0]):
            metrics.add(fields[0])
    return metrics


def parse_ncu_csv(output: str) -> dict[str, str]:
    rows = list(csv.reader(io.StringIO(output)))
    header_index = next(
        (index for index, row in enumerate(rows) if row and row[0] == "ID"), None
    )
    if header_index is None:
        raise ValueError("NCU output does not contain a raw CSV header")
    header = rows[header_index]
    data_rows = [
        row
        for row in rows[header_index + 1 :]
        if len(row) == len(header) and row and row[0].isdigit()
    ]
    if len(data_rows) != 1:
        raise ValueError(f"expected one profiled kernel row, found {len(data_rows)}")
    return dict(zip(header, data_rows[0]))


def kernel_text_bytes(cuobjdump: str, binary: Path) -> int:
    result = run([cuobjdump, "--dump-elf", str(binary)])
    if result.returncode != 0:
        raise RuntimeError(f"cuobjdump failed for {binary}:\n{result.stderr}")
    symbol = re.compile(
        r"^\s*0x[0-9a-f]+\s+\S+\s+(0x[0-9a-f]+)\s+.*footprint_kernel",
        re.IGNORECASE,
    )
    sizes = []
    for line in result.stdout.splitlines():
        match = symbol.search(line)
        if match:
            sizes.append(int(match.group(1), 16))
    if not sizes:
        raise ValueError(f"could not find footprint_kernel text size in {binary}")
    return max(sizes)


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if not value:
        return 0.0
    return float(value.replace(",", ""))


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def derived_values(row: dict[str, str]) -> dict[str, str]:
    icc_total = number(row, "sm__icc_requests.sum")
    icc_hit = number(row, "sm__icc_requests_lookup_hit.sum")
    icc_miss = number(row, "sm__icc_requests_lookup_miss.sum")
    gcc_total = number(row, "gcc__cache_requests_type_instruction.sum")
    gcc_hit = number(row, "gcc__cache_requests_type_instruction_lookup_hit.sum")
    gcc_miss = number(row, "gcc__cache_requests_type_instruction_lookup_miss.sum")
    gcc_to_l2 = number(row, "gcc__gcc2xbar_requests_type_instruction.sum")
    gcc_cycles_sum = number(row, "gcc__cycles_elapsed.sum")
    gcc_cycles_avg = number(row, "gcc__cycles_elapsed.avg")
    values = {
        "icc_miss_rate": ratio(icc_miss, icc_total),
        "gcc_miss_rate": ratio(gcc_miss, gcc_total),
        "inst_per_gcc_request": ratio(
            number(row, "smsp__inst_executed.sum"), gcc_total
        ),
        "lts_sectors_per_gcc2xbar_request": ratio(
            number(row, "lts__t_sectors_srcunit_gcc.sum"), gcc_to_l2
        ),
        "gcc_instances": ratio(gcc_cycles_sum, gcc_cycles_avg),
        "icc_identity_error": icc_total - icc_hit - icc_miss,
        "gcc_identity_error": gcc_total - gcc_hit - gcc_miss,
    }
    return {key: f"{value:.9g}" for key, value in values.items()}


def ncu_command(
    args: argparse.Namespace,
    binary: Path,
    repetitions: int,
    metrics: Iterable[str],
) -> tuple[list[str], int]:
    warmup_launches = 1 if args.mode == "warm" else 0
    command = [
        args.ncu,
        "--csv",
        "--page",
        "raw",
        "--print-units",
        "base",
        "--clock-control",
        args.clock_control,
        "--target-processes",
        "all",
        "--launch-count",
        "1",
        "--metrics",
        ",".join(metrics),
    ]
    if args.mode == "warm":
        command.extend(
            ["--replay-mode", "application", "--cache-control", "none"]
        )
        command.extend(["--launch-skip", str(warmup_launches)])
    else:
        command.extend(["--replay-mode", "kernel", "--cache-control", "all"])
    command.extend(
        [
            str(binary),
            f"--device={args.device}",
            f"--blocks={args.blocks}",
            f"--repetitions={repetitions}",
            f"--warmup-launches={warmup_launches}",
        ]
    )
    return command, warmup_launches


def collect(args: argparse.Namespace) -> list[dict[str, str]]:
    available = query_metrics(args.ncu)
    missing_required = sorted(REQUIRED_BASE_METRICS - available)
    if missing_required and not args.allow_missing:
        joined = ", ".join(missing_required)
        raise RuntimeError(
            f"NCU does not expose required ICC/GCC metrics: {joined}. "
            "Use a newer Nsight Compute or pass --allow-missing for diagnostics."
        )
    metrics = tuple(metric for metric in METRICS if metric_base(metric) in available)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output.parent / f"{args.output.stem}.raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for steps in args.steps:
        binary = args.bin_dir / f"{args.binary_prefix}{steps}"
        if not binary.is_file():
            raise FileNotFoundError(f"missing benchmark binary: {binary}")
        repetitions = (
            0
            if args.mode == "skip"
            else max(1, (args.dynamic_steps + steps - 1) // steps)
        )
        command, warmup_launches = ncu_command(args, binary, repetitions, metrics)
        print(f"[{args.mode}] steps={steps} repetitions={repetitions}", flush=True)
        result = run(command)
        raw_path = raw_dir / f"steps_{steps}.log"
        raw_path.write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(
                f"NCU failed for steps={steps}; inspect {raw_path}\n{result.stderr}"
            )
        profile = parse_ncu_csv(result.stdout)
        row = {
            "mode": args.mode,
            "steps": str(steps),
            "text_bytes": str(kernel_text_bytes(args.cuobjdump, binary)),
            "blocks": str(args.blocks),
            "repetitions": str(repetitions),
            "warmup_launches": str(warmup_launches),
            "device": profile.get("Device", ""),
            "kernel": profile.get("Kernel Name", ""),
        }
        row.update({metric: profile.get(metric, "") for metric in metrics})
        row.update(derived_values(profile))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    metric_columns = [metric for metric in METRICS if any(metric in row for row in rows)]
    columns = [*IDENTITY_COLUMNS, *metric_columns, *DERIVED_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_bin_dir = script_dir.parents[2] / "build" / "bin" / "microbench" / "instruction_cache"
    default_output = script_dir.parents[2] / "run" / "microbench" / "instruction_cache" / "icc_gcc.csv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cold", "warm", "skip"), default="cold")
    parser.add_argument("--steps", type=parse_steps, default=DEFAULT_STEPS)
    parser.add_argument("--dynamic-steps", type=int, default=1 << 20)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--bin-dir", type=Path, default=default_bin_dir)
    parser.add_argument("--binary-prefix", default="icc_gcc_footprint_")
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--clock-control", choices=("base", "boost", "none"), default="base")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    if args.dynamic_steps <= 0 or args.blocks <= 0 or args.device < 0:
        parser.error("dynamic-steps/blocks must be positive and device non-negative")
    return args


def main() -> int:
    args = parse_args()
    try:
        rows = collect(args)
        write_csv(args.output, rows)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
