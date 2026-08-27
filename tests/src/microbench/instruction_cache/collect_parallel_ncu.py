#!/usr/bin/env python3
"""Sweep independent warp PCs to expose ICC request-resource pressure."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import collect_ncu as common


DEFAULT_STREAMS = (1, 2, 4, 8, 9, 10, 11, 12, 16, 20, 24, 28, 32)


def parse_streams(value: str) -> tuple[int, ...]:
    streams = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not streams or any(item < 1 or item > 32 for item in streams):
        raise argparse.ArgumentTypeError("streams must be comma-separated values in [1, 32]")
    return streams


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_binary = (
        script_dir.parents[2]
        / "build"
        / "bin"
        / "microbench"
        / "instruction_cache"
        / "icc_gcc_parallel_pc"
    )
    default_output = (
        script_dir.parents[2]
        / "run"
        / "microbench"
        / "instruction_cache"
        / "icc_parallel_pc.csv"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", type=parse_streams, default=DEFAULT_STREAMS)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--binary", type=Path, default=default_binary)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument(
        "--clock-control", choices=("base", "boost", "none"), default="base"
    )
    args = parser.parse_args()
    if args.repetitions <= 0 or args.device < 0:
        parser.error("repetitions must be positive and device non-negative")
    return args


def main() -> int:
    args = parse_args()
    if not args.binary.is_file():
        print(f"error: missing benchmark binary: {args.binary}", file=sys.stderr)
        return 1
    try:
        available = common.query_metrics(args.ncu)
        missing = sorted(common.REQUIRED_BASE_METRICS - available)
        if missing:
            raise RuntimeError(
                "NCU does not expose required ICC/GCC metrics: " + ", ".join(missing)
            )
        metrics = tuple(
            metric
            for metric in common.METRICS
            if common.metric_base(metric) in available
        )
        rows = []
        raw_dir = args.output.parent / f"{args.output.stem}.raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for streams in args.streams:
            print(f"[parallel-pc] streams={streams}", flush=True)
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
                "--replay-mode",
                "kernel",
                "--cache-control",
                "all",
                "--metrics",
                ",".join(metrics),
                str(args.binary),
                f"--device={args.device}",
                f"--streams={streams}",
                f"--repetitions={args.repetitions}",
            ]
            result = common.run(command)
            raw_path = raw_dir / f"streams_{streams}.log"
            raw_path.write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(
                    f"NCU failed for streams={streams}; inspect {raw_path}"
                )
            profile = common.parse_ncu_csv(result.stdout)
            row = {
                "streams": str(streams),
                "repetitions": str(args.repetitions),
                "device": profile.get("Device", ""),
                "kernel": profile.get("Kernel Name", ""),
            }
            row.update({metric: profile.get(metric, "") for metric in metrics})
            row.update(common.derived_values(profile))
            rows.append(row)
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["streams", "repetitions", "device", "kernel", *metrics]
    columns.extend(common.DERIVED_COLUMNS)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
