#!/usr/bin/env python3
"""Extract commonly used statistics from FlashGPU-Sim run logs.

By default, the final statistics report from each log is printed. Use --all to
show every report, or --csv to save the selected reports in a machine-readable
form.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


INTEGER_METRICS = {
    "gpu_sim_cycle",
    "gpu_sim_insn",
    "gpu_tot_sim_cycle",
    "gpu_tot_sim_insn",
    "gpu_tot_issued_cta",
    "gpu_total_sim_rate",
    "L1D_total_cache_accesses",
    "L1D_total_cache_misses",
    "L2_total_cache_accesses",
    "L2_total_cache_misses",
}

FLOAT_METRICS = {
    "gpu_ipc",
    "gpu_tot_ipc",
    "gpu_occupancy",
    "gpu_tot_occupancy",
    "L1D_total_cache_miss_rate",
    "L2_total_cache_miss_rate",
    "L2_BW",
    "L2_BW_total",
}

DISPLAY_METRICS = (
    "gpu_sim_cycle",
    "gpu_sim_insn",
    "gpu_ipc",
    "gpu_tot_sim_cycle",
    "gpu_tot_sim_insn",
    "gpu_tot_ipc",
    "gpu_occupancy",
    "gpu_tot_occupancy",
    "L1D_total_cache_accesses",
    "L1D_total_cache_misses",
    "L1D_total_cache_miss_rate",
    "L2_total_cache_accesses",
    "L2_total_cache_misses",
    "L2_total_cache_miss_rate",
    "L2_BW",
    "L2_BW_total",
    "dram_reads",
    "dram_writes",
    "dram_writebacks",
    "gpu_total_sim_rate",
)

CSV_FIELDS = (
    "log",
    "report",
    "kernel_name",
    "kernel_launch_uid",
    "kernel_stream_id",
    *DISPLAY_METRICS,
)

KEY_VALUE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
NUMBER_RE = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
)
DRAM_RE = re.compile(
    r"\bn_rd=(\d+)\b.*?\bn_write=(\d+)\b(?:.*?\bn_wr_bk=(\d+)\b)?"
)


@dataclass
class StatsReport:
    kernel_name: str = ""
    kernel_launch_uid: str = ""
    kernel_stream_id: str = ""
    metrics: dict[str, int | float] = field(default_factory=dict)
    dram_reads: int = 0
    dram_writes: int = 0
    dram_writebacks: int = 0
    dram_lines: int = 0

    def finish(self) -> None:
        if self.dram_lines:
            self.metrics["dram_reads"] = self.dram_reads
            self.metrics["dram_writes"] = self.dram_writes
            self.metrics["dram_writebacks"] = self.dram_writebacks


def parse_number(value: str, integer: bool) -> int | float | None:
    match = NUMBER_RE.match(value)
    if match is None:
        return None
    try:
        return int(match.group(0)) if integer else float(match.group(0))
    except ValueError:
        return None


def parse_log_text(text: str) -> list[StatsReport]:
    reports: list[StatsReport] = []
    pending_kernel_name = ""
    pending_kernel_launch_uid = ""
    pending_kernel_stream_id = ""
    current: StatsReport | None = None

    for line in text.splitlines():
        key_value = KEY_VALUE_RE.match(line)
        if key_value:
            key, value = key_value.groups()

            if key == "kernel_name":
                pending_kernel_name = value.strip()
                continue
            if key == "kernel_launch_uid":
                pending_kernel_launch_uid = value.strip()
                continue
            if key == "kernel_stream_id":
                pending_kernel_stream_id = value.strip()
                continue

            if key == "gpu_sim_cycle":
                if current is not None:
                    current.finish()
                    reports.append(current)
                current = StatsReport(
                    kernel_name=pending_kernel_name,
                    kernel_launch_uid=pending_kernel_launch_uid,
                    kernel_stream_id=pending_kernel_stream_id,
                )
                pending_kernel_name = ""
                pending_kernel_launch_uid = ""
                pending_kernel_stream_id = ""

            if current is not None:
                if key in INTEGER_METRICS:
                    parsed = parse_number(value, integer=True)
                elif key in FLOAT_METRICS:
                    parsed = parse_number(value, integer=False)
                else:
                    parsed = None
                if parsed is not None:
                    current.metrics[key] = parsed

        if current is not None:
            dram = DRAM_RE.search(line)
            if dram:
                current.dram_reads += int(dram.group(1))
                current.dram_writes += int(dram.group(2))
                current.dram_writebacks += int(dram.group(3) or 0)
                current.dram_lines += 1

    if current is not None:
        current.finish()
        reports.append(current)

    return reports


def parse_log(path: Path) -> list[StatsReport]:
    return parse_log_text(path.read_text(encoding="utf-8", errors="replace"))


def format_value(key: str, value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    if key in {"gpu_occupancy", "gpu_tot_occupancy"}:
        return f"{value:.4f}%"
    return f"{value:.4f}"


def print_report(path: Path, report: StatsReport, index: int, total: int) -> None:
    print(f"Log: {path}")
    print(f"Report: {index}/{total}")
    if report.kernel_name:
        print(f"Kernel: {report.kernel_name}")
    if report.kernel_launch_uid:
        print(f"Launch UID: {report.kernel_launch_uid}")
    if report.kernel_stream_id:
        print(f"Stream ID: {report.kernel_stream_id}")
    print()

    width = max(len(key) for key in DISPLAY_METRICS)
    for key in DISPLAY_METRICS:
        value = report.metrics.get(key)
        if value is not None:
            print(f"{key:<{width}}  {format_value(key, value)}")


def csv_row(
    path: Path, report: StatsReport, index: int
) -> dict[str, str | int | float]:
    row: dict[str, str | int | float] = {
        "log": str(path),
        "report": index,
        "kernel_name": report.kernel_name,
        "kernel_launch_uid": report.kernel_launch_uid,
        "kernel_stream_id": report.kernel_stream_id,
    }
    for key in DISPLAY_METRICS:
        row[key] = report.metrics.get(key, "")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="simulation log files")
    parser.add_argument(
        "--all",
        action="store_true",
        help="show every statistics report instead of only the final report",
    )
    parser.add_argument("--csv", type=Path, help="write selected reports to CSV")
    args = parser.parse_args()

    selected: list[tuple[Path, StatsReport, int, int]] = []
    had_error = False
    for path in args.logs:
        try:
            reports = parse_log(path)
        except OSError as error:
            print(f"Error: cannot read {path}: {error}", file=sys.stderr)
            had_error = True
            continue

        if not reports:
            print(f"Error: no simulation statistics found in {path}", file=sys.stderr)
            had_error = True
            continue

        if args.all:
            selected.extend(
                (path, report, index, len(reports))
                for index, report in enumerate(reports, start=1)
            )
        else:
            selected.append((path, reports[-1], len(reports), len(reports)))

    for position, (path, report, index, total) in enumerate(selected):
        if position:
            print()
        print_report(path, report, index, total)

    if args.csv and selected:
        try:
            with args.csv.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(
                    csv_row(path, report, index)
                    for path, report, index, _ in selected
                )
        except OSError as error:
            print(f"Error: cannot write {args.csv}: {error}", file=sys.stderr)
            return 1
        print(f"\nCSV written to: {args.csv}")

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
