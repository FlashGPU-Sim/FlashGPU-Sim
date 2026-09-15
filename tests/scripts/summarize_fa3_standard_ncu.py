#!/usr/bin/env python3
"""Summarize per-kernel and per-GTest timing from FA3 NCU reports."""

from __future__ import annotations

import argparse
import csv
import pathlib
from collections import defaultdict


TIMING_METRICS = {
    "Elapsed Cycles",
    "SM Active Cycles",
    "Duration",
    "SM Frequency",
}


def parse_number(value: str) -> float | None:
    value = value.strip().replace(",", "")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def duration_us(value: str, unit: str) -> float | None:
    number = parse_number(value)
    if number is None:
        return None
    factors = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}
    factor = factors.get(unit.strip().lower())
    return None if factor is None else number * factor


def frequency_ghz(value: str, unit: str) -> float | None:
    number = parse_number(value)
    if number is None:
        return None
    factors = {"hz": 1e-9, "khz": 1e-6, "mhz": 1e-3, "ghz": 1.0}
    factor = factors.get(unit.strip().lower())
    return None if factor is None else number * factor


def clean_number(value: float | None) -> str:
    if value is None:
        return ""
    if value.is_integer():
        return str(int(value))
    return f"{value:.9g}"


def read_status(root: pathlib.Path) -> list[dict[str, str]]:
    path = root / "status.tsv"
    if not path.is_file():
        return []
    with path.open(newline="", errors="replace") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def read_kernel_rows(root: pathlib.Path) -> list[dict[str, str]]:
    result = []
    for path in sorted((root / "ncu").glob("*/details.csv")):
        run = path.parent.name
        kernels: dict[tuple[str, str], dict[str, str]] = {}
        metric_values: dict[tuple[str, str], dict[str, tuple[str, str]]] = defaultdict(dict)
        with path.open(newline="", errors="replace") as stream:
            for row in csv.DictReader(stream):
                kernel_id = row.get("ID", "")
                kernel_name = row.get("Kernel Name", "")
                if not kernel_id or not kernel_name:
                    continue
                key = (kernel_id, kernel_name)
                kernels.setdefault(
                    key,
                    {
                        "run": run,
                        "kernel_id": kernel_id,
                        "kernel": kernel_name,
                        "grid": row.get("Grid Size", ""),
                        "block": row.get("Block Size", ""),
                    },
                )
                metric = row.get("Metric Name", "")
                if metric in TIMING_METRICS:
                    metric_values[key][metric] = (
                        row.get("Metric Value", ""),
                        row.get("Metric Unit", ""),
                    )

        def kernel_sort(item: tuple[tuple[str, str], dict[str, str]]) -> tuple[int, str]:
            kernel_id = item[0][0]
            return (int(kernel_id) if kernel_id.isdigit() else 1 << 30, kernel_id)

        for key, kernel in sorted(kernels.items(), key=kernel_sort):
            values = metric_values[key]
            elapsed = parse_number(values.get("Elapsed Cycles", ("", ""))[0])
            sm_active = parse_number(values.get("SM Active Cycles", ("", ""))[0])
            duration = duration_us(*values.get("Duration", ("", "")))
            frequency = frequency_ghz(*values.get("SM Frequency", ("", "")))
            kernel.update(
                {
                    "elapsed_cycles": clean_number(elapsed),
                    "sm_active_cycles": clean_number(sm_active),
                    "duration_us": clean_number(duration),
                    "sm_frequency_ghz": clean_number(frequency),
                }
            )
            result.append(kernel)
    return result


def write_csv(path: pathlib.Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_cases(
    status_rows: list[dict[str, str]], kernel_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in kernel_rows:
        by_run[row["run"]].append(row)

    result = []
    for status in status_rows:
        profile = status["profile"]
        gtest = status["gtest"]
        run = f"{profile}__{status['case_id']}"
        kernels = by_run.get(run, [])
        # Older collections may contain metadata-only ShapeTableHas* GTests.
        # They validate the host-side case catalog but launch no GPU kernel, so
        # they are not end-to-end timing cases and must not appear in the cycle
        # regression table.
        if not kernels:
            continue
        elapsed = [parse_number(row["elapsed_cycles"]) for row in kernels]
        sm_active = [parse_number(row["sm_active_cycles"]) for row in kernels]
        durations = [parse_number(row["duration_us"]) for row in kernels]
        frequencies = [parse_number(row["sm_frequency_ghz"]) for row in kernels]
        elapsed = [value for value in elapsed if value is not None]
        sm_active = [value for value in sm_active if value is not None]
        durations = [value for value in durations if value is not None]
        frequencies = [value for value in frequencies if value is not None]
        if frequencies and durations and len(frequencies) == len(durations) and sum(durations):
            frequency = sum(f * d for f, d in zip(frequencies, durations)) / sum(durations)
        elif frequencies:
            frequency = sum(frequencies) / len(frequencies)
        else:
            frequency = None
        result.append(
            {
                "profile": profile,
                "gtest": gtest,
                "direction": "backward" if "Backward" in gtest else "forward",
                "kernel_count": str(len(kernels)),
                "end_to_end_elapsed_cycles": clean_number(sum(elapsed)) if elapsed else "",
                "sum_sm_active_cycles": clean_number(sum(sm_active)) if sm_active else "",
                "end_to_end_duration_us": clean_number(sum(durations)) if durations else "",
                "duration_weighted_sm_frequency_ghz": clean_number(frequency),
                "native_status": status["native_status"],
                "ncu_status": status["ncu_status"],
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=pathlib.Path)
    args = parser.parse_args()
    root = args.result_dir.resolve()

    status_rows = read_status(root)
    kernel_rows = read_kernel_rows(root)
    if not kernel_rows:
        raise SystemExit(f"no NCU details tables found below {root / 'ncu'}")

    kernel_fields = [
        "run",
        "kernel_id",
        "kernel",
        "grid",
        "block",
        "elapsed_cycles",
        "sm_active_cycles",
        "duration_us",
        "sm_frequency_ghz",
    ]
    case_fields = [
        "profile",
        "gtest",
        "direction",
        "kernel_count",
        "end_to_end_elapsed_cycles",
        "sum_sm_active_cycles",
        "end_to_end_duration_us",
        "duration_weighted_sm_frequency_ghz",
        "native_status",
        "ncu_status",
    ]
    case_rows = summarize_cases(status_rows, kernel_rows)
    write_csv(root / "cycles_summary.csv", kernel_fields, kernel_rows)
    write_csv(root / "case_cycles_summary.csv", case_fields, case_rows)
    print(
        f"summarized {len(kernel_rows)} kernels and {len(case_rows)} test cases "
        f"below {root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
