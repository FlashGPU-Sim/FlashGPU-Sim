#!/usr/bin/env python3
# Source: https://github.com/seanzw/random
# commit 4e8c4f91dd7b00584efcb3ac4b602b33ce2631cd
# directory dsm_bw
# Blog: https://seanzw.github.io/posts/gpu_dsm_bw/
"""Summarize the standalone Hopper DSM benchmark without third-party modules."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


GROUP_FIELDS = (
    "label",
    "method",
    "pattern",
    "cluster_size",
    "param",
    "active_warps",
    "chunk_bytes",
    "pressure_mode",
    "pressure_warps",
    "threads",
    "epoch_bytes_per_active_sm",
    "mixed_kind",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("-o", "--output-dir", required=True, type=Path)
    return parser.parse_args()


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def load_records(path: Path) -> tuple[dict, list[dict], list[dict]]:
    meta: dict = {}
    results: list[dict] = []
    skips: list[dict] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {error}")
            if record.get("type") == "meta":
                meta = record
            elif record.get("type") == "result":
                results.append(record)
            elif record.get("type") == "skip":
                skips.append(record)
    if not results:
        raise SystemExit(f"no result records in {path}")
    return meta, results, skips


def summarize(records: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    defaults = {"mixed_kind": -1, "pressure_mode": 0, "pressure_warps": 0}
    for record in records:
        groups[
            tuple(record.get(field, defaults.get(field, 0)) for field in GROUP_FIELDS)
        ].append(record)

    rows = []
    for key, samples in groups.items():
        row = dict(zip(GROUP_FIELDS, key))
        rates = [float(sample["aggregate_gbps"]) for sample in samples]
        cycle_rates = [
            float(sample["aggregate_bytes_per_cycle"]) for sample in samples
        ]
        cycles = [float(sample["max_cycles"]) for sample in samples]
        nanoseconds = [float(sample["elapsed_ns"]) for sample in samples]
        row.update(
            samples=len(samples),
            median_gbps=median(rates),
            min_gbps=min(rates),
            max_gbps=max(rates),
            median_aggregate_bytes_per_cycle=median(cycle_rates),
            median_cycles=median(cycles),
            median_ns=median(nanoseconds),
            median_cycles_per_ns=median(
                [cycle / ns for cycle, ns in zip(cycles, nanoseconds)]
            ),
        )
        rows.append(row)
    return sorted(rows, key=lambda row: tuple(str(row[field]) for field in GROUP_FIELDS))


def linear_fit(x: list[float], y: list[float]) -> tuple[float, float, float]:
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    if denominator == 0:
        raise ValueError("fit requires at least two distinct x values")
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / denominator
    intercept = y_mean - slope * x_mean
    rmse = math.sqrt(
        statistics.mean(
            (actual - (intercept + slope * value)) ** 2
            for actual, value in zip(y, x)
        )
    )
    return intercept, slope, rmse


def fit_rows(summary: list[dict]) -> list[dict]:
    labels = {
        "load_size_uni": 1,
        "load_size_bi": 2,
        "store_size_uni": 1,
        "store_size_bi": 2,
        "tma_size_uni": 1,
        "tma_size_bi": 2,
        "mixed_load_store_same": 2,
        "mixed_load_store_opposite": 2,
        "mixed_load_tma_same": 2,
        "mixed_load_tma_opposite": 2,
        "tma_scale_n2": 2,
        "tma_scale_n4": 4,
        "tma_scale_n8": 8,
        "tma_scale_n16": 16,
    }
    rows = []
    for label, active_sms in labels.items():
        samples = [row for row in summary if row["label"] == label]
        if len(samples) < 2:
            continue
        x = [float(row["epoch_bytes_per_active_sm"]) for row in samples]
        cycle_intercept, cycle_slope, cycle_rmse = linear_fit(
            x, [float(row["median_cycles"]) for row in samples]
        )
        ns_intercept, ns_slope, ns_rmse = linear_fit(
            x, [float(row["median_ns"]) for row in samples]
        )
        per_sm_bpc = 1.0 / cycle_slope
        per_sm_gbps = 1.0 / ns_slope
        rows.append(
            {
                "label": label,
                "samples": len(samples),
                "active_sms": active_sms,
                "min_payload_bytes_per_sm": min(x),
                "max_payload_bytes_per_sm": max(x),
                "intercept_cycles": cycle_intercept,
                "slope_cycles_per_byte": cycle_slope,
                "bytes_per_cycle_per_sm_or_payload": per_sm_bpc,
                "aggregate_bytes_per_cycle": active_sms * per_sm_bpc,
                "rmse_cycles": cycle_rmse,
                "intercept_ns": ns_intercept,
                "slope_ns_per_byte": ns_slope,
                "gbps_per_sm_or_payload": per_sm_gbps,
                "aggregate_gbps": active_sms * per_sm_gbps,
                "rmse_ns": ns_rmse,
                "slope_clock_cycles_per_ns": cycle_slope / ns_slope,
            }
        )
    return rows


def topology_rows(summary: list[dict]) -> list[dict]:
    patterns = (
        (re.compile(r"^load_ring_n(\d+)_s(\d+)$"), "ring"),
        (re.compile(r"^load_packed_g(\d+)_s(\d+)$"), "packed"),
    )
    rows = []
    for sample in summary:
        for pattern, family in patterns:
            match = pattern.fullmatch(str(sample["label"]))
            if not match:
                continue
            rows.append(
                {
                    "label": sample["label"],
                    "family": family,
                    "cluster_size": int(sample["cluster_size"]),
                    "group_or_ring_size": int(match.group(1)),
                    "stride": int(match.group(2)),
                    "active_sms": int(sample["cluster_size"]),
                    "median_aggregate_gbps": float(sample["median_gbps"]),
                    "median_per_sm_gbps": float(sample["median_gbps"])
                    / int(sample["cluster_size"]),
                    "median_aggregate_bytes_per_cycle": float(
                        sample["median_aggregate_bytes_per_cycle"]
                    ),
                    "median_cycles": float(sample["median_cycles"]),
                    "median_ns": float(sample["median_ns"]),
                }
            )
            break
    return sorted(
        rows,
        key=lambda row: (
            row["family"],
            row["group_or_ring_size"],
            row["stride"],
        ),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta, records, skips = load_records(args.input)
    summary = summarize(records)
    fits = fit_rows(summary)
    topology = topology_rows(summary)
    write_csv(args.output_dir / "summary.csv", summary)
    write_csv(args.output_dir / "fits.csv", fits)
    write_csv(args.output_dir / "topology.csv", topology)
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if skips:
        (args.output_dir / "skips.json").write_text(
            json.dumps(skips, indent=2) + "\n"
        )
    print(f"processed {len(records)} samples into {args.output_dir}")


if __name__ == "__main__":
    main()
