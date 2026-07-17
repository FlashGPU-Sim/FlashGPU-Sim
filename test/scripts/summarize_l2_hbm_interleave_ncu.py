#!/usr/bin/env python3
import argparse
import csv
import math
import re
from pathlib import Path


DEFAULT_METRICS = [
    "lts__t_sectors_srcunit_tex_lookup_miss",
    "lts__t_sectors_srcunit_tex_lookup_hit",
    "lts__t_sectors_srcunit_tex",
    "lts__t_sectors",
    "fbpa__dram_read_sectors",
    "fbpa__dram_read_bytes",
    "fbpa__dram_sectors",
    "dram__bytes_op_read",
    "dram__sectors_op_read",
    "dram__sectors_read",
]


def parse_metric_value(text):
    text = (text or "").strip()
    if not text:
        return None, []
    match = re.match(r"^([-+0-9.eE]+)(?:\s*\((.*)\))?$", text)
    if not match:
        return None, []
    aggregate = float(match.group(1))
    values_text = match.group(2)
    if not values_text:
        return aggregate, []
    values = []
    for item in values_text.split(";"):
        item = item.strip()
        if item:
            values.append(float(item))
    return aggregate, values


def summarize_values(values):
    if not values:
        return {
            "inst_count": 0,
            "inst_min": "",
            "inst_mean": "",
            "inst_max": "",
            "inst_max_min_ratio": "",
            "inst_cv": "",
        }
    total = sum(values)
    mean = total / len(values)
    var = sum((x - mean) * (x - mean) for x in values) / len(values)
    min_v = min(values)
    max_v = max(values)
    return {
        "inst_count": len(values),
        "inst_min": f"{min_v:.6g}",
        "inst_mean": f"{mean:.6g}",
        "inst_max": f"{max_v:.6g}",
        "inst_max_min_ratio": f"{(max_v / min_v):.6g}" if min_v else "",
        "inst_cv": f"{(math.sqrt(var) / mean):.6g}" if mean else "",
    }


def pick_columns(fieldnames, metric_base):
    cols = [c for c in fieldnames if c == metric_base or c.startswith(metric_base + ".")]
    preferred_suffixes = [
        ".sum",
        ".avg",
        ".max",
        ".min",
        "",
    ]
    ordered = []
    for suffix in preferred_suffixes:
        for c in cols:
            if c == metric_base + suffix:
                ordered.append(c)
    ordered.extend(c for c in cols if c not in ordered)
    return ordered


def summarize_file(path, metric_bases):
    rows = list(csv.DictReader(path.open(newline="")))
    if len(rows) < 2:
        return []
    row = rows[-1]
    case = path.name
    if case.endswith(".raw.instances.csv"):
        case = case[: -len(".raw.instances.csv")]
    elif case.endswith(".raw.csv"):
        case = case[: -len(".raw.csv")]

    out_rows = []
    for base in metric_bases:
        for col in pick_columns(row.keys(), base):
            aggregate, values = parse_metric_value(row.get(col, ""))
            if aggregate is None:
                continue
            summary = summarize_values(values)
            out = {
                "case": case,
                "metric": col,
                "aggregate": f"{aggregate:.12g}",
            }
            out.update(summary)
            out_rows.append(out)
    return out_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="raw CSV files or directories")
    parser.add_argument("--metric", action="append", dest="metrics")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    metric_bases = args.metrics or DEFAULT_METRICS
    files = []
    for item in args.paths:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.raw.instances.csv")))
            files.extend(sorted(path.rglob("*.raw.csv")))
        else:
            files.append(path)
    seen = set()
    unique_files = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            unique_files.append(path)
            seen.add(resolved)

    rows = []
    for path in unique_files:
        rows.extend(summarize_file(path, metric_bases))

    fields = [
        "case",
        "metric",
        "aggregate",
        "inst_count",
        "inst_min",
        "inst_mean",
        "inst_max",
        "inst_max_min_ratio",
        "inst_cv",
    ]
    if args.out:
        out_f = open(args.out, "w", newline="")
    else:
        out_f = None
    try:
        target = out_f if out_f is not None else __import__("sys").stdout
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if out_f is not None:
            out_f.close()


if __name__ == "__main__":
    main()
