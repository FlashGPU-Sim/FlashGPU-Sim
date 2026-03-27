#!/usr/bin/env python3
"""Compare cycle counts between NCU profiling and GPGPU-Sim simulation.

Reads summary.csv from ncu-rep/ and sim-log/ under each test's results/
directory and outputs a comparison table.

Usage:
    python3 compare_cycles.py [test_name]          # specific test (default: test_tma_gemm)
    python3 compare_cycles.py --all                 # all tests with results/
"""

import argparse
import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKING_DIR = os.path.join(SCRIPT_DIR, "triton_kernel_tracking")


def read_csv_as_dict(path, key_col, val_col):
    """Read a CSV and return {key_col_value: val_col_value} mapping."""
    result = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k = row[key_col].strip()
            v = row[val_col].strip()
            result[k] = float(v)
    return result


def compare_one_test(test_name):
    results_dir = os.path.join(TRACKING_DIR, test_name, "results")
    ncu_csv = os.path.join(results_dir, "ncu-rep", "summary.csv")
    sim_csv = os.path.join(results_dir, "sim-log", "summary.csv")

    if not os.path.isfile(ncu_csv):
        print(f"[WARN] {test_name}: ncu-rep/summary.csv not found, skipping")
        return False
    if not os.path.isfile(sim_csv):
        print(f"[WARN] {test_name}: sim-log/summary.csv not found, skipping")
        return False

    ncu_data = read_csv_as_dict(ncu_csv, "size", "sm_cycles")
    sim_data = read_csv_as_dict(sim_csv, "size", "sim_cycles")

    all_sizes = sorted(set(ncu_data.keys()) | set(sim_data.keys()), key=lambda s: int(s))

    print(f"\n{'='*60}")
    print(f"  {test_name}")
    print(f"{'='*60}")
    print(f"{'size':>8s} {'sim_cycle':>12s} {'ncu_cycle':>12s} {'diff':>10s} {'diff%':>8s}")
    print(f"{'-'*8:>8s} {'-'*12:>12s} {'-'*12:>12s} {'-'*10:>10s} {'-'*8:>8s}")

    for size in all_sizes:
        sim_c = sim_data.get(size)
        ncu_c = ncu_data.get(size)

        sim_str = f"{sim_c:,.0f}" if sim_c is not None else "N/A"
        ncu_str = f"{ncu_c:,.1f}" if ncu_c is not None else "N/A"

        if sim_c is not None and ncu_c is not None:
            diff = sim_c - ncu_c
            pct = (diff / ncu_c) * 100.0
            diff_str = f"{diff:+,.0f}"
            pct_str = f"{pct:+.1f}%"
        else:
            diff_str = "N/A"
            pct_str = "N/A"

        print(f"{size:>8s} {sim_str:>12s} {ncu_str:>12s} {diff_str:>10s} {pct_str:>8s}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Compare NCU vs Sim cycle counts")
    parser.add_argument("test_name", nargs="?", default=None,
                        help="Test name under triton_kernel_tracking/ (default: test_tma_gemm)")
    parser.add_argument("--all", action="store_true",
                        help="Process all tests that have a results/ directory")
    args = parser.parse_args()

    if args.all:
        found = False
        for name in sorted(os.listdir(TRACKING_DIR)):
            results_dir = os.path.join(TRACKING_DIR, name, "results")
            if os.path.isdir(results_dir):
                compare_one_test(name)
                found = True
        if not found:
            print("No tests with results/ directory found.")
            sys.exit(1)
    else:
        test_name = args.test_name or "test_tma_gemm"
        if not compare_one_test(test_name):
            sys.exit(1)


if __name__ == "__main__":
    main()
