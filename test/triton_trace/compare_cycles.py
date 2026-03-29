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


def read_csv_as_dict(path, val_col):
    """Read a CSV and return {shape_key: val_col_value} mapping.

    The shape key is auto-detected from the header:
      - If 'size' column exists: key = size value
      - If 'm' column exists: key = "m,n,k" composite (GEMM)
      - If 'batch' column exists: key = "batch,nheads,seqlen,headdim,causal" (flash_attn CSV mode)
      - If 'seq_len' column exists: key = seq_len value (flash_attn legacy)
      - Otherwise: key = first column value
    """
    result = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for row in reader:
            if "size" in header:
                k = row["size"].strip()
            elif "m" in header:
                k = f"{row['m'].strip()},{row['n'].strip()},{row['k'].strip()}"
            elif "batch" in header:
                k = (f"{row['batch'].strip()},{row['nheads'].strip()},"
                     f"{row['seqlen'].strip()},{row['headdim'].strip()},"
                     f"{row['causal'].strip()}")
            elif "seq_len" in header:
                k = row["seq_len"].strip()
            else:
                k = row[header[0]].strip()
            v = row[val_col].strip()
            try:
                result[k] = float(v)
            except ValueError:
                pass  # skip N/A entries
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

    ncu_data = read_csv_as_dict(ncu_csv, "sm_cycles")
    sim_data = read_csv_as_dict(sim_csv, "sim_cycles")

    all_keys = sorted(set(ncu_data.keys()) | set(sim_data.keys()))

    # Determine label width from longest key
    label_w = max(8, max((len(k) for k in all_keys), default=8))

    print(f"\n{'='*60}")
    print(f"  {test_name}")
    print(f"{'='*60}")
    print(f"{'shape':>{label_w}s} {'sim_cycle':>12s} {'ncu_cycle':>12s} {'diff':>10s} {'diff%':>8s}")
    print(f"{'-'*label_w:>{label_w}s} {'-'*12:>12s} {'-'*12:>12s} {'-'*10:>10s} {'-'*8:>8s}")

    for key in all_keys:
        sim_c = sim_data.get(key)
        ncu_c = ncu_data.get(key)

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

        print(f"{key:>{label_w}s} {sim_str:>12s} {ncu_str:>12s} {diff_str:>10s} {pct_str:>8s}")

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
