#!/usr/bin/env python3
"""Extract and compare memory traffic metrics from GPGPU-Sim logs and NCU reports.

Usage:
    python3 extract_metrics.py test_tma_gemm          # auto-resolves to triton_kernel_tracking/test_tma_gemm/results
    python3 extract_metrics.py path/to/results_dir    # explicit path with sim-log/ and ncu-rep/ inside

The results_dir must contain:
    sim-log/  — GPGPU-Sim log files (*_gpgpusim.log)
    ncu-rep/  — NCU report files (*.ncu-rep)

Output: metrics_comparison.csv in results_dir
"""

import argparse
import csv
import io
import os
import re
import subprocess
import sys


def parse_sim_log(path):
    """Extract metrics from a GPGPU-Sim log file."""
    with open(path) as f:
        content = f.read()

    metrics = {}

    # L2 cache
    m = re.search(r"L2_total_cache_accesses\s*=\s*(\d+)", content)
    metrics["l2_accesses"] = int(m.group(1)) if m else None

    m = re.search(r"L2_total_cache_misses\s*=\s*(\d+)", content)
    metrics["l2_misses"] = int(m.group(1)) if m else None

    m = re.search(r"L2_total_cache_miss_rate\s*=\s*([\d.]+)", content)
    metrics["l2_miss_rate"] = float(m.group(1)) if m else None

    # L2 traffic (sectors) and hit rate
    if metrics["l2_accesses"] is not None:
        metrics["l2_traffic_mb"] = metrics["l2_accesses"] * 32 / (1024 * 1024)
    else:
        metrics["l2_traffic_mb"] = None

    if metrics["l2_miss_rate"] is not None:
        metrics["l2_hit_rate_pct"] = (1.0 - metrics["l2_miss_rate"]) * 100
    else:
        metrics["l2_hit_rate_pct"] = None

    # DRAM: sum n_rd and n_write+n_wr_bk across all memory controllers
    # n_rd, n_write, n_wr_bk count DRAM column commands (not sectors).
    # Read commands: each serves a 32B sector request (1 command per sector).
    # Write commands: n_write serves per-sector requests (32B each);
    #   n_wr_bk serves L2 write-back requests (full cache lines, 128B each,
    #   requiring 2 commands per line at dram_atom_size=64B).
    rd_values = re.findall(r"\bn_rd=(\d+)\b", content)
    wr_values = re.findall(r"\bn_write=(\d+)\b", content)
    wr_wb_values = re.findall(r"\bn_wr_bk=(\d+)\b", content)
    metrics["dram_read_mb"] = sum(int(v) for v in rd_values) * 32 / (1024 * 1024) if rd_values else None
    # n_write: per-sector writes (32B each), n_wr_bk: line write-backs (64B per command)
    total_wr_bytes = (sum(int(v) for v in wr_values) * 32 if wr_values else 0) + \
                     (sum(int(v) for v in wr_wb_values) * 64 if wr_wb_values else 0)
    metrics["dram_write_mb"] = total_wr_bytes / (1024 * 1024) if (wr_values or wr_wb_values) else None

    # L1D cache: aggregate across cores
    # Format: L1D_cache_core[0]: Access = 0, Miss = 0, ...
    l1_accesses = re.findall(r"L1D_cache_core\[\d+\]:\s*Access\s*=\s*(\d+)", content)
    l1_misses = re.findall(r"L1D_cache_core\[\d+\]:\s*Access\s*=\s*\d+,\s*Miss\s*=\s*(\d+)", content)
    total_l1_access = sum(int(v) for v in l1_accesses)
    total_l1_miss = sum(int(v) for v in l1_misses)
    if total_l1_access > 0:
        metrics["l1_hit_rate_pct"] = (1.0 - total_l1_miss / total_l1_access) * 100
    else:
        metrics["l1_hit_rate_pct"] = None  # N/A (e.g. TMA bypasses L1)

    # Occupancy
    m = re.search(r"gpu_occupancy\s*=\s*([\d.]+)%", content)
    metrics["occupancy_pct"] = float(m.group(1)) if m else None

    return metrics


def parse_ncu_rep(path):
    """Extract metrics from an NCU .ncu-rep file via ncu --csv."""
    try:
        result = subprocess.run(
            ["ncu", "--import", path, "--csv", "--page", "raw"],
            capture_output=True, text=True, timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    reader = csv.reader(io.StringIO(result.stdout))
    rows = list(reader)
    if len(rows) < 2:
        return None

    # rows[0] is header, rows[1] is units, rows[2+] is data
    header = rows[0]
    units = rows[1] if len(rows) >= 3 else None
    data = rows[-1]

    def get_val(metric_name):
        try:
            idx = header.index(metric_name)
            v = data[idx].strip().strip('"')
            if v == "" or v == "n/a":
                return None
            v = v.replace(",", "")
            return float(v)
        except (ValueError, IndexError):
            return None

    def get_unit(metric_name):
        if units is None:
            return None
        try:
            idx = header.index(metric_name)
            return units[idx].strip().strip('"')
        except (ValueError, IndexError):
            return None

    def to_mb(val, unit):
        """Convert NCU value to MB based on its unit string."""
        if val is None:
            return None
        u = (unit or "").lower()
        if u in ("mbyte", "mb"):
            return val
        elif u in ("kbyte", "kb"):
            return val / 1024
        elif u in ("gbyte", "gb"):
            return val * 1024
        elif u in ("byte", "b", ""):
            return val / (1024 * 1024)
        return val  # unknown unit, assume raw bytes

    metrics = {}

    # DRAM traffic
    dram_read = get_val("dram__bytes_read.sum")
    dram_write = get_val("dram__bytes_write.sum")
    metrics["dram_read_mb"] = to_mb(dram_read, get_unit("dram__bytes_read.sum"))
    metrics["dram_write_mb"] = to_mb(dram_write, get_unit("dram__bytes_write.sum"))

    # L2 traffic (sectors — unit-aware)
    l2_sectors = get_val("lts__t_sectors.sum")
    l2_unit = get_unit("lts__t_sectors.sum")
    if l2_sectors is not None:
        if l2_unit and "sector" in l2_unit.lower():
            metrics["l2_traffic_mb"] = l2_sectors * 32 / (1024 * 1024)
        else:
            # unitless or count — assume sectors
            metrics["l2_traffic_mb"] = l2_sectors * 32 / (1024 * 1024)
    else:
        metrics["l2_traffic_mb"] = None

    # L2 hit rate
    metrics["l2_hit_rate_pct"] = get_val("lts__t_sector_hit_rate.pct")

    # L1 hit rate
    metrics["l1_hit_rate_pct"] = get_val("l1tex__t_sector_hit_rate.pct")

    # Occupancy
    metrics["occupancy_pct"] = get_val("sm__warps_active.avg.pct_of_peak_sustained_active")

    return metrics


def workload_name_from_sim_log(filename):
    """Extract workload name: m512_n6000_k2048_gpgpusim.log -> m512_n6000_k2048"""
    return re.sub(r"_gpgpusim\.log$", "", filename)


def workload_name_from_ncu_rep(filename):
    """Extract workload name: m512_n6000_k2048.ncu-rep -> m512_n6000_k2048"""
    return re.sub(r"\.ncu-rep$", "", filename)


def fmt(val, decimals=2):
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def resolve_results_dir(user_arg):
    """Resolve a user argument to a results directory.

    Tries in order:
      1. user_arg as-is (explicit path)
      2. triton_kernel_tracking/<user_arg>/results  (relative to triton_trace dir)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    triton_trace_dir = os.path.dirname(script_dir)

    # 1. Direct path
    if os.path.isdir(os.path.join(user_arg, "sim-log")) or os.path.isdir(os.path.join(user_arg, "ncu-rep")):
        return user_arg

    # 2. Workload name shorthand
    candidate = os.path.join(triton_trace_dir, "triton_kernel_tracking", user_arg, "results")
    if os.path.isdir(candidate):
        return candidate

    # Fall back to user_arg and let validation catch it
    return user_arg


def main():
    parser = argparse.ArgumentParser(description="Extract sim vs NCU metrics comparison")
    parser.add_argument("workload", help="Workload name (e.g. test_tma_gemm) or explicit path to results dir")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path (default: <results_dir>/metrics_comparison.csv)")
    args = parser.parse_args()

    results_dir = resolve_results_dir(args.workload)
    sim_dir = os.path.join(results_dir, "sim-log")
    ncu_dir = os.path.join(results_dir, "ncu-rep")

    if not os.path.isdir(sim_dir) and not os.path.isdir(ncu_dir):
        print(f"Error: cannot find sim-log/ or ncu-rep/ in: {results_dir}", file=sys.stderr)
        print(f"  (resolved from argument: {args.workload})", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(sim_dir):
        print(f"Warning: sim-log directory not found: {sim_dir}", file=sys.stderr)
    if not os.path.isdir(ncu_dir):
        print(f"Warning: ncu-rep directory not found: {ncu_dir}", file=sys.stderr)

    # Discover workloads
    sim_logs = {workload_name_from_sim_log(f): os.path.join(sim_dir, f)
                for f in os.listdir(sim_dir) if f.endswith("_gpgpusim.log")} if os.path.isdir(sim_dir) else {}
    ncu_reps = {workload_name_from_ncu_rep(f): os.path.join(ncu_dir, f)
                for f in os.listdir(ncu_dir) if f.endswith(".ncu-rep")} if os.path.isdir(ncu_dir) else {}

    all_workloads = sorted(set(sim_logs.keys()) | set(ncu_reps.keys()))
    if not all_workloads:
        print("No workloads found.", file=sys.stderr)
        sys.exit(1)

    # Output CSV
    out_path = args.output or os.path.join(results_dir, "metrics_comparison.csv")

    columns = [
        "workload",
        "sim_l2_traffic_mb", "ncu_l2_traffic_mb",
        "sim_dram_read_mb", "ncu_dram_read_mb",
        "sim_dram_write_mb", "ncu_dram_write_mb",
        "sim_l1_hit_rate_pct", "ncu_l1_hit_rate_pct",
        "sim_l2_hit_rate_pct", "ncu_l2_hit_rate_pct",
        "sim_occupancy_pct", "ncu_occupancy_pct",
    ]

    rows = []
    for wl in all_workloads:
        sim = parse_sim_log(sim_logs[wl]) if wl in sim_logs else {}
        ncu = parse_ncu_rep(ncu_reps[wl]) if wl in ncu_reps else {}
        if ncu is None:
            ncu = {}

        row = {
            "workload": wl,
            "sim_l2_traffic_mb": fmt(sim.get("l2_traffic_mb")),
            "ncu_l2_traffic_mb": fmt(ncu.get("l2_traffic_mb")),
            "sim_dram_read_mb": fmt(sim.get("dram_read_mb")),
            "ncu_dram_read_mb": fmt(ncu.get("dram_read_mb")),
            "sim_dram_write_mb": fmt(sim.get("dram_write_mb")),
            "ncu_dram_write_mb": fmt(ncu.get("dram_write_mb")),
            "sim_l1_hit_rate_pct": fmt(sim.get("l1_hit_rate_pct")),
            "ncu_l1_hit_rate_pct": fmt(ncu.get("l1_hit_rate_pct")),
            "sim_l2_hit_rate_pct": fmt(sim.get("l2_hit_rate_pct")),
            "ncu_l2_hit_rate_pct": fmt(ncu.get("l2_hit_rate_pct")),
            "sim_occupancy_pct": fmt(sim.get("occupancy_pct")),
            "ncu_occupancy_pct": fmt(ncu.get("occupancy_pct")),
        }
        rows.append(row)

    # Write CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    # Print table to stdout
    print(f"\n{'Workload':<25} {'L2 Traffic (MB)':>18} {'DRAM Rd (MB)':>15} {'DRAM Wr (MB)':>15} {'L1 Hit%':>16} {'L2 Hit%':>16} {'Occupancy%':>16}")
    print(f"{'':25} {'sim / ncu':>18} {'sim / ncu':>15} {'sim / ncu':>15} {'sim / ncu':>16} {'sim / ncu':>16} {'sim / ncu':>16}")
    print("-" * 126)
    for r in rows:
        l2t = f"{r['sim_l2_traffic_mb']:>7} / {r['ncu_l2_traffic_mb']:>7}"
        dr = f"{r['sim_dram_read_mb']:>5} / {r['ncu_dram_read_mb']:>5}"
        dw = f"{r['sim_dram_write_mb']:>5} / {r['ncu_dram_write_mb']:>5}"
        l1h = f"{r['sim_l1_hit_rate_pct']:>6} / {r['ncu_l1_hit_rate_pct']:>6}"
        l2h = f"{r['sim_l2_hit_rate_pct']:>6} / {r['ncu_l2_hit_rate_pct']:>6}"
        occ = f"{r['sim_occupancy_pct']:>6} / {r['ncu_occupancy_pct']:>6}"
        print(f"{r['workload']:<25} {l2t:>18} {dr:>15} {dw:>15} {l1h:>16} {l2h:>16} {occ:>16}")

    print(f"\nCSV written to: {out_path}")


if __name__ == "__main__":
    main()
