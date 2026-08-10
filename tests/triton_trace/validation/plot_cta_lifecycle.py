#!/usr/bin/env python3
"""
Plot CTA lifecycle from GPGPU-Sim liveness trace logs.

Usage (same pattern as sweep_tests.sh):
  python3 plot_cta_lifecycle.py tma_gemm --shape 512,3000,2816
  python3 plot_cta_lifecycle.py tma_gemm --shape 512,3000,2560 --shape 512,3000,2816
  python3 plot_cta_lifecycle.py tma_gemm --csv configs/gemm_shapes_training.csv
  python3 plot_cta_lifecycle.py tma_gemm 128 512 1024   # square sizes
  python3 plot_cta_lifecycle.py flash_attn --shape 16,32,1024,64,True

Requires: liveness tracing enabled in gpgpusim.config:
  -trace_enabled 1
  -trace_components LIVENESS
  -trace_sampling_core -1
"""

import argparse
import re
import sys
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


SCRIPT_DIR = Path(__file__).parent.resolve()
TRITON_TRACE_DIR = SCRIPT_DIR.parent


def resolve_input_path(path):
    """Resolve input files relative to validation/ first, then triton_trace/."""
    p = Path(path)
    if p.is_absolute():
        return p
    candidate = SCRIPT_DIR / p
    if candidate.exists():
        return candidate
    return TRITON_TRACE_DIR / p


def parse_cta_lifecycle(logfile):
    """Parse LIVENESS trace and return list of (core, slot, start, end, seq)."""
    init_events = []
    finish_events = []

    with open(logfile) as f:
        for line in f:
            if "LIVENESS" not in line:
                continue
            m = re.search(r'Core (\d+).*cta: (\d+),.*initialized @\((\d+),', line)
            if m:
                init_events.append((int(m.group(3)), int(m.group(1)), int(m.group(2))))
                continue
            m = re.search(r'Core (\d+).*Finished CTA #(\d+) \((\d+),', line)
            if m:
                finish_events.append((int(m.group(3)), int(m.group(1)), int(m.group(2))))

    # Match inits to finishes per (core, slot)
    core_slot_inits = defaultdict(list)
    core_slot_finishes = defaultdict(list)
    for cyc, core, slot in sorted(init_events):
        core_slot_inits[(core, slot)].append(cyc)
    for cyc, core, slot in sorted(finish_events):
        core_slot_finishes[(core, slot)].append(cyc)

    # Build spans: (core, slot, start, end)
    raw_spans = []
    for key in sorted(core_slot_inits.keys()):
        inits = core_slot_inits[key]
        fins = core_slot_finishes.get(key, [])
        for i in range(min(len(inits), len(fins))):
            raw_spans.append((key[0], key[1], inits[i], fins[i]))

    # Assign sequence numbers by chronological start time per core
    raw_spans.sort(key=lambda s: (s[0], s[2]))  # sort by (core, start_cycle)
    core_cta_count = defaultdict(int)
    spans = []
    for core, slot, start, end in raw_spans:
        seq = core_cta_count[core]
        core_cta_count[core] += 1
        spans.append((core, slot, start, end, seq))

    return spans


def get_sim_cycles(logfile):
    """Extract gpu_tot_sim_cycle from log."""
    with open(logfile) as f:
        for line in f:
            m = re.match(r'gpu_tot_sim_cycle = (\d+)', line)
            if m:
                return int(m.group(1))
    return None


def plot_lifecycle(spans, sim_cycles, ncu_cycles, title, outfile):
    """Draw CTA lifecycle chart with dual slots per core."""
    if not spans:
        print(f"  No CTA lifecycle data for {title}")
        return

    n_cores_seen = len(set(c for c, _, _, _, _ in spans))
    all_cores_mode = n_cores_seen > 1

    # CTAs per core distribution
    ctas_per_core = defaultdict(int)
    for core, slot, start, end, seq in spans:
        ctas_per_core[core] += 1
    dist = defaultdict(int)
    for n in ctas_per_core.values():
        dist[n] += 1
    dist_str = " + ".join(f"{dist[n]}×{n}" for n in sorted(dist.keys()))

    if all_cores_mode:
        # Full chip: dual-slot view
        core_last = {}
        for core, slot, start, end, seq in spans:
            if core not in core_last or end > core_last[core]:
                core_last[core] = end
        cores_sorted = sorted(core_last.keys(), key=lambda c: core_last[c])
        core_to_idx = {c: i for i, c in enumerate(cores_sorted)}

        row_height = 0.4
        core_gap = 0.2
        fig_height = max(8, n_cores_seen * (2 * row_height + core_gap) * 0.08)
        fig, ax = plt.subplots(figsize=(16, fig_height))

        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0',
                  '#00BCD4', '#795548', '#607D8B', '#E91E63', '#CDDC39']
        for core, slot, start, end, seq in spans:
            idx = core_to_idx[core]
            y = idx * (2 * row_height + core_gap) + slot * row_height
            color = colors[seq % len(colors)]
            ax.barh(y, end - start, left=start, height=row_height * 0.9,
                    color=color, alpha=0.8, linewidth=0)

        # Y-axis labels
        ytick_pos = []
        ytick_lab = []
        for i, core in enumerate(cores_sorted):
            if i % 20 == 0 or i >= len(cores_sorted) - 2:
                y = i * (2 * row_height + core_gap) + row_height * 0.5
                ytick_pos.append(y)
                ytick_lab.append(f'Core {core}')
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_lab, fontsize=7)
        ax.set_ylabel('Core (sorted by finish time)', fontsize=11)

        max_ctas = max(dist.keys())
        patches = [mpatches.Patch(color=colors[i % len(colors)], label=f'CTA #{i+1}')
                   for i in range(max_ctas)]
    else:
        # Single core: simple slot view
        fig, ax = plt.subplots(figsize=(14, 3))
        colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0',
                  '#00BCD4', '#795548', '#607D8B', '#E91E63', '#CDDC39']
        for core, slot, start, end, seq in spans:
            color = colors[seq % len(colors)]
            ax.barh(slot, end - start, left=start, height=0.6,
                    color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.text((start + end) / 2, slot, f'{end - start}',
                    ha='center', va='center', fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Slot 0', 'Slot 1'])
        ax.set_ylabel('CTA Slot')
        max_ctas = max(s[4] for s in spans) + 1
        patches = [mpatches.Patch(color=colors[i % len(colors)], label=f'CTA #{i+1}')
                   for i in range(max_ctas)]

    if ncu_cycles:
        ax.axvline(x=ncu_cycles, color='green', linestyle=':', linewidth=2)
        patches.append(plt.Line2D([0], [0], color='green', linestyle=':',
                                  linewidth=2, label=f'NCU={ncu_cycles}'))
    if sim_cycles:
        ax.axvline(x=sim_cycles, color='red', linestyle='--', linewidth=2)
        patches.append(plt.Line2D([0], [0], color='red', linestyle='--',
                                  linewidth=2, label=f'sim={sim_cycles}'))

    ax.legend(handles=patches, loc='lower right', fontsize=9)
    ax.set_xlabel('Cycle', fontsize=12)

    err_str = ""
    if ncu_cycles and sim_cycles:
        err_str = f", err={((sim_cycles / ncu_cycles) - 1) * 100:+.1f}%"
    scope = f"{n_cores_seen} Cores" if all_cores_mode else "Core 0 only"
    ax.set_title(f'{title} — {scope} CTA Lifecycle\n'
                 f'{sum(ctas_per_core.values())} CTAs ({dist_str}){err_str}',
                 fontsize=12)
    xlim = max(sim_cycles or 0, ncu_cycles or 0,
               max(e for _, _, _, e, _ in spans))
    ax.set_xlim(0, xlim * 1.05)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved: {outfile}")


# ── NCU summary lookup ──

def load_ncu_summary(workload):
    """Load NCU sm_cycles from summary.csv if available."""
    ncu_csv = (TRITON_TRACE_DIR / "triton_kernel_tracking" / workload /
               "results" / "ncu-rep" / "summary.csv")
    ncu = {}
    if ncu_csv.exists():
        with open(ncu_csv) as f:
            header = f.readline().strip().split(',')
            for line in f:
                if not line.strip():
                    continue
                vals = line.strip().split(',')
                row = dict(zip(header, vals))
                if 'm' in row:
                    key = (int(row['m']), int(row['n']), int(row['k']))
                else:
                    # flash_attn: use subdir string as key
                    key = vals[0]  # raw first column value
                ncu[key] = int(float(row['sm_cycles']))
    return ncu


# ── CLI ──

def fa_subdir(batch, nheads, seqlen, headdim, causal):
    """Build flash_attn subdirectory name (matches sweep_tests.sh flash_attn_subdir)."""
    causal_str = "_causal" if causal else ""
    return f"b{batch}_h{nheads}_seq{seqlen}_d{headdim}{causal_str}"


def parse_fa_csv(csv_str):
    """Parse B,H,S,D,causal string into (batch, nheads, seqlen, headdim, causal)."""
    parts = csv_str.split(',')
    batch, nheads, seqlen, headdim = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    causal = parts[4].strip().lower() == 'true' if len(parts) > 4 else False
    return batch, nheads, seqlen, headdim, causal


def parse_shapes(args, workload):
    """Parse shape arguments (same logic as sweep_tests.sh)."""
    shapes = []
    if workload == "test_tma_gemm":
        if args.csv:
            csv_path = resolve_input_path(args.csv)
            with open(csv_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('m,'):
                        continue
                    parts = line.split(',')
                    shapes.append((int(parts[0]), int(parts[1]), int(parts[2])))
        if args.shape:
            for s in args.shape:
                m, n, k = s.split(',')
                shapes.append((int(m), int(n), int(k)))
        if args.sizes:
            for s in args.sizes:
                shapes.append((int(s), int(s), int(s)))
    elif workload == "test_flash_attn":
        if args.shape:
            for s in args.shape:
                shapes.append(parse_fa_csv(s))
        if args.sizes:
            # positional args = seq_len sweep with defaults
            batch, nheads, headdim, causal = 1, 8, 128, False
            for s in args.sizes:
                shapes.append((batch, nheads, int(s), headdim, causal))
    return shapes


def main():
    parser = argparse.ArgumentParser(description="Plot CTA lifecycle from sim logs")
    parser.add_argument('workload', choices=['tma_gemm', 'flash_attn'],
                        help='Workload name')
    parser.add_argument('sizes', nargs='*', help='Square sizes (tma_gemm) or seq_len (flash_attn)')
    parser.add_argument('--shape', action='append',
                        help='CSV shape: M,N,K (tma_gemm) or B,H,S,D,causal (flash_attn). Can be repeated.')
    parser.add_argument('--csv', help='CSV file with shapes')
    args = parser.parse_args()

    workload = f"test_{args.workload}"
    shapes = parse_shapes(args, workload)

    if not shapes:
        print("No shapes specified. Use --shape <csv> or positional values.")
        sys.exit(1)

    results_dir = TRITON_TRACE_DIR / "triton_kernel_tracking" / workload / "results"
    sim_log_dir = results_dir / "sim-log"
    cta_lifecycle_dir = results_dir / "cta_lifecycle"
    cta_lifecycle_dir.mkdir(parents=True, exist_ok=True)
    ncu_data = load_ncu_summary(workload)

    for shape in shapes:
        if workload == "test_flash_attn":
            batch, nheads, seqlen, headdim, causal = shape
            label = fa_subdir(batch, nheads, seqlen, headdim, causal)
            log_name = f"{label}_gpgpusim.log"
            ncu_key = f"{batch},{nheads},{seqlen},{headdim},{'True' if causal else 'False'}"
            ncu_cycles = ncu_data.get(ncu_key)
        else:
            m, n, k = shape
            if m == n == k:
                log_name = f"size_{m}_gpgpusim.log"
                label = f"size_{m}"
            else:
                log_name = f"m{m}_n{n}_k{k}_gpgpusim.log"
                label = f"m{m}_n{n}_k{k}"
            ncu_cycles = ncu_data.get(shape)

        logfile = sim_log_dir / log_name
        if not logfile.exists():
            print(f"[SKIP] {label}: log not found at {logfile}")
            continue

        print(f"[{label}]")
        spans = parse_cta_lifecycle(str(logfile))
        sim_cycles = get_sim_cycles(str(logfile))

        if not spans:
            print("  No LIVENESS CTA events found. Enable tracing:")
            print("    -trace_enabled 1")
            print("    -trace_components LIVENESS")
            print("    -trace_sampling_core -1")
            continue

        outfile = cta_lifecycle_dir / f"{label}.png"
        plot_lifecycle(spans, sim_cycles, ncu_cycles, label, str(outfile))


if __name__ == '__main__':
    main()
