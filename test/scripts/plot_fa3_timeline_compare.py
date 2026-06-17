#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import defaultdict
from statistics import mean, median
from xml.sax.saxutils import escape


ACTORS = ["producer", "consumer_low", "consumer_high"]

INTERVALS = [
    ("producer", "K load", "producer_load_k_begin", "producer_load_k_end", "#4e79a7"),
    ("producer", "V load", "producer_load_v_begin", "producer_load_v_end", "#59a14f"),
    ("producer", "O barrier", "producer_barrier_o_begin", "producer_barrier_o_end", "#f28e2b"),
    ("producer", "tail", "producer_tail_begin", "producer_tail_end", "#b07aa1"),
    ("consumer", "K wait", "consumer_k_wait_begin", "consumer_k_wait_end", "#e15759"),
    ("consumer", "scheduler wait", "consumer_scheduler_wait_begin", "consumer_scheduler_wait_end", "#8cd17d"),
    ("consumer", "QK issue", "consumer_qk_issue_begin", "consumer_qk_issue_end", "#4e79a7"),
    ("consumer", "PV wait", "consumer_pv_wait_begin", "consumer_pv_wait_end", "#f28e2b"),
    ("consumer", "PV issue->arrive", "consumer_pv_issue_begin", "consumer_scheduler_arrive", "#76b7b2"),
    ("consumer", "arrive->wait1", "consumer_scheduler_arrive", "consumer_wait1_end", "#edc948"),
    ("consumer", "wait0", "consumer_wait0_begin", "consumer_wait0_end", "#b07aa1"),
]

POINT_EVENTS = [
    ("consumer_k_release", "#222222"),
    ("consumer_v_release", "#555555"),
    ("tile_begin", "#111111"),
]

RESIDUAL_COLOR = "#f3c7c9"


def read_timeline(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["clock_i"] = int(row["clock"])
            row["step_i"] = int(row["step"])
            row["n_block_i"] = int(row["n_block"])
            rows.append(row)
    t0 = min(r["clock_i"] for r in rows)
    for row in rows:
        row["t"] = row["clock_i"] - t0
    return rows


def group_events(rows):
    grouped = defaultdict(dict)
    meta = {}
    for row in rows:
        key = (row["actor"], row["step_i"], row["n_block_i"])
        grouped[key][row["event"]] = row["t"]
        meta[key] = row
    return grouped, meta


def actor_matches(actor_rule, actor):
    if actor_rule == "consumer":
        return actor.startswith("consumer_")
    return actor_rule == actor


def collect_intervals(rows):
    grouped, _ = group_events(rows)
    spans = []
    for (actor, step, n_block), events in grouped.items():
        for actor_rule, name, begin, end, color in INTERVALS:
            if not actor_matches(actor_rule, actor):
                continue
            if begin in events and end in events:
                spans.append(
                    {
                        "actor": actor,
                        "step": step,
                        "n_block": n_block,
                        "name": name,
                        "begin_event": begin,
                        "end_event": end,
                        "start": events[begin],
                        "end": events[end],
                        "duration": events[end] - events[begin],
                        "color": color,
                    }
                )
    return spans


def collect_step_spans(rows):
    grouped, _ = group_events(rows)
    spans = []
    for (actor, step, n_block), events in grouped.items():
        if not actor.startswith("consumer_"):
            continue
        if "consumer_step_begin" in events and "consumer_step_end" in events:
            spans.append(
                {
                    "actor": actor,
                    "step": step,
                    "n_block": n_block,
                    "start": events["consumer_step_begin"],
                    "end": events["consumer_step_end"],
                    "duration": events["consumer_step_end"] - events["consumer_step_begin"],
                }
            )
    return spans


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def collect_consumer_residuals(rows):
    spans = collect_intervals(rows)
    step_spans = collect_step_spans(rows)
    by_actor_step = defaultdict(list)
    for sp in spans:
        if sp["actor"].startswith("consumer_"):
            by_actor_step[(sp["actor"], sp["step"])].append((sp["start"], sp["end"]))
    residuals = []
    for step in step_spans:
        intervals = []
        for s, e in by_actor_step.get((step["actor"], step["step"]), []):
            s = max(s, step["start"])
            e = min(e, step["end"])
            if e > s:
                intervals.append((s, e))
        covered = sum(e - s for s, e in merge_intervals(intervals))
        residuals.append(
            {
                "actor": step["actor"],
                "step": step["step"],
                "n_block": step["n_block"],
                "duration": step["duration"] - covered,
            }
        )
    return residuals


def collect_consumer_residual_segments(rows):
    spans = collect_intervals(rows)
    step_spans = collect_step_spans(rows)
    by_actor_step = defaultdict(list)
    for sp in spans:
        if sp["actor"].startswith("consumer_"):
            by_actor_step[(sp["actor"], sp["step"])].append((sp["start"], sp["end"]))
    residuals = []
    for step in step_spans:
        intervals = []
        for s, e in by_actor_step.get((step["actor"], step["step"]), []):
            s = max(s, step["start"])
            e = min(e, step["end"])
            if e > s:
                intervals.append((s, e))
        merged = merge_intervals(intervals)
        cur = step["start"]
        for s, e in merged:
            if s > cur:
                residuals.append({**step, "start": cur, "end": s, "duration": s - cur})
            cur = max(cur, e)
        if step["end"] > cur:
            residuals.append({**step, "start": cur, "end": step["end"], "duration": step["end"] - cur})
    return residuals


def nice_ticks(max_x, target=10):
    if max_x <= 0:
        return [0]
    rough = max_x / target
    mag = 10 ** math.floor(math.log10(rough))
    for mult in [1, 2, 5, 10]:
        step = mult * mag
        if rough <= step:
            break
    ticks = []
    x = 0
    while x <= max_x * 1.001:
        ticks.append(int(x))
        x += step
    return ticks


def svg_text(x, y, text, size=13, anchor="start", weight="normal", fill="#222"):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="{fill}">{escape(str(text))}</text>'
    )


def svg_rect(x, y, w, h, fill, stroke="none", opacity=1.0, rx=1):
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0):.2f}" height="{h:.2f}" '
        f'rx="{rx}" fill="{fill}" stroke="{stroke}" opacity="{opacity:.3f}"/>'
    )


def draw_timeline_svg(datasets, out_path, title, window=None, width=1800):
    left = 190
    right = 40
    top = 58
    panel_gap = 64
    lane_h = 34
    actor_gap = 14
    panel_h = len(ACTORS) * (lane_h + actor_gap) + 58
    legend_h = 66
    height = top + len(datasets) * panel_h + (len(datasets) - 1) * panel_gap + legend_h
    if window:
        x0, x1 = window
    else:
        x0 = 0
        x1 = max(max(r["t"] for r in ds["rows"]) for ds in datasets)
    plot_w = width - left - right
    scale = plot_w / max(1, x1 - x0)

    def sx(t):
        return left + (t - x0) * scale

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        svg_text(24, 30, title, 20, weight="700"),
    ]

    # Legend.
    legend_y = height - legend_h + 18
    lx = left
    legend_items = [("residual/unattributed", RESIDUAL_COLOR)] + [(name, color) for _, name, _, _, color in INTERVALS]
    seen = set()
    compact = []
    for item in legend_items:
        if item[0] not in seen:
            compact.append(item)
            seen.add(item[0])
    for name, color in compact:
        parts.append(svg_rect(lx, legend_y - 11, 18, 10, color, opacity=0.85, rx=2))
        parts.append(svg_text(lx + 24, legend_y - 2, name, 12))
        lx += max(92, len(name) * 7 + 36)
        if lx > width - 220:
            lx = left
            legend_y += 22

    ticks = [t for t in nice_ticks(x1 - x0) if x0 + t <= x1]
    for pidx, ds in enumerate(datasets):
        y_base = top + pidx * (panel_h + panel_gap)
        parts.append(svg_text(24, y_base - 18, ds["label"], 16, weight="700"))
        parts.append(svg_text(24, y_base + 4, f"span={int(max(r['t'] for r in ds['rows']))} cycles", 12, fill="#555"))
        axis_y = y_base + len(ACTORS) * (lane_h + actor_gap) + 8
        parts.append(f'<line x1="{left}" y1="{axis_y:.1f}" x2="{width-right}" y2="{axis_y:.1f}" stroke="#888" stroke-width="1"/>')
        for tick in ticks:
            abs_tick = x0 + tick
            tx = sx(abs_tick)
            parts.append(f'<line x1="{tx:.1f}" y1="{y_base-10:.1f}" x2="{tx:.1f}" y2="{axis_y+4:.1f}" stroke="#e6e6e6" stroke-width="1"/>')
            parts.append(svg_text(tx, axis_y + 20, f"T+{int(abs_tick)//1000}K" if abs_tick else "T+0", 11, anchor="middle", fill="#555"))

        spans = collect_intervals(ds["rows"])
        step_spans = collect_step_spans(ds["rows"])
        residual_segments = collect_consumer_residual_segments(ds["rows"])
        point_by_actor = defaultdict(list)
        for r in ds["rows"]:
            for event, color in POINT_EVENTS:
                if r["event"] == event:
                    point_by_actor[r["actor"]].append((r["t"], color, event))

        for aidx, actor in enumerate(ACTORS):
            y = y_base + aidx * (lane_h + actor_gap)
            parts.append(svg_text(24, y + 22, actor, 13, weight="700"))
            parts.append(svg_rect(left, y, plot_w, lane_h, "#f7f7f7", stroke="#dddddd", opacity=1.0, rx=2))

            for ss in step_spans:
                if ss["actor"] != actor:
                    continue
                s, e = ss["start"], ss["end"]
                if e < x0 or s > x1:
                    continue
                color = "#eeeeee" if ss["step"] % 2 == 0 else "#e6e6e6"
                parts.append(svg_rect(sx(max(s, x0)), y + 2, max(1.0, (min(e, x1) - max(s, x0)) * scale), lane_h - 4, color, opacity=0.65, rx=1))
                if scale * (e - s) > 34:
                    parts.append(svg_text(sx((s + e) / 2), y + lane_h - 5, ss["step"], 9, anchor="middle", fill="#777"))

            for rs in residual_segments:
                if rs["actor"] != actor:
                    continue
                s, e = rs["start"], rs["end"]
                if e < x0 or s > x1:
                    continue
                parts.append(svg_rect(sx(max(s, x0)), y + 4, max(1.0, (min(e, x1) - max(s, x0)) * scale), lane_h - 8, RESIDUAL_COLOR, opacity=0.72, rx=1))

            for sp in spans:
                if sp["actor"] != actor:
                    continue
                s, e = sp["start"], sp["end"]
                if e < x0 or s > x1:
                    continue
                yy = y + 9
                hh = 14
                parts.append(svg_rect(sx(max(s, x0)), yy, max(1.5, (min(e, x1) - max(s, x0)) * scale), hh, sp["color"], opacity=0.86, rx=2))

            for t, color, _ in point_by_actor.get(actor, []):
                if x0 <= t <= x1:
                    x = sx(t)
                    parts.append(f'<line x1="{x:.1f}" y1="{y+5:.1f}" x2="{x:.1f}" y2="{y+lane_h-5:.1f}" stroke="{color}" stroke-width="1" opacity="0.75"/>')

    parts.append("</svg>")
    with open(out_path, "w") as f:
        f.write("\n".join(parts))


def summarize_intervals(label, rows):
    spans = collect_intervals(rows)
    by_key = defaultdict(list)
    for sp in spans:
        by_key[(sp["actor"], sp["name"])].append(sp["duration"])
    out = []
    for (actor, name), vals in sorted(by_key.items()):
        out.append(
            {
                "dataset": label,
                "actor": actor,
                "interval": name,
                "count": len(vals),
                "total": sum(vals),
                "avg": mean(vals),
                "median": median(vals),
                "min": min(vals),
                "max": max(vals),
            }
        )
    residual_by_actor = defaultdict(list)
    for row in collect_consumer_residuals(rows):
        residual_by_actor[row["actor"]].append(row["duration"])
    for actor, vals in sorted(residual_by_actor.items()):
        out.append(
            {
                "dataset": label,
                "actor": actor,
                "interval": "residual/unattributed",
                "count": len(vals),
                "total": sum(vals),
                "avg": mean(vals),
                "median": median(vals),
                "min": min(vals),
                "max": max(vals),
            }
        )
    return out


def write_summary_csv(sim_rows, ncu_rows, out_dir):
    summaries = summarize_intervals("sim", sim_rows) + summarize_intervals("ncu", ncu_rows)
    path = os.path.join(out_dir, "interval_summary.csv")
    fields = ["dataset", "actor", "interval", "count", "total", "avg", "median", "min", "max"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summaries:
            w.writerow(row)

    keyed = defaultdict(dict)
    for row in summaries:
        keyed[(row["actor"], row["interval"])][row["dataset"]] = row
    delta_path = os.path.join(out_dir, "interval_delta_summary.csv")
    delta_fields = [
        "actor",
        "interval",
        "count_sim",
        "count_ncu",
        "avg_sim",
        "avg_ncu",
        "avg_ncu_minus_sim",
        "total_sim",
        "total_ncu",
        "total_ncu_minus_sim",
    ]
    deltas = []
    for key, pair in sorted(keyed.items()):
        if "sim" not in pair or "ncu" not in pair:
            continue
        s, n = pair["sim"], pair["ncu"]
        deltas.append(
            {
                "actor": key[0],
                "interval": key[1],
                "count_sim": s["count"],
                "count_ncu": n["count"],
                "avg_sim": s["avg"],
                "avg_ncu": n["avg"],
                "avg_ncu_minus_sim": n["avg"] - s["avg"],
                "total_sim": s["total"],
                "total_ncu": n["total"],
                "total_ncu_minus_sim": n["total"] - s["total"],
            }
        )
    with open(delta_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=delta_fields)
        w.writeheader()
        for row in deltas:
            w.writerow(row)
    return path, delta_path, deltas


def write_step_delta_csv(sim_rows, ncu_rows, out_dir):
    def step_map(rows):
        return {(s["actor"], s["step"]): s for s in collect_step_spans(rows)}

    sim = step_map(sim_rows)
    ncu = step_map(ncu_rows)
    path = os.path.join(out_dir, "step_duration_delta.csv")
    fields = ["actor", "step", "sim_duration", "ncu_duration", "ncu_minus_sim"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in sorted(set(sim) & set(ncu)):
            s, n = sim[key], ncu[key]
            w.writerow(
                {
                    "actor": key[0],
                    "step": key[1],
                    "sim_duration": s["duration"],
                    "ncu_duration": n["duration"],
                    "ncu_minus_sim": n["duration"] - s["duration"],
                }
            )
    return path


def summarize_event_gaps(label, rows):
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["actor"], row["step_i"], row["n_block_i"])].append(row)
    gaps = defaultdict(list)
    for (actor, _, _), group in by_key.items():
        group = sorted(group, key=lambda r: r["t"])
        for a, b in zip(group, group[1:]):
            name = f"{a['event']} -> {b['event']}"
            gaps[(actor, name)].append(b["t"] - a["t"])
    out = []
    for (actor, name), vals in sorted(gaps.items()):
        out.append(
            {
                "dataset": label,
                "actor": actor,
                "gap": name,
                "count": len(vals),
                "total": sum(vals),
                "avg": mean(vals),
                "median": median(vals),
                "min": min(vals),
                "max": max(vals),
            }
        )
    return out


def write_event_gap_delta_csv(sim_rows, ncu_rows, out_dir):
    summaries = summarize_event_gaps("sim", sim_rows) + summarize_event_gaps("ncu", ncu_rows)
    keyed = defaultdict(dict)
    for row in summaries:
        keyed[(row["actor"], row["gap"])][row["dataset"]] = row
    path = os.path.join(out_dir, "event_gap_delta_summary.csv")
    fields = [
        "actor",
        "gap",
        "count_sim",
        "count_ncu",
        "avg_sim",
        "avg_ncu",
        "avg_ncu_minus_sim",
        "total_sim",
        "total_ncu",
        "total_ncu_minus_sim",
    ]
    rows = []
    for key, pair in sorted(keyed.items()):
        if "sim" not in pair or "ncu" not in pair:
            continue
        s, n = pair["sim"], pair["ncu"]
        rows.append(
            {
                "actor": key[0],
                "gap": key[1],
                "count_sim": s["count"],
                "count_ncu": n["count"],
                "avg_sim": s["avg"],
                "avg_ncu": n["avg"],
                "avg_ncu_minus_sim": n["avg"] - s["avg"],
                "total_sim": s["total"],
                "total_ncu": n["total"],
                "total_ncu_minus_sim": n["total"] - s["total"],
            }
        )
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path, rows


def write_notes(out_dir, deltas, gap_deltas):
    top = sorted(deltas, key=lambda r: r["total_ncu_minus_sim"], reverse=True)[:16]
    neg = sorted(deltas, key=lambda r: r["total_ncu_minus_sim"])[:10]
    path = os.path.join(out_dir, "timeline_compare_notes.md")
    with open(path, "w") as f:
        f.write("# FA3 qk_pv_only_no_tma timeline compare\n\n")
        f.write("Positive delta means H100 under NCU is slower than simulator for that interval.\n\n")
        f.write("## Largest H100-slower intervals\n\n")
        f.write("| actor | interval | avg sim | avg ncu | avg ncu-sim | total ncu-sim |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in top:
            f.write(
                f"| {r['actor']} | {r['interval']} | {r['avg_sim']:.1f} | {r['avg_ncu']:.1f} | "
                f"{r['avg_ncu_minus_sim']:.1f} | {r['total_ncu_minus_sim']:.0f} |\n"
            )
        f.write("\n## Intervals where simulator is slower\n\n")
        f.write("| actor | interval | avg sim | avg ncu | avg ncu-sim | total ncu-sim |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in neg:
            f.write(
                f"| {r['actor']} | {r['interval']} | {r['avg_sim']:.1f} | {r['avg_ncu']:.1f} | "
                f"{r['avg_ncu_minus_sim']:.1f} | {r['total_ncu_minus_sim']:.0f} |\n"
            )
        f.write("\n## Largest adjacent timestamp gaps where H100 is slower\n\n")
        f.write("| actor | gap | avg sim | avg ncu | avg ncu-sim | total ncu-sim |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in sorted(gap_deltas, key=lambda x: x["total_ncu_minus_sim"], reverse=True)[:18]:
            f.write(
                f"| {r['actor']} | {r['gap']} | {r['avg_sim']:.1f} | {r['avg_ncu']:.1f} | "
                f"{r['avg_ncu_minus_sim']:.1f} | {r['total_ncu_minus_sim']:.0f} |\n"
            )
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--ncu", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sim_rows = read_timeline(args.sim)
    ncu_rows = read_timeline(args.ncu)
    datasets = [
        {"label": "SIM qk_pv_only_no_tma", "rows": sim_rows},
        {"label": "H100 under NCU qk_pv_only_no_tma", "rows": ncu_rows},
    ]

    max_span = max(max(r["t"] for r in sim_rows), max(r["t"] for r in ncu_rows))
    draw_timeline_svg(
        datasets,
        os.path.join(args.out_dir, "timeline_overview.svg"),
        "FA3 qk_pv_only_no_tma: SIM vs H100 under NCU timeline",
        window=(0, max_span),
    )
    draw_timeline_svg(
        datasets,
        os.path.join(args.out_dir, "timeline_first_30k.svg"),
        "FA3 qk_pv_only_no_tma: first 30K cycles",
        window=(0, 30000),
    )
    draw_timeline_svg(
        datasets,
        os.path.join(args.out_dir, "timeline_steady_45k_85k.svg"),
        "FA3 qk_pv_only_no_tma: steady-state window 45K-85K cycles",
        window=(45000, 85000),
    )
    _, _, deltas = write_summary_csv(sim_rows, ncu_rows, args.out_dir)
    write_step_delta_csv(sim_rows, ncu_rows, args.out_dir)
    _, gap_deltas = write_event_gap_delta_csv(sim_rows, ncu_rows, args.out_dir)
    write_notes(args.out_dir, deltas, gap_deltas)

    print(args.out_dir)


if __name__ == "__main__":
    main()
