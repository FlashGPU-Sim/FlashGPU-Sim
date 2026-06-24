#!/usr/bin/env python3
import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


METRICS = [
    "lts__t_sectors_srcunit_tex.sum",
    "lts__t_sectors_srcunit_tex.max",
    "lts__t_sectors_srcunit_tex.avg",
    "lts__t_sectors_srcunit_tex.min",
    "lts__t_sectors_srcunit_tex_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_lookup_miss.sum",
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg",
    "device__attribute_l2s_count",
    "device__attribute_fbp_count",
    "device__attribute_num_l2s_per_fbp",
]


def parse_float(value):
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_ncu_raw(path):
    rows = []
    header = None
    with Path(path).open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0] == "ID":
                header = row
                continue
            if header is None:
                continue
            if not row[0]:
                continue
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            rows.append(dict(zip(header, row)))
    return rows


def read_pair_rows(path):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows):
        row.setdefault("launch_index", str(i))
    return rows


def classify(sum_value, max_value, same_threshold, diff_threshold):
    if not sum_value or not max_value or max_value <= 0:
        return "unknown", math.nan
    ratio = sum_value / max_value
    if ratio <= same_threshold:
        return "same", ratio
    if ratio >= diff_threshold:
        return "different", ratio
    return "ambiguous", ratio


def summarize_group(rows):
    total = len(rows)
    counts = defaultdict(int)
    ratios = []
    for row in rows:
        counts[row["class"]] += 1
        if row["sum_over_max"] == row["sum_over_max"]:
            ratios.append(row["sum_over_max"])
    mean_ratio = sum(ratios) / len(ratios) if ratios else math.nan
    return {
        "pairs": total,
        "same": counts["same"],
        "different": counts["different"],
        "ambiguous": counts["ambiguous"],
        "unknown": counts["unknown"],
        "same_frac": counts["same"] / total if total else math.nan,
        "different_frac": counts["different"] / total if total else math.nan,
        "mean_sum_over_max": mean_ratio,
    }


def write_joined(path, rows):
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "launch_index",
        "kind",
        "tag",
        "bit",
        "page_left",
        "page_right",
        "line_left",
        "line_right",
        "left_offset",
        "right_offset",
        "event_ms",
        "sum",
        "max",
        "avg",
        "min",
        "hit_sum",
        "miss_sum",
        "sum_over_max",
        "class",
        "gpu_time_ns",
        "sm_cycles",
    ]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True,
                        help="Batch CSV emitted by l2_hbm_interleave_bench.")
    parser.add_argument("--ncu-raw", required=True)
    parser.add_argument("--joined-out", default="")
    parser.add_argument("--same-threshold", type=float, default=1.35)
    parser.add_argument("--different-threshold", type=float, default=1.65)
    parser.add_argument("--nonparticipating-threshold", type=float, default=0.95)
    args = parser.parse_args()

    pair_rows = read_pair_rows(args.pairs)
    ncu_rows = read_ncu_raw(args.ncu_raw)
    if len(ncu_rows) != len(pair_rows):
        print(f"warning: pair rows={len(pair_rows)} ncu rows={len(ncu_rows)}; "
              f"joining first {min(len(pair_rows), len(ncu_rows))}",
              file=sys.stderr)

    joined = []
    for i, (pair, ncu) in enumerate(zip(pair_rows, ncu_rows)):
        sum_value = parse_float(ncu.get("lts__t_sectors_srcunit_tex.sum"))
        max_value = parse_float(ncu.get("lts__t_sectors_srcunit_tex.max"))
        cls, ratio = classify(sum_value, max_value, args.same_threshold,
                              args.different_threshold)
        joined.append({
            **pair,
            "launch_index": pair.get("launch_index", str(i)),
            "sum": sum_value,
            "max": max_value,
            "avg": parse_float(ncu.get("lts__t_sectors_srcunit_tex.avg")),
            "min": parse_float(ncu.get("lts__t_sectors_srcunit_tex.min")),
            "hit_sum": parse_float(
                ncu.get("lts__t_sectors_srcunit_tex_lookup_hit.sum")),
            "miss_sum": parse_float(
                ncu.get("lts__t_sectors_srcunit_tex_lookup_miss.sum")),
            "gpu_time_ns": parse_float(ncu.get("gpu__time_duration.sum")),
            "sm_cycles": parse_float(ncu.get("sm__cycles_elapsed.avg")),
            "sum_over_max": ratio,
            "class": cls,
        })

    write_joined(args.joined_out, joined)

    first_ncu = ncu_rows[0] if ncu_rows else {}
    l2s = parse_float(first_ncu.get("device__attribute_l2s_count"))
    fbp = parse_float(first_ncu.get("device__attribute_fbp_count"))
    l2s_per_fbp = parse_float(first_ncu.get("device__attribute_num_l2s_per_fbp"))
    print(f"joined_pairs={len(joined)}")
    print(f"device_l2s_count={int(l2s) if l2s else 'unknown'} "
          f"fbp_count={int(fbp) if fbp else 'unknown'} "
          f"l2s_per_fbp={int(l2s_per_fbp) if l2s_per_fbp else 'unknown'}")
    if args.joined_out:
        print(f"joined_csv={args.joined_out}")

    by_kind = defaultdict(list)
    by_bit = defaultdict(list)
    by_cross_page = defaultdict(list)
    for row in joined:
        by_kind[row.get("kind", "")].append(row)
        if row.get("kind") == "derivative":
            by_bit[int(row.get("bit", -1))].append(row)
        if row.get("kind") == "cross_page_same_line":
            by_cross_page[row.get("page_right", "")].append(row)

    print("\nby_kind")
    print("kind,pairs,same,different,ambiguous,unknown,same_frac,different_frac,mean_sum_over_max")
    for kind in sorted(by_kind):
        s = summarize_group(by_kind[kind])
        print(f"{kind},{s['pairs']},{s['same']},{s['different']},"
              f"{s['ambiguous']},{s['unknown']},{s['same_frac']:.4f},"
              f"{s['different_frac']:.4f},{s['mean_sum_over_max']:.4f}")

    if by_bit:
        print("\nderivative_by_bit")
        print("bit,pairs,same,different,ambiguous,unknown,same_frac,different_frac,mean_sum_over_max,candidate_nonparticipating")
        candidates = []
        for bit in sorted(by_bit):
            s = summarize_group(by_bit[bit])
            candidate = s["same_frac"] >= args.nonparticipating_threshold
            if candidate:
                candidates.append(bit)
            print(f"{bit},{s['pairs']},{s['same']},{s['different']},"
                  f"{s['ambiguous']},{s['unknown']},{s['same_frac']:.4f},"
                  f"{s['different_frac']:.4f},{s['mean_sum_over_max']:.4f},"
                  f"{int(candidate)}")
        if candidates:
            saving = 1 << len(candidates)
            print(f"\ncandidate_nonparticipating_bits={candidates} "
                  f"candidate_line_space_reduction={saving}x")
        else:
            print("\ncandidate_nonparticipating_bits=[]")

    if by_cross_page:
        print("\ncross_page_same_line_by_page")
        print("page_right,pairs,same,different,ambiguous,unknown,same_frac,different_frac,mean_sum_over_max")
        for page in sorted(by_cross_page, key=lambda x: int(x)):
            s = summarize_group(by_cross_page[page])
            print(f"{page},{s['pairs']},{s['same']},{s['different']},"
                  f"{s['ambiguous']},{s['unknown']},{s['same_frac']:.4f},"
                  f"{s['different_frac']:.4f},{s['mean_sum_over_max']:.4f}")


if __name__ == "__main__":
    main()
