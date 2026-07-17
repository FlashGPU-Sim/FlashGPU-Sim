#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    if not xs:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) * (x - m) for x in xs) / len(xs))


def cosine_distance(a, b):
    keys = sorted(set(a) & set(b))
    if not keys:
        return float("nan")
    av = [a[k] for k in keys]
    bv = [b[k] for k in keys]
    am = mean(av)
    bm = mean(bv)
    ax = [x - am for x in av]
    bx = [x - bm for x in bv]
    dot = sum(x * y for x, y in zip(ax, bx))
    na = math.sqrt(sum(x * x for x in ax))
    nb = math.sqrt(sum(x * x for x in bx))
    if na == 0 or nb == 0:
        return float("nan")
    return 1.0 - dot / (na * nb)


def bit_name(offset):
    if offset > 0 and offset & (offset - 1) == 0:
        return f"bit{offset.bit_length() - 1}"
    return ""


def load_vectors(path):
    values = defaultdict(lambda: defaultdict(list))
    offsets = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            offset_idx = int(row["offset_idx"])
            smid = int(row["smid"])
            offset = int(row["offset"])
            cycles_per_load = float(row["cycles_per_load"])
            values[offset_idx][smid].append(cycles_per_load)
            offsets[offset_idx] = offset
    vectors = {}
    for offset_idx, per_sm in values.items():
        vectors[offset_idx] = {smid: mean(xs) for smid, xs in per_sm.items()}
    return offsets, vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--out", default="")
    parser.add_argument("--cluster-threshold", type=float, default=0.08)
    args = parser.parse_args()

    offsets, vectors = load_vectors(args.csv)
    base_idx = min(offsets, key=lambda i: offsets[i])
    base_vec = vectors[base_idx]

    rows = []
    for offset_idx in sorted(offsets, key=lambda i: offsets[i]):
        offset = offsets[offset_idx]
        vec = vectors[offset_idx]
        vals = list(vec.values())
        dist = cosine_distance(base_vec, vec)
        sorted_sms = sorted(vec.items(), key=lambda kv: kv[1])
        low = ";".join(f"{s}:{v:.2f}" for s, v in sorted_sms[:8])
        high = ";".join(f"{s}:{v:.2f}" for s, v in sorted_sms[-8:])
        rows.append(
            {
                "offset_idx": offset_idx,
                "offset": offset,
                "bit": bit_name(offset),
                "sm_count": len(vec),
                "mean_cycles_per_load": mean(vals),
                "stdev_cycles_per_load": stdev(vals),
                "min_cycles_per_load": min(vals),
                "max_cycles_per_load": max(vals),
                "cosine_distance_from_base": dist,
                "nearest_sms": low,
                "farthest_sms": high,
            }
        )

    # Greedy clustering by latency-vector shape. This gives stable candidate
    # L2-slice groups without assuming the undocumented hash form.
    clusters = []
    for row in rows:
        idx = row["offset_idx"]
        placed = False
        for cluster in clusters:
            rep = cluster[0]["offset_idx"]
            dist = cosine_distance(vectors[idx], vectors[rep])
            if not math.isnan(dist) and dist <= args.cluster_threshold:
                cluster.append(row)
                placed = True
                break
        if not placed:
            clusters.append([row])

    target = open(args.out, "w", newline="") if args.out else None
    try:
        out = target
        if out is None:
            import sys

            out = sys.stdout
        fields = [
            "offset_idx",
            "offset",
            "bit",
            "cluster",
            "sm_count",
            "mean_cycles_per_load",
            "stdev_cycles_per_load",
            "min_cycles_per_load",
            "max_cycles_per_load",
            "cosine_distance_from_base",
            "nearest_sms",
            "farthest_sms",
        ]
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        cluster_by_idx = {}
        for cid, cluster in enumerate(clusters):
            for row in cluster:
                cluster_by_idx[row["offset_idx"]] = cid
        for row in rows:
            row = dict(row)
            row["cluster"] = cluster_by_idx[row["offset_idx"]]
            for k in [
                "mean_cycles_per_load",
                "stdev_cycles_per_load",
                "min_cycles_per_load",
                "max_cycles_per_load",
                "cosine_distance_from_base",
            ]:
                row[k] = f"{row[k]:.6g}"
            writer.writerow(row)
    finally:
        if target is not None:
            target.close()


if __name__ == "__main__":
    main()
