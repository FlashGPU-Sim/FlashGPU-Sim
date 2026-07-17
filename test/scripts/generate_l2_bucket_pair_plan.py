#!/usr/bin/env python3
import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path


def sample_lines(total_lines, count, seed):
    if count >= total_lines:
        return list(range(total_lines))
    rng = random.Random(seed)
    lines = {0}
    # Mix evenly spaced anchors with random lines so low and high line bits both move.
    for i in range(count // 2):
        lines.add((i * total_lines) // max(1, count // 2))
    while len(lines) < count:
        lines.add(rng.randrange(total_lines))
    return sorted(lines)


def add_row(rows, kind, tag, bit, page_left, page_right, line_left, line_right,
            page_bytes, line_bytes):
    if page_left == page_right and line_left == line_right:
        return
    rows.append({
        "kind": kind,
        "tag": tag,
        "bit": bit,
        "page_left": page_left,
        "page_right": page_right,
        "line_left": line_left,
        "line_right": line_right,
        "left_offset": page_left * page_bytes + line_left * line_bytes,
        "right_offset": page_right * page_bytes + line_right * line_bytes,
    })


def main():
    parser = argparse.ArgumentParser(
        description="Generate a bounded pair plan for L2/LTS bucket probing.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--sample-lines", type=int, default=64)
    parser.add_argument("--pairs-per-bit", type=int, default=8)
    parser.add_argument("--random-pairs", type=int, default=0)
    parser.add_argument("--bit-first", type=int, default=7)
    parser.add_argument("--bit-last", type=int, default=20)
    parser.add_argument("--page-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--line-bytes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="Optional hard cap; rows are deterministically sampled.")
    args = parser.parse_args()

    if args.pages < 1:
        raise SystemExit("--pages must be >= 1")
    if args.page_bytes % args.line_bytes != 0:
        raise SystemExit("--page-bytes must be divisible by --line-bytes")
    if args.line_bytes & (args.line_bytes - 1):
        raise SystemExit("--line-bytes must be a power of two")
    if args.bit_first < args.line_bytes.bit_length() - 1:
        raise SystemExit("--bit-first is below the cache-line offset bits")

    page_lines = args.page_bytes // args.line_bytes
    line_bit_base = args.line_bytes.bit_length() - 1
    max_page_bit = line_bit_base + page_lines.bit_length() - 2
    if args.bit_last > max_page_bit:
        print(f"warning: clamping --bit-last from {args.bit_last} to "
              f"{max_page_bit} for one {args.page_bytes}-byte page",
              file=sys.stderr)
        args.bit_last = max_page_bit

    selected_lines = sample_lines(page_lines, args.sample_lines, args.seed)
    rng = random.Random(args.seed)
    rows = []

    for page in range(args.pages):
        for line in selected_lines:
            add_row(rows, "page_anchor", f"p{page}", -1, page, page, 0, line,
                    args.page_bytes, args.line_bytes)

    for page in range(args.pages):
        for bit in range(args.bit_first, args.bit_last + 1):
            line_bit = bit - line_bit_base
            delta = 1 << line_bit
            candidates = [line for line in range(page_lines)
                          if (line & delta) == 0 and (line ^ delta) < page_lines]
            if not candidates:
                continue
            chosen = rng.sample(candidates, min(args.pairs_per_bit,
                                                len(candidates)))
            for line in chosen:
                add_row(rows, "derivative", f"p{page}_b{bit}", bit, page,
                        page, line, line ^ delta, args.page_bytes,
                        args.line_bytes)

    if args.pages > 1:
        for page in range(1, args.pages):
            for line in selected_lines:
                add_row(rows, "cross_page_same_line", f"p0_p{page}", -1, 0,
                        page, line, line, args.page_bytes, args.line_bytes)

    for i in range(args.random_pairs):
        page = rng.randrange(args.pages)
        left = rng.randrange(page_lines)
        right = rng.randrange(page_lines)
        if left == right:
            right = (right + 1) % page_lines
        add_row(rows, "random_intra_page", f"r{i}", -1, page, page, left,
                right, args.page_bytes, args.line_bytes)

    pretrim_count = len(rows)
    if args.max_pairs and len(rows) > args.max_pairs:
        rng.shuffle(rows)
        rows = rows[:args.max_pairs]
        rows.sort(key=lambda r: (r["page_left"], r["page_right"], r["kind"],
                                 r["bit"], r["line_left"], r["line_right"]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "kind",
        "tag",
        "bit",
        "page_left",
        "page_right",
        "line_left",
        "line_right",
        "left_offset",
        "right_offset",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["kind"] for row in rows)
    print(f"out={out_path}", file=sys.stderr)
    print(f"page_lines={page_lines} pages={args.pages} "
          f"selected_lines={len(selected_lines)} "
          f"pairs={len(rows)} pretrim_pairs={pretrim_count}",
          file=sys.stderr)
    for kind, count in sorted(counts.items()):
        print(f"{kind}={count}", file=sys.stderr)


if __name__ == "__main__":
    main()
