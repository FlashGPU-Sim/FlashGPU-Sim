#!/usr/bin/env python3
"""Export a normal FA4 forward invocation for simulator replay."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

import torch
import CutedslTrace
from flash_attn.cute.interface import flash_attn_func


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="Override the default exports/fa4 directory")
    parser.add_argument("--suite", choices=("smoke", "small", "medium", "large", "all"),
                        help="Preset group (default: all, unless a case or manual shape is selected)")
    parser.add_argument("--case", help="Select one named case from the presets")
    parser.add_argument("--list", action="store_true", help="List selected presets without exporting")
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=None)
    parser.add_argument("--sm-count", type=int, default=148)
    parser.add_argument("--non-causal", action="store_true", default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--seqlen", type=int, default=None)
    args = parser.parse_args()
    if args.suite is None and args.case is None and all(
            value is None for value in
            (args.batch, args.heads, args.seqlen, args.head_dim, args.non_causal)):
        args.suite = "all"
    export_root = Path(__file__).resolve().parents[1] / "exports" / "fa4"
    if args.suite or args.case or args.list:
        if any(value is not None for value in
               (args.batch, args.heads, args.seqlen, args.head_dim, args.non_causal)):
            parser.error("preset selection cannot be combined with shape or --non-causal overrides")
        with (Path(__file__).parent / "configs" / "fa4_cases.csv").open() as source:
            cases = [row for row in csv.DictReader(source)
                     if (args.suite in (None, "all") or row["suite"] == args.suite)
                     and (args.case is None or row["name"] == args.case)]
        if not cases:
            parser.error("no matching FA4 case")
        if args.list:
            for row in cases:
                print(",".join(row.values()))
            return
        for row in cases:
            if (row["seqlen_q"] != row["seqlen_k"]
                    or row["head_dim"] != row["head_dim_v"]
                    or row["dtype"] != "fp16" or row["causal"] not in ("true", "false")):
                parser.error(f"unsupported preset fields: {row['name']}")
            output_dir = (Path(args.output_dir) if args.output_dir else export_root / row["suite"]) / row["name"]
            # A fresh process gives each export its own FA4 compilation cache.
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--output-dir", str(output_dir),
                       "--sm-count", str(args.sm_count),
                       "--batch", row["batch"], "--heads", row["heads"],
                       "--seqlen", row["seqlen_q"], "--head-dim", row["head_dim"]]
            if row["causal"] == "false":
                command.append("--non-causal")
            print(f"Exporting {row['suite']}/{row['name']}", flush=True)
            subprocess.run(command, check=True)
        return
    args.batch = args.batch if args.batch is not None else 2
    args.heads = args.heads if args.heads is not None else 32
    args.seqlen = args.seqlen if args.seqlen is not None else 128
    args.head_dim = args.head_dim if args.head_dim is not None else 64
    if min(args.batch, args.heads, args.seqlen, args.sm_count) <= 0:
        parser.error("dimensions and --sm-count must be positive")

    if args.output_dir is None:
        mode = "Full" if args.non_causal else "Causal"
        name = f"H{args.heads}D{args.head_dim}{mode}B{args.batch}S{args.seqlen}"
        args.output_dir = str(export_root / name)
    print(f"Output directory: {Path(args.output_dir).resolve()}", flush=True)

    # Workload selection belongs here, not in the exporter.
    os.environ["FA_DISABLE_2CTA"] = "1"
    tracker = CutedslTrace.Tracker(args.output_dir, target="sm_100a", sm_count=args.sm_count)
    tracker.enable()
    try:
        shape = (args.batch, args.seqlen, args.heads, args.head_dim)
        q = torch.zeros(shape, device="cuda", dtype=torch.float16)
        k = torch.zeros_like(q)
        v = torch.ones_like(q)
        out, _ = flash_attn_func(q, k, v, causal=not args.non_causal)
        # Optional existing smoke oracle; no CPU attention implementation needed.
        tracker.check(out, expected=1)
        tracker.export()
    finally:
        tracker.disable()


if __name__ == "__main__":
    main()
