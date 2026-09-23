#!/usr/bin/env python3
"""Check the SS width diagnostic against job 2119329's measured wait times.

Input: h200_wgmma_width_diagnostic.csv from the unchanged vendor timing loop.
Targets: WgmmaAsyncLatencyBench.F16SsShapeSweep.csv, N=64/128, g1/o1.
This checks completion timing only, not numerical GEMM correctness or issue.
"""
import csv
import sys


def check(path):
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2, "expected exactly the two diagnostic widths"
    assert {int(row["n"]) for row in rows} == {64, 128}
    for row in rows:
        assert int(row["k"]) == 16 and int(row["ops_per_group"]) == 1
        assert int(row["groups_before_wait"]) == 1
        target = {64: 75.0, 128: 107.0}[int(row["n"])]
        value = float(row["wait_cycles_per_round"])
        error = abs(value - target) / target
        assert error < 0.10, f"N={row['n']}: wait {value} vs {target} cycles ({error:.1%})"
    print("WGMMA SS N=64/128 wait timings both within 10%")


if __name__ == "__main__":
    check(sys.argv[1])
