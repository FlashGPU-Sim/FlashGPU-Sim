#!/usr/bin/env python3
"""Require one reference-checked SASS medium forward case within 0.5% drift."""

import csv
from pathlib import Path
import re
import sys


def check(log: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    baseline = root / "tests/baselines/fa3-sass-ccd2-20260914/results.csv"
    with baseline.open() as source:
        expected = {
            row["case"]: int(row["sim_cycles"])
            for row in csv.DictReader(source) if row["profile"] == "medium"
        }
    # The CI runner enables GTest colors even when capturing output to a file.
    # Normalize presentation only; keep the case/kernel and numerical gates strict.
    text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text())
    cases = re.findall(
        r"^\[ RUN      \] Fa3PrefillFp16MediumTest\.(\w+)$", text, re.M)
    cycles = re.findall(r"^gpu_kernel_sim_cycle\[\d+\] = (\d+)$", text, re.M)
    if len(cases) != 1 or cases[0] not in expected or len(cycles) != 1:
        raise SystemExit(f"{log}: require exactly one known medium case/kernel")
    case = cases[0]
    if ("[  PASSED  ] 1 test." not in text or
            "[  FAILED  ]" in text or
            f"FA3 reference PASS: {case} O and LSE" not in text):
        raise SystemExit(f"{log}: missing successful O/LSE reference validation")
    if "mode=execution-driven SASS timing simulation" not in text:
        raise SystemExit(f"{log}: missing actual SASS timing execution")
    actual = int(cycles[0])
    frozen = expected[case]
    drift = 100.0 * (actual - frozen) / frozen
    print(f"{case}: frozen={frozen} current={actual} drift={drift:+.4f}%")
    if abs(drift) > 0.5:
        raise SystemExit("FAIL: cycle drift exceeds 0.5%; baseline unchanged")
    print("PASS: O/LSE reference comparison and SASS cycle drift <= 0.5%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} LOG")
    check(Path(sys.argv[1]))
