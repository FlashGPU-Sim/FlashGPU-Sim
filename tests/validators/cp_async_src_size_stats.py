#!/usr/bin/env python3
"""Validate timing traffic for the SM90 cp.async source-size tests."""

from __future__ import annotations

import re
import sys
from pathlib import Path


# Per test: (128-byte timing transactions, physical bytes).
EXPECTED = {
    "FullCopy": (4, 512),
    "ImmediateSourceSize": (4, 512),
    "PerLaneRegisterSourceSize": (3, 384),
    "OddSourceSizes": (4, 512),
    "AllZeroSourceSize": (0, 0),
    "IgnoreSourcePredicate": (2, 256),
    "CachePolicyWithoutSourceSize": (4, 512),
    "SourceSizeWithCachePolicy": (3, 384),
}

COUNTERS = (
    "tx_started",
    "tx_completed",
    "mf_issued",
    "bytes_issued",
    "bytes_completed",
)

RUN_RE = re.compile(r"\[ RUN +\].*CpAsyncSrcSizeTest\.([A-Za-z0-9_]+)")
OK_RE = re.compile(r"\[ +OK +\].*CpAsyncSrcSizeTest\.([A-Za-z0-9_]+)")
COUNTER_RE = re.compile(r"cp_async_debug_([a-z_]+) = ([0-9]+)")


def validate(log_path: Path) -> None:
    current = {name: 0 for name in COUNTERS}
    active_test: str | None = None
    baseline: dict[str, int] = {}
    validated: set[str] = set()

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        run_match = RUN_RE.search(line)
        if run_match and run_match.group(1) in EXPECTED:
            active_test = run_match.group(1)
            baseline = current.copy()

        counter_match = COUNTER_RE.search(line)
        if counter_match and counter_match.group(1) in current:
            current[counter_match.group(1)] = int(counter_match.group(2))

        ok_match = OK_RE.search(line)
        if not ok_match or ok_match.group(1) != active_test:
            continue

        transactions, physical_bytes = EXPECTED[active_test]
        expected_delta = {
            "tx_started": transactions,
            "tx_completed": transactions,
            "mf_issued": transactions,
            "bytes_issued": physical_bytes,
            "bytes_completed": physical_bytes,
        }
        actual_delta = {
            name: current[name] - baseline[name] for name in COUNTERS
        }
        if actual_delta != expected_delta:
            raise ValueError(
                f"{active_test}: expected {expected_delta}, got {actual_delta}"
            )
        validated.add(active_test)
        active_test = None

    missing = set(EXPECTED) - validated
    if missing:
        raise ValueError(f"missing completed tests: {', '.join(sorted(missing))}")

    print(f"Validated cp.async timing traffic for {len(validated)} tests")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} LOG", file=sys.stderr)
        return 2

    try:
        validate(Path(sys.argv[1]))
    except (OSError, ValueError) as error:
        print(f"cp.async timing validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
