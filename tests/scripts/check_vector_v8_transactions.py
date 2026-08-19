#!/usr/bin/env python3
"""Check coalesced traffic for the SM120 vector load/store test."""

from __future__ import annotations

import re
import sys
from pathlib import Path


EXPECTED = {
    "GLOBAL_ACC_R": 5,
    "GLOBAL_ACC_W": 32,
}

COUNTER_RE = re.compile(
    r"Total_core_cache_stats_breakdown\[(GLOBAL_ACC_[RW])\]"
    r"\[TOTAL_ACCESS\] = ([0-9]+)"
)


def validate(log_path: Path) -> None:
    actual: dict[str, int] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = COUNTER_RE.search(line)
        if match:
            actual[match.group(1)] = int(match.group(2))

    if actual != EXPECTED:
        raise ValueError(f"expected {EXPECTED}, got {actual}")

    print("Validated .v8 load/store coalescing traffic")


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} LOG", file=sys.stderr)
        return 2

    try:
        validate(Path(sys.argv[1]))
    except (OSError, ValueError) as error:
        print(f".v8 traffic validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
