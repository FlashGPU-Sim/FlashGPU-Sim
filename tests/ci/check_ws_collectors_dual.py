#!/usr/bin/env python3
"""Validate numbered WS B collectors with mutated shared-memory backing."""
from tcgen05_dual import run_suite
if __name__ == "__main__":
    run_suite("Tcgen05WsCollectors.*", 11, 11, timeout=480)
