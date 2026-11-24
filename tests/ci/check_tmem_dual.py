#!/usr/bin/env python3
"""Qualify the original TMEM suite independently of added MMA fixtures."""
from tcgen05_dual import run_suite
if __name__ == "__main__":
    run_suite("Tcgen05Fragments.*:Tcgen05Copy.*", 12, 14)
