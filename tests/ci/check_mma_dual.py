#!/usr/bin/env python3
"""Qualify the same Blackwell MMA executable through PTX and SASS timing."""
from tcgen05_dual import run_suite
if __name__ == "__main__":
    run_suite("Tcgen05Mma.*", 18, 36)
