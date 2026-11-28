#!/usr/bin/env python3
"""Qualify BF16 scalar/packed conversions through both execution frontends."""
from tcgen05_dual import run_suite

if __name__ == "__main__":
    run_suite("BfloatConversion.*", 4, 8, group="conversion")
