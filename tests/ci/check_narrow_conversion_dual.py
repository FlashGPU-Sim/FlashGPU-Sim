#!/usr/bin/env python3
"""CUDA reference coverage for retained packed low-precision conversions."""
from tcgen05_dual import run_suite
if __name__ == "__main__":
    run_suite("NarrowLaunch.*", 6, 6, timeout=300, group="narrow_conversion")
