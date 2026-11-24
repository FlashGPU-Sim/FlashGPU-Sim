#!/usr/bin/env python3
from tcgen05_dual import run_suite
if __name__ == "__main__":
    run_suite("LeadingBitLaunch.*", 5, 5, group="leading_bit")
    run_suite("LeadingBitBase.*", 2, 2, group="leading_bit_base")
