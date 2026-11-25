#!/usr/bin/env python3
from tcgen05_dual import run_suite

if __name__ == "__main__":
    run_suite("CtaReduction.*", 4, 6, group="barrier", frontends=("ptx",))
