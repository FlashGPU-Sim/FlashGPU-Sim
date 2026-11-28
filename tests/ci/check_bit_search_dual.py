#!/usr/bin/env python3
from tcgen05_dual import run_suite
if __name__ == "__main__":
    run_suite("BitSearchLaunch.*", 3, 3, group="bit_search")
    run_suite("BitSearchPtxOnly.*", 1, 1, group="bit_search", frontends=("ptx",))
