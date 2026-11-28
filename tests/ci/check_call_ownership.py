#!/usr/bin/env python3
from tcgen05_dual import run_suite
if __name__ == "__main__":
    run_suite("PtxCallOwnership.*", 1, 1, group="call_ownership", frontends=("ptx",))
