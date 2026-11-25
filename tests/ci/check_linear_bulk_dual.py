#!/usr/bin/env python3
from tcgen05_dual import run_suite

if __name__ == "__main__":
    run_suite("LinearBulkLaunch.*", 2, 2, group="memory_forms")
