#!/usr/bin/env python3
from tcgen05_dual import run_suite
if __name__ == "__main__":
    run_suite("WarpReduceLaunch.*:WarpShiftLaunch.*", 8, 8, group="warp_reduce")
    run_suite("CachedLoadCompletion.*", 1, 1, group="cached_completion")
