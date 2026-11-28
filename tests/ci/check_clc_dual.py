#!/usr/bin/env python3
"""Run actual launch cancellation through the shared PTX/SASS service."""
from tcgen05_dual import run_suite
if __name__ == "__main__":
    run_suite("ClcLaunch.*:PtxB128.*", 7, 13, cluster_validation=True, group="clc")
