#!/usr/bin/env python3
"""Qualify retained SM100 descriptor layouts through both execution frontends."""
from tcgen05_dual import run_suite

if __name__ == "__main__":
    run_suite("DriverTma.*", 5, 12, cluster_validation=True, group="driver_tma")
