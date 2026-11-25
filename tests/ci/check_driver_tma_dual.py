#!/usr/bin/env python3
"""Qualify retained SM100 descriptor layouts through both execution frontends."""
from tcgen05_dual import run_suite

if __name__ == "__main__":
    run_suite("DriverTma.*:TmaTensorCluster.*", 11, 38, cluster_validation=True, group="driver_tma")

    run_suite("SharedAddress.*", 1, 1, cluster_validation=True, group="shared_address")
