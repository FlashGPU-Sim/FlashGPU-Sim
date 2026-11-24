#!/usr/bin/env python3
"""Qualify retained cache hints and shared bulk stores through both frontends."""
from tcgen05_dual import run_suite
if __name__ == "__main__":
    run_suite("SharedBulkLaunch.*:CacheHintLaunch.*", 7, 7, group="memory_forms")
