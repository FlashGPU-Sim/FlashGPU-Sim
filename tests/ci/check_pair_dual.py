#!/usr/bin/env python3
"""Qualify paired-CTA launches through both frontends and the shared backend."""
from tcgen05_dual import run_suite
if __name__ == "__main__":
    run_suite("Tcgen05Pair.*", 6, 17, timeout=600, cluster_validation=True)
