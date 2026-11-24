#!/usr/bin/env python3
"""Regression for legacy PTX mov packing with mixed literal/register sources."""
from tcgen05_dual import run_suite
if __name__ == "__main__":
    run_suite("MovLiteralPack.*", 1, 1, group="mov_pack")
