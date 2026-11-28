#!/usr/bin/env python3
"""Qualify the same Blackwell MMA executable through PTX and SASS timing."""
from pathlib import Path
import subprocess
import sys
from tcgen05_dual import run_suite

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    subprocess.run([sys.executable, str(root / "tools/calibration/generate_tcgen05_recurrence.py"), "--check"], check=True)
    run_suite("Tcgen05Mma.*", 31, 68, timeout=600)
