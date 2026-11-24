#!/usr/bin/env python3
"""Blocking PTX/SASS launch qualification for the same Blackwell TMEM binary.

Called after the selected host component tests. CUDA 13.3 is pinned in jobs.toml.
Scratch configs disable PTX SASS-guided reorder: synthetic inline-PTX templates
have no complete line-anchor correspondence. Production presets stay unchanged.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

def run_suite(test_filter, tests, launches, timeout=300):
    root = Path(__file__).resolve().parents[2]
    subprocess.run(["make", "-C", str(root / "tests"), "build-sm100-tcgen05",
                    "-j" + os.environ.get("CI_BUILD_JOBS", "2")], check=True)
    parent = root / "tests/logs/ci/tmem"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="dual-", dir=parent))
    subprocess.run([
        sys.executable, str(root / "tests/scripts/run_tcgen05_dual_frontend.py"),
        "--root", str(root),
        "--binary", str(root / "tests/build/bin/sm100/tcgen05_tests"),
        "--config-dir", str(root / "configs/SM100_B200"),
        "--cuda", os.environ["CUDA_INSTALL_PATH"],
        "--output", str(output), "--disable-ptx-reorder",
        "--runtime-library-dir", str(root / "lib" / os.environ["GPGPUSIM_CONFIG"]),
        "--filter", test_filter, "--expected-tests", str(tests),
        "--threads", os.environ.get("OMP_NUM_THREADS", "4"),
        "--timeout", str(timeout),
    ], check=True)
    report = json.loads((output / "report.json").read_text())
    for frontend in ("ptx", "sass-timing"):
        assert report["runs"][frontend]["passed"]
        assert len(report["runs"][frontend]["gpu_sim_cycle"]) == launches
    print(f"PASS: identical executable, {tests} numerical tests / {launches} launches per frontend")
    print("Evidence:", output / "report.json")
