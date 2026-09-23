#!/usr/bin/env python3
"""Check the actual copy-engine trace for an isolated high-address H2D preload.

After rebuilding with TRACE=1 (TRACING_ON), copy SM90_H200_CLUSTER132's
config files to a fresh temporary run directory. Append to its gpgpusim.config:
  -gpgpu_perf_sim_memcpy 1
  -trace_enabled 1
  -trace_components MEMORY_PARTITION_UNIT
  -trace_sampling_memory_partition -1
With the usual simulator setup environment and OMP_NUM_THREADS=4, run the
existing binary there (use an absolute path from the temporary directory):
  calibration/kernels/h200_probes/vendor/microbench/bin/flashgpu/memory/l2_latency_bench
with these verified arguments:
  --case=l2-hit --bytes=4096 --steps=4 --samples=1 --warmup-steps=0 --cache-op=cg
Capture stdout/stderr, then: python3 test/check_memcpy_preload.py RUN.log
Do not use run_calibration_sim.sh on that directory: it overwrites the config.

This benchmark's first allocation/preload is its 4096-byte pointer chain;
the current allocator starts at GLOBAL_HEAP_START=0xC00000000. If the allocator
or benchmark changes, pass the independently verified --base and --bytes.
Require successful benchmark completion separately. This check validates the
addresses passed to L2's preload path, not subsequent cache-hit timing. It
does not exercise the >4-GiB copy-length counter (that requires a huge copy).
"""

import argparse
from pathlib import Path
import re


def check(text, base=0xC00000000, count=4096):
    if base < 1 << 32 or base % 32 or count <= 0 or count % 32:
        raise ValueError("require a base above 4 GiB and a positive, sector-aligned span")
    addresses = [int(value, 16) for value in re.findall(
        r"Copy Engine Request Received For Address=([0-9a-fA-F]+),", text)]
    expected = list(range(base, base + count, 32))
    if addresses[:len(expected)] != expected:
        raise ValueError("missing, truncated, misaligned, or unexpected initial preload addresses")
    return len(expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--base", type=lambda value: int(value, 0), default=0xC00000000)
    parser.add_argument("--bytes", type=lambda value: int(value, 0), default=4096)
    args = parser.parse_args()
    sectors = check(args.log.read_text(errors="replace"), args.base, args.bytes)
    print(f"PASS: first {sectors} copy-engine sectors preserve the high address")
