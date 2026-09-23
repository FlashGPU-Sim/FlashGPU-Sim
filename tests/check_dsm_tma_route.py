#!/usr/bin/env python3
"""Check opt-in DSM counters in a completed, isolated simulator run.

After rebuilding the simulator, from the repository root with its normal
setup environment, run (shell backslashes join lines):

  FLASHGPU_DSM_STATS=1 bash scripts/run_calibration_sim.sh \\
    --config SM90_H200_CLUSTER132 --run-dir calibration/runs/dsm-tma-global-mcast \\
    -- "$PWD/test/build/bin/sm120/run_integration_tests" \\
    --gtest_filter='TmaMulticastMaskTest.*:TMAClusterMulticastTensor2DTest.ClusterMulticastTensor2DTMALoad'

Capture stdout/stderr to RUN.log, require exit 0 and non-skipped passing tests,
then run: python3 test/check_dsm_tma_route.py RUN.log
The tensor test exercises legacy shared-cluster fan-out, not explicit tensor
multicast syntax; also check the hardware-matched multicast GEMM run.

Positive control (capture output to CONTROL.log and require correctness):

  FLASHGPU_DSM_STATS=1 bash scripts/run_calibration_sim.sh \\
    --config SM90_H200_CLUSTER132 --run-dir calibration/runs/dsm-tma-mapped-control \\
    -- "$PWD/calibration/kernels/h200_probes/vendor/dsm_bw/dsm_h200.out" \\
    --suite smoke --warmups 0 --iterations 1 --output control.jsonl
  python3 test/check_dsm_tma_route.py CONTROL.log --expect mapped

This includes mapped shared-to-shared TMA. Choose fresh run directories/logs.

Counters are cumulative: use separate processes for global and mapped tests.
Zero mode forbids all DSM payload classes, allowing mbarrier control traffic.
This checks routing only; separately require kernel correctness and exit 0.
Missing/disabled endpoints, incomplete dumps and missing counters fail.
"""

import argparse
import gzip
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(text, expect="global"):
    require(expect in ("global", "mapped"), "unknown routing expectation")
    payload = ("stores", "loads", "read_data", "atoms", "tma")
    clusters = None
    current = None
    seen = set()
    dumps = 0
    mapped = False
    for line in text.splitlines():
        if line.startswith("DSM_ROUTE_STATS_BEGIN clusters="):
            require(clusters is None, "nested/incomplete DSM dump")
            header = dict(field.split("=", 1) for field in line.split()[1:])
            clusters = int(header["clusters"])
            require(header["enabled"] == "1", "DSM model disabled")
            require(clusters > 0, "no DSM clusters")
            seen = set()
        elif line.startswith("DSM_ROUTE_STATS_CLUSTER id="):
            require(clusters is not None and current is None, "missing endpoint counters")
            current = int(line.split("=")[1])
            require(0 <= current < clusters and current not in seen, "invalid/duplicate cluster")
        elif clusters is not None and line.lstrip().startswith("stores="):
            require(current is not None, "unattributed/duplicate endpoint counters")
            values = dict(field.split("=", 1) for field in line.split())
            counts = [int(values[key]) for key in payload]
            require(all(value >= 0 for value in counts), "negative packet count")
            if expect == "global":
                require(not any(counts), f"DSM payload observed at cluster {current}: {values}")
            mapped |= int(values["tma"]) > 0
            seen.add(current)
            current = None
        elif line == "DSM_ROUTE_STATS_END":
            require(clusters is not None and current is None, "incomplete DSM dump")
            require(seen == set(range(clusters)), "missing endpoint counters")
            dumps += 1
            clusters = None
    require(clusters is None, "truncated DSM dump")
    require(dumps, "no DSM counter dumps; enable FLASHGPU_DSM_STATS=1 and DSM")
    require(expect != "mapped" or mapped, "positive control produced no mapped TMA packets")
    return dumps


def self_test():
    begin = "DSM_ROUTE_STATS_BEGIN clusters=1 enabled=1\nDSM_ROUTE_STATS_CLUSTER id=0\n"
    counters = " stores=0 loads=0 read_data=0 atoms=0 tma=0 mbar_req=4 mbar_done=4\n"
    end = "DSM_ROUTE_STATS_END\n"
    good = begin + counters + end
    require(check(good + good) == 2, "valid repeated dumps rejected")
    positive = good.replace("tma=0", "tma=2")
    require(check(positive, "mapped") == 1, "positive control rejected")
    for text, mode in (("", "global"), (begin + end, "global"),
                       (begin + counters, "global"), (positive, "global"),
                       (good, "mapped"), (good.replace("clusters=1", "clusters=2"), "global"),
                       (good.replace("tma=0", "other=0"), "global"),
                       (good.replace("stores=0", "stores=1"), "global"),
                       (good.replace("enabled=1", "enabled=0"), "global"),
                       (good.replace(" enabled=1", ""), "global")):
        try:
            check(text, mode)
        except (KeyError, ValueError):
            pass
        else:
            raise AssertionError("accepted invalid routing evidence")
    print("self-test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?")
    parser.add_argument("--expect", choices=("global", "mapped"), default="global")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif not args.log:
        parser.error("provide a completed simulator log or --self-test")
    else:
        path = Path(args.log)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            dumps = check(stream.read(), args.expect)
        print(f"{dumps} DSM dumps verified ({args.expect}); kernel correctness checked separately")
