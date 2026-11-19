#!/usr/bin/env python3
"""Run a fixed RTX 4060 GEMM subset; capture/NCU and simulator replay serially by default."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, Future, FIRST_COMPLETED, wait
from pathlib import Path


def complete(case_dir: Path) -> bool:
    reports = list((case_dir / "ncu").glob("reference_*.csv"))
    sim_log = case_dir / "simulator.log"
    if len(reports) != 5 or not sim_log.exists():
        return False
    text = sim_log.read_text(encoding="utf-8", errors="replace")
    return "gpu_tot_sim_cycle =" in text and "Validation PASSED" in text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--max-sim-workers", type=int, default=1)
    args = parser.parse_args()
    runner = args.repo / "tests/triton_trace/calibration/run_gemm_calibration.sh"
    artifact_root = args.repo / "reference/SM89_RTX4060/gemm/full_set"
    driver_logs = args.progress.parent / "fullset_driver_logs"
    driver_logs.mkdir(parents=True, exist_ok=True)
    cases = list(csv.DictReader(args.cases.open(newline="", encoding="utf-8-sig")))
    if not cases:
        raise SystemExit("Case CSV is empty")
    total = len(cases)
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GEMM_CASES_CSV"] = str(args.cases.resolve())
    sim_pool = ThreadPoolExecutor(max_workers=args.max_sim_workers)
    pending: dict[Future[int], tuple[int, str, str, float, Path]] = {}
    failed = 0

    def report_done(future: Future[int]) -> None:
        nonlocal failed
        index, case_id, dtype, start, log_path = pending.pop(future)
        rc = future.result()
        record = {"index": index, "case_id": case_id, "dtype": dtype,
                  "returncode": rc, "elapsed_seconds": round(time.time() - start, 1),
                  "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        with args.progress.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        failed += bool(rc)
        print(f"[{index}/{total}] {'DONE' if not rc else 'FAIL'} "
              f"elapsed={record['elapsed_seconds']}s {case_id}/{dtype}; "
              f"sim log={log_path}", flush=True)

    def run_sim(case_id: str, dtype: str, log_path: Path) -> int:
        with log_path.open("a", encoding="utf-8") as stream:
            return subprocess.run(["bash", str(runner), "sim", case_id, dtype],
                                  cwd=args.repo, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT).returncode

    try:
        for index, row in enumerate(cases, 1):
            case_id, dtype = row["case_id"], row["dtype"]
            case_dir = artifact_root / dtype / case_id
            if complete(case_dir):
                print(f"[{index}/{total}] SKIP already complete {case_id}/{dtype}", flush=True)
                continue
            print(f"[{index}/{total}] CAPTURE+NCU {case_id}/{dtype}", flush=True)
            driver_log = driver_logs / f"{index:03d}_{case_id}_{dtype}.log"
            with driver_log.open("a", encoding="utf-8") as stream:
                if not (case_dir / "experiment-manifest.json").exists():
                    rc = subprocess.run(["bash", str(runner), "capture", case_id, dtype],
                                        cwd=args.repo, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT).returncode
                    if rc:
                        failed += 1
                        print(f"[{index}/{total}] CAPTURE_FAIL rc={rc} {case_id}/{dtype}", flush=True)
                        continue
                if len(list((case_dir / "ncu").glob("reference_*.csv"))) != 5:
                    existing = list((case_dir / "ncu").glob("reference_*.ncu-rep"))
                    if existing:
                        failed += 1
                        print(f"[{index}/{total}] NCU_INCOMPLETE existing reports prevent safe rerun: {case_id}", flush=True)
                        continue
                    rc = subprocess.run(["bash", str(runner), "ncu", case_id, dtype],
                                        cwd=args.repo, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT).returncode
                    if rc:
                        failed += 1
                        print(f"[{index}/{total}] NCU_FAIL rc={rc} {case_id}/{dtype}", flush=True)
                        continue
            sim_log = driver_logs / f"{index:03d}_{case_id}_{dtype}_sim.log"
            start = time.time()
            future = sim_pool.submit(run_sim, case_id, dtype, sim_log)
            pending[future] = (index, case_id, dtype, start, sim_log)
            print(f"[{index}/{total}] SIM_QUEUED active={len(pending)} {case_id}/{dtype}", flush=True)
            if len(pending) >= args.max_sim_workers:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for item in done:
                    report_done(item)
        for item in list(pending):
            report_done(item)
    finally:
        sim_pool.shutdown(wait=True)
    print(f"BATCH_DONE total={len(cases)} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
