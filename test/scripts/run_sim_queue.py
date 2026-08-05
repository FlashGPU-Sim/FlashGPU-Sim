#!/usr/bin/env python3
"""Run simulator jobs with a dynamic worker queue.

The job file is a TSV with at least these columns:

  job_id  stage  case  binary  gtest_filter

Optional columns:

  args    extra arguments appended to the command
  config  simulator config directory name for this job
  skip    if true/1/yes, the job is marked skipped

Paths in `binary` may be absolute or relative to --root.

Use --print-example-jobs to print a minimal TSV template.
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


STATUS_FIELDS = [
    "job_id",
    "index",
    "stage",
    "case",
    "state",
    "rc",
    "elapsed_sec",
    "slot",
    "cpu_set",
    "config",
    "gpu_sim_cycle",
    "gpu_tot_sim_cycle",
    "sim_time_us",
    "rss_mb",
    "max_rss_mb",
    "log",
    "workdir",
]


EXAMPLE_JOBS_TSV = """job_id\tstage\tcase\tbinary\tgtest_filter\tconfig\targs\tskip
cp_async_src_size\tsmoke\tCpAsyncSrcSizeTest\ttest/build/bin/sm90/integration_tests\tCpAsyncSrcSizeTest.*\tSM90_H100\t\t0
"""


@dataclass(frozen=True)
class Job:
    index: int
    job_id: str
    stage: str
    case: str
    binary: Path
    gtest_filter: str
    args: tuple[str, ...]
    config: str
    skip: bool = False


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y", "skip"}


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", name).strip("_")


def parse_core_clock_mhz(config_path: Path) -> float | None:
    text = config_path.read_text(errors="replace")
    match = re.search(r"(?m)^\s*-gpgpu_clock_domains\s+([0-9.]+):", text)
    return float(match.group(1)) if match else None


def parse_cycles(log_path: Path) -> tuple[int | None, int | None]:
    if not log_path.exists():
        return None, None
    text = log_path.read_text(errors="replace")
    sim_matches = re.findall(r"\bgpu_sim_cycle\s*=\s*([0-9]+)", text)
    total_matches = re.findall(r"\bgpu_tot_sim_cycle\s*=\s*([0-9]+)", text)
    sim_cycle = int(sim_matches[-1]) if sim_matches else None
    total_cycle = int(total_matches[-1]) if total_matches else None
    return sim_cycle, total_cycle


def proc_tree_pids(root_pid: int) -> list[int]:
    parent: dict[int, int] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = stat_path.read_text(errors="replace")
            close = text.rfind(")")
            fields = text[close + 2 :].split()
            pid = int(stat_path.parent.name)
            ppid = int(fields[1])
            parent[pid] = ppid
        except (OSError, ValueError, IndexError):
            continue

    pids = [root_pid]
    seen = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parent.items():
            if pid not in seen and ppid in seen:
                seen.add(pid)
                pids.append(pid)
                changed = True
    return pids


def rss_mb_for_pids(pids: Iterable[int]) -> float:
    total_kb = 0
    for pid in pids:
        status_path = Path("/proc") / str(pid) / "status"
        try:
            for line in status_path.read_text(errors="replace").splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        total_kb += int(parts[1])
                    break
        except (OSError, ValueError):
            continue
    return total_kb / 1024.0


def read_jobs(path: Path, root: Path) -> list[Job]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"job_id", "stage", "case", "binary", "gtest_filter"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(f"jobs file must contain columns: {', '.join(sorted(required))}")
        jobs: list[Job] = []
        seen_job_ids: set[str] = set()
        for index, row in enumerate(reader):
            if not any((value or "").strip() for value in row.values()):
                continue
            for field in required:
                if not (row.get(field) or "").strip():
                    raise SystemExit(f"row {index + 2}: {field} must not be empty")
            binary = Path(row["binary"])
            if not binary.is_absolute():
                binary = root / binary
            job_id = sanitize(row["job_id"])
            if not job_id:
                raise SystemExit(f"row {index + 2}: job_id has no usable characters")
            if job_id in seen_job_ids:
                raise SystemExit(f"row {index + 2}: duplicate job_id {job_id!r}")
            seen_job_ids.add(job_id)
            jobs.append(
                Job(
                    index=index,
                    job_id=job_id,
                    stage=row["stage"].strip(),
                    case=row["case"].strip(),
                    binary=binary,
                    gtest_filter=row["gtest_filter"].strip(),
                    args=tuple(shlex.split(row.get("args") or "")),
                    config=(row.get("config") or "").strip(),
                    skip=truthy(row.get("skip")),
                )
            )
    if not jobs:
        raise SystemExit(f"no jobs found in {path}")
    return jobs


def print_example_jobs() -> None:
    print(EXAMPLE_JOBS_TSV, end="")


def write_status(status_dir: Path, job: Job, values: dict[str, object]) -> None:
    status_path = status_dir / f"{job.job_id}.status"
    merged = {
        "job_id": job.job_id,
        "index": job.index,
        "stage": job.stage,
        "case": job.case,
        "config": job.config,
        **values,
    }
    lines = [f"{key}={merged.get(key, '')}" for key in STATUS_FIELDS]
    status_path.write_text("\n".join(lines) + "\n")


def read_status_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key] = value
    return data


def refresh_summary(status_dir: Path) -> None:
    rows: list[dict[str, str]] = []
    for path in sorted(status_dir.glob("*.status")):
        rows.append(read_status_file(path))
    summary_path = status_dir / "summary.tsv"
    tmp_path = status_dir / "summary.tsv.tmp"
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in STATUS_FIELDS})
    tmp_path.replace(summary_path)


def copy_config(config_dir: Path, workdir: Path) -> None:
    for path in config_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, workdir / path.name)


def make_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    cuda = str(args.cuda_path)
    gpgpusim_config = args.gpgpusim_config
    root = str(args.root)
    env.update(
        {
            "CUDA_INSTALL_PATH": cuda,
            "CUDA_VERSION_NUMBER": args.cuda_version_number,
            "GPGPUSIM_ROOT": root,
            "GPGPUSIM_CONFIG": gpgpusim_config,
            "GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN": "1",
            "GPGPUSIM_POWER_MODEL": f"{root}/src/accelwattch/",
            "PTXAS_CUDA_INSTALL_PATH": cuda,
            "QTINC": env.get("QTINC", "/usr/include"),
            "OMP_NUM_THREADS": str(args.threads_per_job),
            "OPENBLAS_NUM_THREADS": str(args.threads_per_job),
            "MKL_NUM_THREADS": str(args.threads_per_job),
            "NUMEXPR_NUM_THREADS": str(args.threads_per_job),
            "GTEST_COLOR": "no",
        }
    )
    for key, value in (
        ("PTX_SIM_USE_PTX_FILE", args.ptx_sim_use_ptx_file),
        ("PTX_SIM_KERNELFILE", args.ptx_sim_kernelfile),
        ("CUOBJDUMP_SIM_FILE", args.cuobjdump_sim_file),
    ):
        if value:
            env[key] = value
        else:
            env.pop(key, None)
    path_parts = [f"{root}/bin", f"{cuda}/bin", env.get("PATH", "")]
    ld_parts = [f"{root}/lib/{gpgpusim_config}", f"{cuda}/lib64", env.get("LD_LIBRARY_PATH", "")]
    env["PATH"] = ":".join(part for part in path_parts if part)
    env["LD_LIBRARY_PATH"] = ":".join(part for part in ld_parts if part)
    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def command_for(job: Job, cpu_set: str | None) -> list[str]:
    cmd = [str(job.binary), f"--gtest_filter={job.gtest_filter}", *job.args]
    if cpu_set:
        return ["taskset", "-c", cpu_set, *cmd]
    return cmd


def run_job(
    args: argparse.Namespace,
    job: Job,
    slot: int,
    cpu_set: str | None,
    env: dict[str, str],
    core_clock_mhz: float | None,
    summary_lock: threading.Lock,
) -> int:
    logs_dir = args.run_root / "logs"
    work_root = args.run_root / "work"
    status_dir = args.run_root / "status"
    workdir = work_root / f"{job.index:03d}_{sanitize(job.stage)}_{sanitize(job.case)}"
    log_path = logs_dir / f"{job.index:03d}_{sanitize(job.stage)}_{sanitize(job.case)}.log"
    config_name = job.config or args.config
    config_dir = args.root / "configs" / config_name
    if not config_dir.is_dir():
        write_status(
            status_dir,
            job,
            {
                "state": "missing_config",
                "rc": 126,
                "elapsed_sec": 0,
                "slot": slot,
                "cpu_set": cpu_set or "",
                "config": config_name,
                "log": log_path,
                "workdir": workdir,
            },
        )
        with summary_lock:
            refresh_summary(status_dir)
        return 126

    if args.resume:
        existing = status_dir / f"{job.job_id}.status"
        if existing.exists() and read_status_file(existing).get("state") == "done":
            return 0

    if job.skip:
        write_status(
            status_dir,
            job,
            {
                "state": "skipped",
                "rc": "skip",
                "elapsed_sec": 0,
                "slot": slot,
                "cpu_set": cpu_set or "",
                "config": config_name,
                "log": log_path,
                "workdir": workdir,
            },
        )
        with summary_lock:
            refresh_summary(status_dir)
        return 0

    if not job.binary.exists():
        write_status(
            status_dir,
            job,
            {
                "state": "missing_binary",
                "rc": 127,
                "elapsed_sec": 0,
                "slot": slot,
                "cpu_set": cpu_set or "",
                "config": config_name,
                "log": log_path,
                "workdir": workdir,
            },
        )
        with summary_lock:
            refresh_summary(status_dir)
        return 127

    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    copy_config(config_dir, workdir)
    job_core_clock_mhz = parse_core_clock_mhz(config_dir / "gpgpusim.config")

    cmd = command_for(job, cpu_set if shutil.which("taskset") else None)
    start = time.time()
    write_status(
        status_dir,
        job,
        {
            "state": "running",
            "rc": "",
            "elapsed_sec": "",
            "slot": slot,
            "cpu_set": cpu_set or "",
            "config": config_name,
            "log": log_path,
            "workdir": workdir,
        },
    )
    with summary_lock:
        refresh_summary(status_dir)

    with log_path.open("w") as log:
        log.write(f"[{timestamp()}] START job_id={job.job_id} stage={job.stage} case={job.case}\n")
        log.write(f"[{timestamp()}] workdir={workdir}\n")
        log.write(f"[{timestamp()}] cpu_set={cpu_set or ''}\n")
        log.write(f"[{timestamp()}] command={' '.join(shlex.quote(part) for part in cmd)}\n")
        log.write(f"[{timestamp()}] config={config_name}\n")
        log.write(f"[{timestamp()}] core_clock_mhz={job_core_clock_mhz or core_clock_mhz or ''}\n")
        log.write(f"[{timestamp()}] LD_LIBRARY_PATH={env.get('LD_LIBRARY_PATH', '')}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        rc: int | None = None
        max_rss_mb = 0.0
        max_rss_limit_mb = args.max_rss_gb * 1024.0 if args.max_rss_gb > 0 else 0.0
        deadline = start + args.timeout if args.timeout > 0 else None
        while True:
            current_rc = proc.poll()
            pids = proc_tree_pids(proc.pid) if current_rc is None else [proc.pid]
            rss_mb = rss_mb_for_pids(pids)
            max_rss_mb = max(max_rss_mb, rss_mb)
            elapsed_now = int(time.time() - start)

            if current_rc is not None:
                rc = current_rc
                break

            if deadline is not None and time.time() >= deadline:
                rc = 124
                log.write(f"\n[{timestamp()}] TIMEOUT after {args.timeout}s\n")
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                break

            if max_rss_limit_mb and rss_mb > max_rss_limit_mb:
                rc = 137
                log.write(
                    f"\n[{timestamp()}] RSS_LIMIT rss_mb={rss_mb:.1f} "
                    f"limit_mb={max_rss_limit_mb:.1f}; terminating job\n"
                )
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=10)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                break

            write_status(
                status_dir,
                job,
                {
                    "state": "running",
                    "rc": "",
                    "elapsed_sec": elapsed_now,
                    "slot": slot,
                    "cpu_set": cpu_set or "",
                    "config": config_name,
                    "rss_mb": f"{rss_mb:.1f}",
                    "max_rss_mb": f"{max_rss_mb:.1f}",
                    "log": log_path,
                    "workdir": workdir,
                },
            )
            with summary_lock:
                refresh_summary(status_dir)
            time.sleep(args.heartbeat_interval)

    elapsed = int(time.time() - start)
    sim_cycle, total_cycle = parse_cycles(log_path)
    selected_cycle = total_cycle if total_cycle is not None else sim_cycle
    sim_time_us = ""
    selected_core_clock_mhz = job_core_clock_mhz or core_clock_mhz
    if selected_cycle is not None and selected_core_clock_mhz:
        sim_time_us = f"{selected_cycle / selected_core_clock_mhz:.6f}"
    state = "done" if rc == 0 else "failed"
    write_status(
        status_dir,
        job,
        {
            "state": state,
            "rc": rc,
            "elapsed_sec": elapsed,
            "slot": slot,
            "cpu_set": cpu_set or "",
            "config": config_name,
            "gpu_sim_cycle": sim_cycle or "",
            "gpu_tot_sim_cycle": total_cycle or "",
            "sim_time_us": sim_time_us,
            "rss_mb": "",
            "max_rss_mb": f"{max_rss_mb:.1f}" if "max_rss_mb" in locals() else "",
            "log": log_path,
            "workdir": workdir,
        },
    )
    with summary_lock:
        refresh_summary(status_dir)
    return rc


def worker(
    args: argparse.Namespace,
    slot: int,
    jobs: "queue.Queue[Job]",
    env: dict[str, str],
    core_clock_mhz: float | None,
    summary_lock: threading.Lock,
    rc_by_slot: list[int],
) -> None:
    cpu_set = args.cpu_sets[slot] if slot < len(args.cpu_sets) else None
    rc_any = 0
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            break
        print(f"[{timestamp()}] START slot={slot} cpu={cpu_set or ''} job={job.job_id}", flush=True)
        rc = run_job(args, job, slot, cpu_set, env, core_clock_mhz, summary_lock)
        if rc != 0:
            rc_any = rc_any or rc
            print(f"[{timestamp()}] FAIL  slot={slot} rc={rc} job={job.job_id}", flush=True)
        else:
            print(f"[{timestamp()}] DONE  slot={slot} job={job.job_id}", flush=True)
        jobs.task_done()
    rc_by_slot[slot] = rc_any


def write_metadata(args: argparse.Namespace, jobs: Iterable[Job], core_clock_mhz: float | None) -> None:
    lines = [
        f"created_at={timestamp()}",
        f"root={args.root}",
        f"run_root={args.run_root}",
        f"jobs={args.jobs}",
        f"config={args.config}",
        f"config_dir={args.config_dir}",
        f"core_clock_mhz={core_clock_mhz or ''}",
        f"max_parallel={args.max_parallel}",
        f"cpu_sets={' '.join(args.cpu_sets)}",
        f"timeout={args.timeout}",
        f"cuda_path={args.cuda_path}",
        f"gpgpusim_config={args.gpgpusim_config}",
    ]
    (args.run_root / "metadata.txt").write_text("\n".join(lines) + "\n")
    queue_path = args.run_root / "status" / "queue.tsv"
    with queue_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "job_id",
                "index",
                "stage",
                "case",
                "binary",
                "gtest_filter",
                "config",
                "args",
                "skip",
            ]
        )
        for job in jobs:
            writer.writerow(
                [
                    job.job_id,
                    job.index,
                    job.stage,
                    job.case,
                    job.binary,
                    job.gtest_filter,
                    job.config or args.config,
                    " ".join(shlex.quote(part) for part in job.args),
                    int(job.skip),
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--config", default="SM90_H100")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument(
        "--cpu-sets",
        nargs="*",
        default=[],
        help="Optional taskset CPU lists, one per worker slot",
    )
    parser.add_argument("--threads-per-job", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--heartbeat-interval", type=int, default=30)
    parser.add_argument("--max-rss-gb", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--cuda-path",
        type=Path,
        default=Path(os.environ["CUDA_INSTALL_PATH"])
        if os.environ.get("CUDA_INSTALL_PATH")
        else None,
        help="CUDA Toolkit root (defaults to CUDA_INSTALL_PATH)",
    )
    parser.add_argument(
        "--cuda-version-number",
        default=os.environ.get("CUDA_VERSION_NUMBER", "12080"),
    )
    parser.add_argument(
        "--gpgpusim-config",
        default=os.environ.get(
            "GPGPUSIM_CONFIG", "gcc-13.3.0/cuda-12080/release"
        ),
    )
    parser.add_argument("--ptx-sim-use-ptx-file", default=None)
    parser.add_argument("--ptx-sim-kernelfile", default=None)
    parser.add_argument("--cuobjdump-sim-file", default="jj")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--print-example-jobs", action="store_true")
    args = parser.parse_args()

    if args.print_example_jobs:
        print_example_jobs()
        raise SystemExit(0)

    args.root = args.root.resolve()
    if args.run_root is None:
        raise SystemExit("--run-root is required")
    if args.jobs is None:
        raise SystemExit("--jobs is required")
    if args.cuda_path is None:
        raise SystemExit("--cuda-path is required when CUDA_INSTALL_PATH is not set")
    args.run_root = args.run_root.resolve()
    args.jobs = args.jobs.resolve()
    args.config_dir = args.root / "configs" / args.config
    if not args.config_dir.is_dir():
        raise SystemExit(f"missing config dir: {args.config_dir}")
    if not args.jobs.exists():
        raise SystemExit(f"missing jobs file: {args.jobs}")
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    logs_dir = args.run_root / "logs"
    status_dir = args.run_root / "status"
    work_dir = args.run_root / "work"
    for path in (logs_dir, status_dir, work_dir):
        path.mkdir(parents=True, exist_ok=True)

    core_clock_mhz = parse_core_clock_mhz(args.config_dir / "gpgpusim.config")
    jobs = read_jobs(args.jobs, args.root)
    write_metadata(args, jobs, core_clock_mhz)

    for job in jobs:
        write_status(
            status_dir,
            job,
            {
                "state": "queued",
                "rc": "",
                "elapsed_sec": "",
                "slot": "",
                "cpu_set": "",
                "log": logs_dir / f"{job.index:03d}_{sanitize(job.stage)}_{sanitize(job.case)}.log",
                "workdir": work_dir / f"{job.index:03d}_{sanitize(job.stage)}_{sanitize(job.case)}",
            },
        )
    refresh_summary(status_dir)

    env = make_env(args)
    job_queue: "queue.Queue[Job]" = queue.Queue()
    for job in jobs:
        job_queue.put(job)

    worker_count = min(args.max_parallel, len(jobs))
    summary_lock = threading.Lock()
    rc_by_slot = [0 for _ in range(worker_count)]
    threads = [
        threading.Thread(
            target=worker,
            args=(args, slot, job_queue, env, core_clock_mhz, summary_lock, rc_by_slot),
            daemon=False,
        )
        for slot in range(worker_count)
    ]

    print(f"[{timestamp()}] RUN_ROOT={args.run_root}", flush=True)
    print(f"[{timestamp()}] jobs={len(jobs)} workers={worker_count} config={args.config}", flush=True)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    refresh_summary(status_dir)
    rc = 0 if all(code == 0 for code in rc_by_slot) else 1
    (status_dir / "complete").write_text(f"rc={rc}\ncompleted_at={timestamp()}\n")
    print(f"[{timestamp()}] ALL_DONE rc={rc} summary={status_dir / 'summary.tsv'}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
