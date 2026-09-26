#!/usr/bin/env python3
"""Run independent commands with a bounded, CPU-aware worker queue.

The job manifest may be CSV or TSV.  Its required columns are ``job_id`` and
``executable`` (legacy ``binary`` is accepted); ``args`` is optional.  Unknown
columns are retained as opaque metadata and never affect execution.
"""

from __future__ import annotations

import argparse
import csv
import errno
import io
import json
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from cpu_affinity import (
    allowed_cpus,
    parse_cpu_list,
    select_cpu_sets,
    validate_cpu_sets,
)


SUMMARY_FIELDS = [
    "job_id",
    "index",
    "state",
    "rc",
    "elapsed_sec",
    "slot",
    "cpu_set",
    "threads_per_job",
    "rss_mb",
    "max_rss_mb",
    "log",
    "workdir",
    "metadata",
    "metrics",
]

EXECUTION_FIELDS = {
    "job_id",
    "executable",
    "binary",
    "args",
    "cwd",
    "env",
    "timeout",
    "enabled",
    # Compatibility fields from the original simulator manifest.
    "gtest_filter",
    "skip",
}

EXAMPLE_JOBS_CSV = """job_id,executable,args,suite
example,tests/example_runner,"--mode smoke --size 128",smoke
"""


@dataclass(frozen=True)
class Job:
    index: int
    job_id: str
    executable: Path
    args: tuple[str, ...]
    cwd: Path | None
    env: dict[str, str]
    timeout: int | None
    enabled: bool
    metadata: dict[str, str]
    gtest_filter: str = ""
    legacy_config: str = ""
    legacy_manifest: bool = False


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", name).strip("_")


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
        "skip",
    }


def enabled_value(value: str | None) -> bool:
    if value is None or not value.strip():
        return True
    return value.strip().lower() not in {"0", "false", "no", "n", "off", "disabled"}


def parse_env(value: str, row_number: int) -> dict[str, str]:
    value = value.strip()
    if not value:
        return {}
    if value.startswith("{"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"row {row_number}: invalid env JSON: {exc}") from exc
        if not isinstance(parsed, dict) or not all(
            isinstance(key, str) and isinstance(item, (str, int, float, bool))
            for key, item in parsed.items()
        ):
            raise SystemExit(
                f"row {row_number}: env JSON must be an object of scalar values"
            )
        return {key: str(item) for key, item in parsed.items()}

    result: dict[str, str] = {}
    try:
        entries = shlex.split(value)
    except ValueError as exc:
        raise SystemExit(f"row {row_number}: invalid env: {exc}") from exc
    for entry in entries:
        if "=" not in entry:
            raise SystemExit(
                f"row {row_number}: env expects KEY=VALUE entries, got {entry!r}"
            )
        key, item = entry.split("=", 1)
        if not key:
            raise SystemExit(f"row {row_number}: env key must not be empty")
        result[key] = item
    return result


def detect_delimiter(name: str, text: str, requested: str) -> str:
    if requested == "csv":
        return ","
    if requested == "tsv":
        return "\t"
    suffix = Path(name).suffix.lower() if name != "-" else ""
    if suffix == ".csv":
        return ","
    if suffix in {".tsv", ".tab"}:
        return "\t"
    try:
        return csv.Sniffer().sniff(text[:8192], delimiters=",\t").delimiter
    except csv.Error as exc:
        raise SystemExit(
            "could not detect job format; pass --jobs-format csv or tsv"
        ) from exc


def read_manifest_text(jobs_argument: str) -> tuple[str, str]:
    if jobs_argument == "-":
        return "-", sys.stdin.read()
    path = Path(jobs_argument).resolve()
    if not path.is_file():
        raise SystemExit(f"jobs file does not exist: {path}")
    return str(path), path.read_text()


def read_jobs(
    jobs_argument: str, jobs_format: str, root: Path
) -> tuple[list[Job], str, str]:
    source_name, text = read_manifest_text(jobs_argument)
    delimiter = detect_delimiter(source_name, text, jobs_format)
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if reader.fieldnames is None:
        raise SystemExit("job manifest has no header")
    reader.fieldnames = [field.lstrip("\ufeff").strip() for field in reader.fieldnames]
    fieldnames = set(reader.fieldnames)
    if "job_id" not in fieldnames:
        raise SystemExit("job manifest must contain job_id")
    if not ({"executable", "binary"} & fieldnames):
        raise SystemExit("job manifest must contain executable (or legacy binary)")

    legacy_manifest = "binary" in fieldnames and "executable" not in fieldnames
    jobs: list[Job] = []
    seen_job_ids: set[str] = set()
    for input_index, row in enumerate(reader):
        row_number = input_index + 2
        if None in row:
            raise SystemExit(f"row {row_number}: more values than header columns")
        normalized = {str(key).strip(): (value or "").strip() for key, value in row.items()}
        if not any(normalized.values()):
            continue

        raw_job_id = normalized.get("job_id", "")
        job_id = sanitize(raw_job_id)
        if not job_id:
            raise SystemExit(f"row {row_number}: job_id has no usable characters")
        if job_id != raw_job_id:
            raise SystemExit(
                f"row {row_number}: job_id must already be filename-safe; "
                f"suggested value: {job_id!r}"
            )
        if job_id in seen_job_ids:
            raise SystemExit(f"row {row_number}: duplicate job_id {job_id!r}")
        seen_job_ids.add(job_id)

        executable_text = normalized.get("executable") or normalized.get("binary")
        if not executable_text:
            raise SystemExit(f"row {row_number}: executable must not be empty")
        executable = Path(executable_text)
        if not executable.is_absolute():
            executable = root / executable

        try:
            command_args = tuple(shlex.split(normalized.get("args", "")))
        except ValueError as exc:
            raise SystemExit(f"row {row_number}: invalid args: {exc}") from exc

        cwd_text = normalized.get("cwd", "")
        cwd = Path(cwd_text) if cwd_text else None
        if cwd is not None and not cwd.is_absolute():
            cwd = root / cwd

        timeout_text = normalized.get("timeout", "")
        timeout_value: int | None = None
        if timeout_text:
            try:
                timeout_value = int(timeout_text)
            except ValueError as exc:
                raise SystemExit(
                    f"row {row_number}: timeout must be a non-negative integer"
                ) from exc
            if timeout_value < 0:
                raise SystemExit(
                    f"row {row_number}: timeout must be a non-negative integer"
                )

        enabled = enabled_value(normalized.get("enabled"))
        if legacy_manifest and truthy(normalized.get("skip")):
            enabled = False

        execution_fields = EXECUTION_FIELDS | ({"config"} if legacy_manifest else set())
        metadata = {
            key: value
            for key, value in normalized.items()
            if key not in execution_fields and value
        }
        legacy_config = normalized.get("config", "") if legacy_manifest else ""
        jobs.append(
            Job(
                index=len(jobs),
                job_id=job_id,
                executable=executable.resolve(),
                args=command_args,
                cwd=cwd.resolve() if cwd else None,
                env=parse_env(normalized.get("env", ""), row_number),
                timeout=timeout_value,
                enabled=enabled,
                metadata=metadata,
                gtest_filter=normalized.get("gtest_filter", ""),
                legacy_config=legacy_config,
                legacy_manifest=legacy_manifest,
            )
        )
    if not jobs:
        raise SystemExit(f"no jobs found in {source_name}")
    return jobs, source_name, "csv" if delimiter == "," else "tsv"


def print_example_jobs() -> None:
    print(EXAMPLE_JOBS_CSV, end="")


def status_path(status_dir: Path, job: Job) -> Path:
    return status_dir / f"{job.job_id}.json"


def base_status(job: Job) -> dict[str, object]:
    return {
        "job_id": job.job_id,
        "index": job.index,
        "state": "queued" if job.enabled else "disabled",
        "rc": "" if job.enabled else "disabled",
        "elapsed_sec": 0,
        "slot": "",
        "cpu_set": "",
        "threads_per_job": "",
        "rss_mb": "",
        "max_rss_mb": "",
        "log": "",
        "workdir": "",
        "metadata": job.metadata,
        "metrics": {},
    }


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_status(status_dir: Path, job: Job, values: dict[str, object]) -> None:
    merged = base_status(job)
    merged.update(values)
    write_json_atomic(status_path(status_dir, job), merged)


def read_status(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def summary_cell(value: object) -> object:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def refresh_summary(status_dir: Path) -> None:
    rows = [read_status(path) for path in status_dir.glob("*.json")]
    rows = [row for row in rows if row]
    rows.sort(key=lambda row: int(row.get("index", 0)))
    for suffix, delimiter in (("csv", ","), ("tsv", "\t")):
        summary_path = status_dir / f"summary.{suffix}"
        temporary = status_dir / f"summary.{suffix}.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, delimiter=delimiter)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {field: summary_cell(row.get(field, "")) for field in SUMMARY_FIELDS}
                )
        temporary.replace(summary_path)


def parse_core_clock_mhz(config_path: Path | None) -> float | None:
    if config_path is None or not config_path.is_file():
        return None
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
    if total_cycle is None:
        summary_matches = re.findall(r"\bcycles=([0-9]+)", text)
        total_cycle = int(summary_matches[-1]) if summary_matches else None
    return sim_cycle, total_cycle


def proc_tree_pids(root_pid: int) -> list[int]:
    parent: dict[int, int] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = stat_path.read_text(errors="replace")
            close = text.rfind(")")
            fields = text[close + 2 :].split()
            parent[int(stat_path.parent.name)] = int(fields[1])
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
        try:
            lines = (Path("/proc") / str(pid) / "status").read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            if line.startswith("VmRSS:"):
                try:
                    total_kb += int(line.split()[1])
                except (ValueError, IndexError):
                    pass
                break
    return total_kb / 1024.0


def send_process_group_signal(proc: subprocess.Popen[bytes], signum: int) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signum)
    except (ProcessLookupError, PermissionError):
        pass


def terminate_process_group(
    proc: subprocess.Popen[bytes], grace_seconds: float = 10.0
) -> None:
    send_process_group_signal(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        send_process_group_signal(proc, signal.SIGKILL)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


class CancellationController:
    def __init__(self) -> None:
        self.event = threading.Event()
        self._lock = threading.RLock()
        self._active: dict[int, subprocess.Popen[bytes]] = {}
        self._signal_number: int | None = None
        self._request_count = 0

    @property
    def signal_number(self) -> int | None:
        return self._signal_number

    @property
    def exit_code(self) -> int:
        return 128 + (self._signal_number or signal.SIGINT)

    def register(self, proc: subprocess.Popen[bytes]) -> None:
        with self._lock:
            self._active[proc.pid] = proc
            cancelled = self.event.is_set()
            force = self._request_count > 1
        if cancelled:
            send_process_group_signal(proc, signal.SIGKILL if force else signal.SIGTERM)

    def unregister(self, proc: subprocess.Popen[bytes]) -> None:
        with self._lock:
            self._active.pop(proc.pid, None)

    def request(self, signum: int) -> None:
        with self._lock:
            first_request = not self.event.is_set()
            if first_request:
                self._signal_number = signum
            self._request_count += 1
            self.event.set()
            active = tuple(self._active.values())
        forwarded = signal.SIGTERM if first_request else signal.SIGKILL
        message = (
            f"\n[{timestamp()}] INTERRUPT signal={signal.Signals(signum).name} "
            f"active={len(active)} action={signal.Signals(forwarded).name}\n"
        )
        try:
            os.write(2, message.encode())
        except OSError:
            pass
        for proc in active:
            send_process_group_signal(proc, forwarded)


def make_base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.simulator_compatibility:
        root = str(args.root)
        if args.cuda_path is not None:
            cuda = str(args.cuda_path)
            env.update(
                {
                    "CUDA_INSTALL_PATH": cuda,
                    "CUDA_VERSION_NUMBER": args.cuda_version_number,
                    "PTXAS_CUDA_INSTALL_PATH": cuda,
                }
            )
            env["PATH"] = ":".join(
                part
                for part in (f"{root}/bin", f"{cuda}/bin", env.get("PATH", ""))
                if part
            )
            env["LD_LIBRARY_PATH"] = ":".join(
                part
                for part in (
                    f"{root}/lib/{args.gpgpusim_config}",
                    f"{cuda}/lib64",
                    env.get("LD_LIBRARY_PATH", ""),
                )
                if part
            )
        env.update(
            {
                "GPGPUSIM_ROOT": root,
                "GPGPUSIM_CONFIG": args.gpgpusim_config,
                "GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN": "1",
                "GPGPUSIM_POWER_MODEL": f"{root}/src/accelwattch/",
                "QTINC": env.get("QTINC", "/usr/include"),
                "GTEST_COLOR": "no",
            }
        )
        for key, value in (
            ("PTX_SIM_USE_PTX_FILE", args.ptx_sim_use_ptx_file),
            ("PTX_SIM_KERNELFILE", args.ptx_sim_kernelfile),
            ("CUOBJDUMP_SIM_FILE", args.cuobjdump_sim_file or "jj"),
        ):
            if value:
                env[key] = value
            else:
                env.pop(key, None)
    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise SystemExit("--env key must not be empty")
        env[key] = value
    return env


def copy_config(config_dir: Path | None, workdir: Path) -> float | None:
    if config_dir is None:
        return None
    for path in config_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, workdir / path.name)
    return parse_core_clock_mhz(config_dir / "gpgpusim.config")


def job_paths(args: argparse.Namespace, job: Job) -> tuple[Path, Path, Path]:
    stem = f"{job.index:03d}_{job.job_id}"
    queue_workdir = args.run_root / "work" / stem
    log_path = args.run_root / "logs" / f"{stem}.log"
    execution_cwd = job.cwd or queue_workdir
    return queue_workdir, log_path, execution_cwd


def job_command(job: Job, cpu_set: str | None) -> list[str]:
    command = [str(job.executable)]
    if job.gtest_filter:
        command.append(f"--gtest_filter={job.gtest_filter}")
    command.extend(job.args)
    return ["taskset", "-c", cpu_set, *command] if cpu_set else command


def run_job(
    args: argparse.Namespace,
    job: Job,
    slot: int,
    cpu_set: str | None,
    base_env: dict[str, str],
    summary_lock: threading.Lock,
    cancellation: CancellationController,
) -> int:
    status_dir = args.run_root / "status"
    queue_workdir, log_path, execution_cwd = job_paths(args, job)
    existing_path = status_path(status_dir, job)
    if args.resume and existing_path.exists():
        if read_status(existing_path).get("state") == "done":
            return 0

    if not job.executable.exists():
        write_status(
            status_dir,
            job,
            {
                "state": "missing_executable",
                "rc": 127,
                "slot": slot,
                "cpu_set": cpu_set or "",
                "threads_per_job": args.threads_per_job,
                "log": str(log_path),
                "workdir": str(execution_cwd),
            },
        )
        with summary_lock:
            refresh_summary(status_dir)
        return 127

    if queue_workdir.exists():
        shutil.rmtree(queue_workdir)
    queue_workdir.mkdir(parents=True)
    if job.cwd is not None:
        execution_cwd.mkdir(parents=True, exist_ok=True)

    config_name = job.legacy_config or args.config
    if not config_name and job.legacy_manifest:
        config_name = "SM90_H100"
    config_dir = args.root / "configs" / config_name if config_name else None
    if config_dir is not None and not config_dir.is_dir():
        write_status(
            status_dir,
            job,
            {
                "state": "missing_config",
                "rc": 126,
                "slot": slot,
                "cpu_set": cpu_set or "",
                "threads_per_job": args.threads_per_job,
                "log": str(log_path),
                "workdir": str(execution_cwd),
            },
        )
        with summary_lock:
            refresh_summary(status_dir)
        return 126
    core_clock_mhz = copy_config(config_dir, queue_workdir)

    command = job_command(job, cpu_set)
    env = base_env.copy()
    env.update(job.env)
    env.update(
        {
            "OMP_NUM_THREADS": str(args.threads_per_job),
            "OPENBLAS_NUM_THREADS": str(args.threads_per_job),
            "MKL_NUM_THREADS": str(args.threads_per_job),
            "NUMEXPR_NUM_THREADS": str(args.threads_per_job),
            "RUN_QUEUE_ROOT": str(args.run_root),
            "RUN_QUEUE_JOB_ID": job.job_id,
            "RUN_QUEUE_WORKDIR": str(queue_workdir),
            "RUN_QUEUE_LOG": str(log_path),
            "RUN_QUEUE_SLOT": str(slot),
            "RUN_QUEUE_CPU_SET": cpu_set or "",
            "RUN_QUEUE_AFFINITY_MODE": args.affinity_mode,
            "RUN_QUEUE_CPUS_PER_JOB": str(args.cpus_per_job),
            "RUN_QUEUE_THREADS_PER_JOB": str(args.threads_per_job),
        }
    )
    timeout_seconds = job.timeout if job.timeout is not None else args.timeout
    max_rss_limit_mb = args.max_rss_gb * 1024.0 if args.max_rss_gb > 0 else 0.0
    start = time.time()
    running_status = {
        "state": "running",
        "rc": "",
        "elapsed_sec": 0,
        "slot": slot,
        "cpu_set": cpu_set or "",
        "threads_per_job": args.threads_per_job,
        "log": str(log_path),
        "workdir": str(execution_cwd),
    }
    write_status(status_dir, job, running_status)
    with summary_lock:
        refresh_summary(status_dir)

    rc = 1
    final_state = "failed"
    max_rss_mb = 0.0
    with log_path.open("w") as log:
        log.write(f"[{timestamp()}] START job_id={job.job_id}\n")
        log.write(f"[{timestamp()}] workdir={execution_cwd}\n")
        log.write(f"[{timestamp()}] cpu_set={cpu_set or ''}\n")
        log.write(f"[{timestamp()}] threads_per_job={args.threads_per_job}\n")
        log.write(
            f"[{timestamp()}] command="
            f"{' '.join(shlex.quote(part) for part in command)}\n"
        )
        if config_name:
            log.write(f"[{timestamp()}] config={config_name}\n")
        log.flush()
        try:
            proc = subprocess.Popen(
                command,
                cwd=execution_cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            rc = 127 if exc.errno == errno.ENOENT else 126
            final_state = "spawn_failed"
            log.write(f"[{timestamp()}] SPAWN_FAILED rc={rc} error={exc}\n")
            log.flush()
        else:
            cancellation.register(proc)
            deadline = start + timeout_seconds if timeout_seconds > 0 else None
            try:
                while True:
                    current_rc = proc.poll()
                    pids = proc_tree_pids(proc.pid) if current_rc is None else [proc.pid]
                    rss_mb = rss_mb_for_pids(pids)
                    max_rss_mb = max(max_rss_mb, rss_mb)
                    elapsed_now = int(time.time() - start)

                    if cancellation.event.is_set() and current_rc is None:
                        rc = cancellation.exit_code
                        final_state = "interrupted"
                        log.write(f"\n[{timestamp()}] INTERRUPTED rc={rc}\n")
                        log.flush()
                        terminate_process_group(proc)
                        break
                    if current_rc is not None:
                        rc = current_rc
                        final_state = "done" if rc == 0 else "failed"
                        break
                    if deadline is not None and time.time() >= deadline:
                        rc = 124
                        final_state = "timed_out"
                        log.write(f"\n[{timestamp()}] TIMEOUT after {timeout_seconds}s\n")
                        log.flush()
                        terminate_process_group(proc)
                        break
                    if max_rss_limit_mb and rss_mb > max_rss_limit_mb:
                        rc = 137
                        final_state = "rss_limit"
                        log.write(
                            f"\n[{timestamp()}] RSS_LIMIT rss_mb={rss_mb:.1f} "
                            f"limit_mb={max_rss_limit_mb:.1f}\n"
                        )
                        log.flush()
                        terminate_process_group(proc)
                        break

                    write_status(
                        status_dir,
                        job,
                        {
                            **running_status,
                            "elapsed_sec": elapsed_now,
                            "rss_mb": f"{rss_mb:.1f}",
                            "max_rss_mb": f"{max_rss_mb:.1f}",
                        },
                    )
                    with summary_lock:
                        refresh_summary(status_dir)
                    cancellation.event.wait(args.heartbeat_interval)
            finally:
                cancellation.unregister(proc)

    sim_cycle, total_cycle = parse_cycles(log_path) if final_state != "interrupted" else (None, None)
    selected_cycle = total_cycle if total_cycle is not None else sim_cycle
    metrics: dict[str, object] = {}
    if sim_cycle is not None:
        metrics["gpu_sim_cycle"] = sim_cycle
    if total_cycle is not None:
        metrics["gpu_tot_sim_cycle"] = total_cycle
    if selected_cycle is not None and core_clock_mhz:
        metrics["sim_time_us"] = selected_cycle / core_clock_mhz
    write_status(
        status_dir,
        job,
        {
            "state": final_state,
            "rc": rc,
            "elapsed_sec": int(time.time() - start),
            "slot": slot,
            "cpu_set": cpu_set or "",
            "threads_per_job": args.threads_per_job,
            "rss_mb": "",
            "max_rss_mb": f"{max_rss_mb:.1f}",
            "log": str(log_path),
            "workdir": str(execution_cwd),
            "metrics": metrics,
        },
    )
    with summary_lock:
        refresh_summary(status_dir)
    return rc


def worker(
    args: argparse.Namespace,
    slot: int,
    jobs: "queue.Queue[Job]",
    base_env: dict[str, str],
    summary_lock: threading.Lock,
    rc_by_slot: list[int],
    cancellation: CancellationController,
) -> None:
    cpu_set = args.cpu_sets[slot] if slot < len(args.cpu_sets) else None
    rc_any = 0
    while not cancellation.event.is_set():
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            break
        if cancellation.event.is_set():
            jobs.task_done()
            break
        print(
            f"[{timestamp()}] START slot={slot} cpu={cpu_set or ''} "
            f"job={job.job_id}",
            flush=True,
        )
        rc = run_job(
            args,
            job,
            slot,
            cpu_set,
            base_env,
            summary_lock,
            cancellation,
        )
        rc_any = rc_any or rc
        label = "DONE" if rc == 0 else ("INTERRUPTED" if cancellation.event.is_set() else "FAIL")
        print(f"[{timestamp()}] {label} slot={slot} rc={rc} job={job.job_id}", flush=True)
        jobs.task_done()
    rc_by_slot[slot] = rc_any


def mark_pending_jobs_interrupted(
    args: argparse.Namespace, jobs: Iterable[Job], cancellation: CancellationController
) -> None:
    status_dir = args.run_root / "status"
    for job in jobs:
        path = status_path(status_dir, job)
        existing = read_status(path) if path.exists() else base_status(job)
        if existing.get("state") not in {"queued", "running"}:
            continue
        existing.update({"state": "interrupted", "rc": cancellation.exit_code})
        write_status(status_dir, job, existing)


def prepare_run_root(args: argparse.Namespace) -> None:
    queue_path = args.run_root / "queue.json"
    known_paths = [args.run_root / name for name in ("logs", "status", "work")]
    if args.overwrite:
        for path in known_paths:
            if path.exists():
                shutil.rmtree(path)
        for path in (queue_path, args.run_root / "metadata.txt"):
            if path.exists():
                path.unlink()
    elif not args.resume and (queue_path.exists() or any(path.exists() for path in known_paths)):
        raise SystemExit(
            f"run root already contains queue state: {args.run_root}; "
            "use --resume or --overwrite"
        )
    args.run_root.mkdir(parents=True, exist_ok=True)
    for path in known_paths:
        path.mkdir(parents=True, exist_ok=True)


def write_queue_metadata(
    args: argparse.Namespace,
    jobs: list[Job],
    jobs_source: str,
    jobs_format: str,
    state: str,
    rc: int | None = None,
) -> None:
    metadata = {
        "created_at": args.created_at,
        "updated_at": timestamp(),
        "state": state,
        "rc": rc,
        "root": str(args.root),
        "run_root": str(args.run_root),
        "jobs_source": jobs_source,
        "jobs_format": jobs_format,
        "job_count": len(jobs),
        "enabled_job_count": sum(job.enabled for job in jobs),
        "runnable_job_count": args.runnable_job_count,
        "requested_workers": args.max_parallel,
        "effective_workers": args.worker_count,
        "affinity_mode": args.affinity_mode,
        "allowed_cpus": allowed_cpus(),
        "cpus_per_job": args.cpus_per_job,
        "threads_per_job": args.threads_per_job,
        "cpu_sets": args.cpu_sets,
        "timeout": args.timeout,
        "max_rss_gb": args.max_rss_gb,
        "simulator_compatibility": args.simulator_compatibility,
        "config": args.config,
    }
    write_json_atomic(args.run_root / "queue.json", metadata)
    (args.run_root / "metadata.txt").write_text(
        "\n".join(
            f"{key}={json.dumps(value) if isinstance(value, (list, dict)) else value}"
            for key, value in metadata.items()
        )
        + "\n"
    )


def resolve_resources(args: argparse.Namespace, enabled_jobs: int) -> None:
    requested_workers = min(args.max_parallel, enabled_jobs)
    if requested_workers == 0:
        requested_mode = (
            "manual" if args.cpu_sets else ("unpinned" if args.no_pin else "auto")
        )
        args.worker_count = 0
        args.cpu_sets = []
        args.cpus_per_job = args.cpus_per_job or 4
        args.threads_per_job = args.threads_per_job or args.cpus_per_job
        args.affinity_mode = requested_mode
        return

    if args.cpu_sets:
        if shutil.which("taskset") is None:
            raise SystemExit("taskset is required for --cpu-sets; use --no-pin")
        try:
            normalized = validate_cpu_sets(args.cpu_sets)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        widths = [len(parse_cpu_list(value)) for value in normalized]
        if len(set(widths)) != 1:
            raise SystemExit("all explicit --cpu-sets must contain the same number of CPUs")
        inferred_width = widths[0]
        if args.cpus_per_job is not None and args.cpus_per_job != inferred_width:
            raise SystemExit(
                f"--cpus-per-job={args.cpus_per_job} does not match explicit "
                f"CPU-set width {inferred_width}"
            )
        args.cpus_per_job = inferred_width
        args.cpu_sets = normalized[:requested_workers]
        args.worker_count = min(requested_workers, len(args.cpu_sets))
        args.cpu_sets = args.cpu_sets[: args.worker_count]
        args.affinity_mode = "manual"
    elif args.no_pin:
        args.cpus_per_job = args.cpus_per_job or 4
        args.worker_count = requested_workers
        args.cpu_sets = []
        args.affinity_mode = "unpinned"
    else:
        args.cpus_per_job = args.cpus_per_job or 4
        if shutil.which("taskset") is None:
            raise SystemExit("taskset is required for automatic CPU affinity; use --no-pin")
        try:
            args.cpu_sets = select_cpu_sets(
                requested_workers,
                args.cpus_per_job,
                sample_seconds=args.cpu_sample_seconds,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        args.worker_count = len(args.cpu_sets)
        args.affinity_mode = "auto"
    args.threads_per_job = args.threads_per_job or args.cpus_per_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Base for relative executable/cwd paths (default: repository root)",
    )
    parser.add_argument("--run-root", type=Path, help="Dedicated queue output directory")
    parser.add_argument("--jobs", help="CSV/TSV job manifest path, or - for stdin")
    parser.add_argument(
        "--jobs-format",
        choices=("auto", "csv", "tsv"),
        default="auto",
        help="Manifest format (default: infer from suffix/content)",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=4, help="Maximum workers (default: 4)"
    )
    parser.add_argument(
        "--cpus-per-job",
        type=int,
        help="Physical cores per automatic worker (default: 4)",
    )
    parser.add_argument(
        "--threads-per-job",
        type=int,
        help="OMP/BLAS thread limit (default: cpus-per-job)",
    )
    affinity = parser.add_mutually_exclusive_group()
    affinity.add_argument(
        "--cpu-sets",
        nargs="+",
        default=[],
        metavar="LIST",
        help="Manual taskset list per worker; default is automatic selection",
    )
    affinity.add_argument(
        "--no-pin",
        action="store_true",
        help="Do not apply taskset; workers inherit all allowed CPUs",
    )
    parser.add_argument("--cpu-sample-seconds", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--heartbeat-interval", type=float, default=30.0)
    parser.add_argument("--max-rss-gb", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument(
        "--config",
        help=(
            "Optional queue-wide GPGPU-Sim config; enables simulator compatibility "
            "(legacy manifests default to SM90_H100)"
        ),
    )
    parser.add_argument(
        "--cuda-path",
        type=Path,
        default=Path(os.environ["CUDA_INSTALL_PATH"])
        if os.environ.get("CUDA_INSTALL_PATH")
        else None,
    )
    parser.add_argument(
        "--cuda-version-number", default=os.environ.get("CUDA_VERSION_NUMBER", "12080")
    )
    parser.add_argument(
        "--gpgpusim-config",
        default=os.environ.get(
            "GPGPUSIM_CONFIG", "gcc-13.3.0/cuda-12080/release"
        ),
    )
    parser.add_argument("--ptx-sim-use-ptx-file")
    parser.add_argument("--ptx-sim-kernelfile")
    parser.add_argument("--cuobjdump-sim-file")
    parser.add_argument("--print-example-jobs", action="store_true")
    args = parser.parse_args()

    if args.print_example_jobs:
        print_example_jobs()
        raise SystemExit(0)
    if args.run_root is None:
        raise SystemExit("--run-root is required")
    if args.jobs is None:
        raise SystemExit("--jobs is required")
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be >= 1")
    for name in ("cpus_per_job", "threads_per_job"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")
    if args.timeout < 0:
        raise SystemExit("--timeout must be >= 0")
    if args.heartbeat_interval <= 0:
        raise SystemExit("--heartbeat-interval must be > 0")
    if args.cpu_sample_seconds < 0:
        raise SystemExit("--cpu-sample-seconds must be >= 0")
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    args.root = args.root.resolve()
    args.run_root = args.run_root.resolve()
    unsafe_run_roots = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path(tempfile.gettempdir()).resolve(),
        args.root,
    }
    if args.run_root in unsafe_run_roots:
        raise SystemExit(
            f"refusing broad --run-root {args.run_root}; choose a dedicated subdirectory"
        )
    if args.cuda_path is not None:
        args.cuda_path = args.cuda_path.resolve()
    return args


def main() -> int:
    args = parse_args()
    jobs, jobs_source, jobs_format = read_jobs(args.jobs, args.jobs_format, args.root)
    args.simulator_compatibility = bool(
        args.config
        or any(job.legacy_manifest for job in jobs)
        or args.ptx_sim_use_ptx_file
        or args.ptx_sim_kernelfile
        or args.cuobjdump_sim_file
    )
    enabled_jobs = [job for job in jobs if job.enabled]
    status_dir = args.run_root / "status"
    runnable_jobs = [
        job
        for job in enabled_jobs
        if not (
            args.resume
            and status_path(status_dir, job).exists()
            and read_status(status_path(status_dir, job)).get("state") == "done"
        )
    ]
    args.runnable_job_count = len(runnable_jobs)
    resolve_resources(args, len(runnable_jobs))
    args.created_at = timestamp()
    prepare_run_root(args)

    for job in jobs:
        _, log_path, execution_cwd = job_paths(args, job)
        existing_path = status_path(status_dir, job)
        if args.resume and existing_path.exists():
            if read_status(existing_path).get("state") == "done":
                continue
        initial = base_status(job)
        initial.update(
            {
                "threads_per_job": args.threads_per_job,
                "log": str(log_path),
                "workdir": str(execution_cwd),
            }
        )
        write_status(status_dir, job, initial)
    refresh_summary(status_dir)
    write_queue_metadata(args, jobs, jobs_source, jobs_format, "running")

    if args.worker_count == 0:
        write_queue_metadata(args, jobs, jobs_source, jobs_format, "complete", 0)
        (status_dir / "complete").write_text(
            f"rc=0\nstate=complete\nsignal=\ncompleted_at={timestamp()}\n"
        )
        print(f"[{timestamp()}] ALL_DONE rc=0 summary={status_dir / 'summary.tsv'}")
        return 0

    job_queue: "queue.Queue[Job]" = queue.Queue()
    for job in runnable_jobs:
        job_queue.put(job)

    base_env = make_base_env(args)
    summary_lock = threading.Lock()
    cancellation = CancellationController()
    rc_by_slot = [0 for _ in range(args.worker_count)]
    threads = [
        threading.Thread(
            target=worker,
            args=(
                args,
                slot,
                job_queue,
                base_env,
                summary_lock,
                rc_by_slot,
                cancellation,
            ),
        )
        for slot in range(args.worker_count)
    ]

    def handle_signal(signum: int, _frame: object) -> None:
        cancellation.request(signum)

    previous_handlers = {
        signum: signal.signal(signum, handle_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    print(f"[{timestamp()}] RUN_ROOT={args.run_root}", flush=True)
    print(
        f"[{timestamp()}] jobs={len(runnable_jobs)} workers={args.worker_count} "
        f"affinity={args.affinity_mode} cpus_per_job={args.cpus_per_job} "
        f"threads_per_job={args.threads_per_job}",
        flush=True,
    )
    print(f"[{timestamp()}] cpu_sets={' '.join(args.cpu_sets)}", flush=True)
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)

    if cancellation.event.is_set():
        mark_pending_jobs_interrupted(args, jobs, cancellation)
    refresh_summary(status_dir)
    if cancellation.event.is_set():
        rc = cancellation.exit_code
        state = "interrupted"
    else:
        rc = 0 if all(code == 0 for code in rc_by_slot) else 1
        state = "complete"
    write_queue_metadata(args, jobs, jobs_source, jobs_format, state, rc)
    signal_name = (
        signal.Signals(cancellation.signal_number).name
        if cancellation.signal_number is not None
        else ""
    )
    (status_dir / "complete").write_text(
        f"rc={rc}\nstate={state}\nsignal={signal_name}\ncompleted_at={timestamp()}\n"
    )
    label = "INTERRUPTED" if cancellation.event.is_set() else "ALL_DONE"
    print(f"[{timestamp()}] {label} rc={rc} summary={status_dir / 'summary.tsv'}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
