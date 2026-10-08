#!/usr/bin/env python3
"""Select disjoint, topology-aware CPU sets for worker processes."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SYS_CPU_ROOT = Path("/sys/devices/system/cpu")


@dataclass(frozen=True)
class PhysicalCore:
    package_id: str
    core_id: str
    siblings: tuple[int, ...]
    representative: int
    capacity: int
    busy: float


def allowed_cpus() -> list[int]:
    """Return the logical CPUs available to the calling process."""
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return list(range(os.cpu_count() or 1))


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def parse_cpu_list(value: str) -> list[int]:
    """Parse Linux taskset/list syntax such as ``0,2,4-7``."""
    cpus: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"invalid empty CPU-list component in {value!r}")
        if "-" in part:
            first_text, last_text = part.split("-", 1)
            first = int(first_text)
            last = int(last_text)
            if first < 0 or last < first:
                raise ValueError(f"invalid CPU range {part!r}")
            cpus.update(range(first, last + 1))
        else:
            cpu = int(part)
            if cpu < 0:
                raise ValueError(f"invalid CPU {cpu}")
            cpus.add(cpu)
    return sorted(cpus)


def format_cpu_list(cpus: Iterable[int]) -> str:
    """Format CPUs as a stable taskset-compatible comma-separated list."""
    return ",".join(str(cpu) for cpu in sorted(set(cpus)))


def validate_cpu_sets(
    cpu_sets: Iterable[str], *, require_disjoint: bool = True
) -> list[str]:
    """Validate explicit CPU sets against the caller's allowed affinity."""
    allowed = set(allowed_cpus())
    used: set[int] = set()
    normalized: list[str] = []
    for index, value in enumerate(cpu_sets):
        try:
            cpus = parse_cpu_list(value)
        except ValueError as exc:
            raise ValueError(f"CPU set {index}: {exc}") from exc
        if not cpus:
            raise ValueError(f"CPU set {index} is empty")
        outside = set(cpus) - allowed
        if outside:
            raise ValueError(
                f"CPU set {index} contains disallowed CPUs: {format_cpu_list(outside)}"
            )
        overlap = set(cpus) & used
        if require_disjoint and overlap:
            raise ValueError(
                f"CPU set {index} overlaps an earlier set on CPUs: "
                f"{format_cpu_list(overlap)}"
            )
        used.update(cpus)
        normalized.append(format_cpu_list(cpus))
    return normalized


def _read_cpu_times() -> dict[int, tuple[int, int]]:
    """Return logical CPU cumulative ``(busy, total)`` scheduler ticks."""
    result: dict[int, tuple[int, int]] = {}
    try:
        lines = Path("/proc/stat").read_text().splitlines()
    except OSError:
        return result
    for line in lines:
        fields = line.split()
        if not fields or not fields[0].startswith("cpu") or fields[0] == "cpu":
            continue
        suffix = fields[0][3:]
        if not suffix.isdigit():
            continue
        try:
            values = [int(value) for value in fields[1:]]
        except ValueError:
            continue
        total = sum(values)
        idle = (values[3] if len(values) > 3 else 0) + (
            values[4] if len(values) > 4 else 0
        )
        result[int(suffix)] = (total - idle, total)
    return result


def sample_cpu_busy(sample_seconds: float = 0.2) -> dict[int, float]:
    """Sample per-logical-CPU utilization in the inclusive range [0, 1]."""
    before = _read_cpu_times()
    if sample_seconds > 0:
        time.sleep(sample_seconds)
    after = _read_cpu_times()
    busy: dict[int, float] = {}
    for cpu, (busy_after, total_after) in after.items():
        busy_before, total_before = before.get(cpu, (busy_after, total_after))
        total_delta = total_after - total_before
        busy_delta = busy_after - busy_before
        busy[cpu] = (
            max(0.0, min(1.0, busy_delta / total_delta))
            if total_delta > 0
            else 0.0
        )
    return busy


def _cpu_capacity(cpu: int) -> int:
    cpu_dir = SYS_CPU_ROOT / f"cpu{cpu}"
    for relative in (
        "cpufreq/cpuinfo_max_freq",
        "cpufreq/scaling_max_freq",
        "cpu_capacity",
    ):
        value = _read_text(cpu_dir / relative)
        if value and value.isdigit():
            return int(value)
    return 0


def discover_physical_cores(
    cpus: Iterable[int] | None = None, *, sample_seconds: float = 0.2
) -> list[PhysicalCore]:
    """Collapse allowed logical CPUs into physical cores."""
    selected_cpus = sorted(set(cpus if cpus is not None else allowed_cpus()))
    logical_busy = sample_cpu_busy(sample_seconds)
    grouped: dict[tuple[str, str], list[int]] = {}
    for cpu in selected_cpus:
        topology = SYS_CPU_ROOT / f"cpu{cpu}" / "topology"
        package_id = _read_text(topology / "physical_package_id") or "0"
        core_id = _read_text(topology / "core_id") or str(cpu)
        grouped.setdefault((package_id, core_id), []).append(cpu)

    cores: list[PhysicalCore] = []
    for (package_id, core_id), siblings in grouped.items():
        representative = min(
            siblings, key=lambda cpu: (logical_busy.get(cpu, 0.0), cpu)
        )
        cores.append(
            PhysicalCore(
                package_id=package_id,
                core_id=core_id,
                siblings=tuple(sorted(siblings)),
                representative=representative,
                capacity=max((_cpu_capacity(cpu) for cpu in siblings), default=0),
                # One busy SMT sibling means that the physical core is busy.
                busy=max((logical_busy.get(cpu, 0.0) for cpu in siblings), default=0.0),
            )
        )
    return cores


def _capacity_tiers(cores: Iterable[PhysicalCore]) -> list[list[PhysicalCore]]:
    """Group cores with broadly similar capacity to avoid hybrid-core mixing."""
    ordered = sorted(cores, key=lambda core: (-core.capacity, core.representative))
    tiers: list[list[PhysicalCore]] = []
    for core in ordered:
        if not tiers:
            tiers.append([core])
            continue
        tier_capacity = max(item.capacity for item in tiers[-1])
        if tier_capacity == 0 or core.capacity >= tier_capacity * 0.85:
            tiers[-1].append(core)
        else:
            tiers.append([core])
    return tiers


def select_cpu_sets(
    max_workers: int,
    cpus_per_job: int,
    *,
    sample_seconds: float = 0.2,
    cpus: Iterable[int] | None = None,
) -> list[str]:
    """Return up to ``max_workers`` disjoint physical-core CPU sets."""
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if cpus_per_job < 1:
        raise ValueError("cpus_per_job must be >= 1")

    cores = discover_physical_cores(cpus, sample_seconds=sample_seconds)
    possible_workers = min(max_workers, len(cores) // cpus_per_job)
    if possible_workers == 0:
        raise ValueError(
            f"need {cpus_per_job} physical cores per job, but only "
            f"{len(cores)} are available"
        )

    candidates: list[list[PhysicalCore]] = []
    leftovers: list[PhysicalCore] = []
    for tier in _capacity_tiers(cores):
        tier.sort(key=lambda core: (core.busy, -core.capacity, core.representative))
        complete = len(tier) // cpus_per_job
        for index in range(complete):
            first = index * cpus_per_job
            candidates.append(tier[first : first + cpus_per_job])
        leftovers.extend(tier[complete * cpus_per_job :])

    leftovers.sort(key=lambda core: (core.busy, -core.capacity, core.representative))
    while len(leftovers) >= cpus_per_job:
        candidates.append(leftovers[:cpus_per_job])
        del leftovers[:cpus_per_job]

    def group_score(group: list[PhysicalCore]) -> tuple[float, float, float, int]:
        return (
            max(core.busy for core in group),
            sum(core.busy for core in group) / len(group),
            -(sum(core.capacity for core in group) / len(group)),
            min(core.representative for core in group),
        )

    candidates.sort(key=group_score)
    chosen = candidates[:possible_workers]
    return [
        format_cpu_list(core.representative for core in group) for group in chosen
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--cpus-per-job", type=int, default=4)
    parser.add_argument("--sample-seconds", type=float, default=0.2)
    parser.add_argument("--format", choices=("lines", "json"), default="lines")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        cpu_sets = select_cpu_sets(
            args.workers,
            args.cpus_per_job,
            sample_seconds=args.sample_seconds,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.format == "json":
        print(json.dumps(cpu_sets))
    else:
        print("\n".join(cpu_sets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
