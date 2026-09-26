#!/usr/bin/env python3
"""Validate and summarize structured CTA_LIFECYCLE trace events."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


EVENT_MARKER = "CTA_LIFECYCLE "
REQUIRED_FIELDS = {
    "admit": {
        "kernel_uid",
        "sid",
        "hw_cta",
        "logical_cta",
        "generation",
        "cycle",
        "slot_reuse",
        "previous_release",
        "release_to_admit",
    },
    "threads_exit": {
        "kernel_uid",
        "sid",
        "hw_cta",
        "logical_cta",
        "generation",
        "cycle",
        "pending_tma",
    },
    "release": {
        "kernel_uid",
        "sid",
        "hw_cta",
        "logical_cta",
        "generation",
        "cycle",
        "active_cycles",
        "threads_exit",
        "exit_to_release",
        "pending_tma",
    },
}


class LifecycleError(ValueError):
    pass


@dataclass(frozen=True)
class Event:
    name: str
    fields: dict[str, int]
    line_number: int

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (
            self.fields["kernel_uid"],
            self.fields["sid"],
            self.fields["hw_cta"],
            self.fields["generation"],
        )


@dataclass(frozen=True)
class Lifecycle:
    admit: Event
    threads_exit: Event
    release: Event


def parse_event(line: str, line_number: int) -> Event | None:
    marker = line.find(EVENT_MARKER)
    if marker < 0:
        return None
    fields_text = line[marker + len(EVENT_MARKER) :].strip()
    raw_fields: dict[str, str] = {}
    for token in fields_text.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        raw_fields[key] = value
    name = raw_fields.pop("event", None)
    if name not in REQUIRED_FIELDS:
        raise LifecycleError(f"line {line_number}: unsupported event {name!r}")
    missing = REQUIRED_FIELDS[name] - raw_fields.keys()
    if missing:
        raise LifecycleError(
            f"line {line_number}: {name} missing fields {sorted(missing)}"
        )
    try:
        fields = {key: int(value, 0) for key, value in raw_fields.items()}
    except ValueError as error:
        raise LifecycleError(
            f"line {line_number}: non-integer lifecycle field"
        ) from error
    if name == "admit":
        if fields["slot_reuse"] not in (0, 1):
            raise LifecycleError(f"line {line_number}: invalid admit boolean")
        if fields["slot_reuse"]:
            if fields["cycle"] < fields["previous_release"]:
                raise LifecycleError(
                    f"line {line_number}: admit precedes previous release"
                )
            if fields["release_to_admit"] != (
                fields["cycle"] - fields["previous_release"]
            ):
                raise LifecycleError(
                    f"line {line_number}: release_to_admit mismatch"
                )
        elif fields["previous_release"] != 0 or fields["release_to_admit"] != 0:
            raise LifecycleError(
                f"line {line_number}: unused slot has prior release timing"
            )
    return Event(name=name, fields=fields, line_number=line_number)


def read_lifecycles(lines: Iterable[str]) -> list[Lifecycle]:
    active: dict[tuple[int, int, int, int], dict[str, Event]] = {}
    completed: list[Lifecycle] = []
    event_count = 0
    for line_number, line in enumerate(lines, start=1):
        event = parse_event(line, line_number)
        if event is None:
            continue
        event_count += 1
        key = event.key
        if event.name == "admit":
            if key in active:
                raise LifecycleError(f"line {line_number}: duplicate admit for {key}")
            active[key] = {"admit": event}
            continue
        if key not in active:
            raise LifecycleError(
                f"line {line_number}: {event.name} without admit for {key}"
            )
        current = active[key]
        if event.name == "threads_exit":
            if "threads_exit" in current:
                raise LifecycleError(
                    f"line {line_number}: duplicate threads_exit for {key}"
                )
            current["threads_exit"] = event
            continue
        if "threads_exit" not in current:
            raise LifecycleError(
                f"line {line_number}: release precedes threads_exit for {key}"
            )
        admit = current["admit"]
        threads_exit = current["threads_exit"]
        logical_ids = {
            admit.fields["logical_cta"],
            threads_exit.fields["logical_cta"],
            event.fields["logical_cta"],
        }
        if len(logical_ids) != 1:
            raise LifecycleError(
                f"line {line_number}: logical CTA changed within lifecycle {key}"
            )
        if not (
            admit.fields["cycle"]
            <= threads_exit.fields["cycle"]
            <= event.fields["cycle"]
        ):
            raise LifecycleError(
                f"line {line_number}: non-monotonic lifecycle timestamps for {key}"
            )
        if event.fields["active_cycles"] != (
            event.fields["cycle"] - admit.fields["cycle"]
        ):
            raise LifecycleError(
                f"line {line_number}: active_cycles mismatch for {key}"
            )
        if event.fields["threads_exit"] != threads_exit.fields["cycle"]:
            raise LifecycleError(
                f"line {line_number}: threads_exit timestamp mismatch for {key}"
            )
        if event.fields["exit_to_release"] != (
            event.fields["cycle"] - threads_exit.fields["cycle"]
        ):
            raise LifecycleError(
                f"line {line_number}: exit_to_release mismatch for {key}"
            )
        if event.fields["pending_tma"] != threads_exit.fields["pending_tma"]:
            raise LifecycleError(
                f"line {line_number}: pending_tma changed for {key}"
            )
        completed.append(Lifecycle(admit, threads_exit, event))
        del active[key]
    if event_count == 0:
        raise LifecycleError("no CTA_LIFECYCLE events found")
    if active:
        sample = next(iter(active))
        raise LifecycleError(
            f"{len(active)} admitted CTA lifecycles were not released; first={sample}"
        )

    by_slot: dict[tuple[int, int], list[Lifecycle]] = defaultdict(list)
    for lifecycle in completed:
        by_slot[
            (lifecycle.admit.fields["sid"], lifecycle.admit.fields["hw_cta"])
        ].append(lifecycle)
    for slot, slot_lifecycles in by_slot.items():
        slot_lifecycles.sort(key=lambda lifecycle: lifecycle.admit.fields["generation"])
        first_admit = slot_lifecycles[0].admit
        if first_admit.fields["generation"] != 1:
            raise LifecycleError(f"slot {slot}: first generation is not 1")
        if first_admit.fields["slot_reuse"]:
            raise LifecycleError(f"slot {slot}: first admission is marked reused")
        for previous, current in zip(slot_lifecycles, slot_lifecycles[1:]):
            previous_generation = previous.admit.fields["generation"]
            current_generation = current.admit.fields["generation"]
            if current_generation != previous_generation + 1:
                raise LifecycleError(
                    f"slot {slot}: non-consecutive generations "
                    f"{previous_generation}->{current_generation}"
                )
            if not current.admit.fields["slot_reuse"]:
                raise LifecycleError(f"slot {slot}: replacement is marked unused")
            if current.admit.fields["previous_release"] != previous.release.fields[
                "cycle"
            ]:
                raise LifecycleError(f"slot {slot}: previous release mismatch")
            replacement_ready = previous.release.fields.get("replacement_ready")
            if (
                replacement_ready is not None
                and current.admit.fields["cycle"] < replacement_ready
            ):
                raise LifecycleError(f"slot {slot}: replacement admitted before ready")
            replacement_latency = previous.release.fields.get("replacement_latency")
            if replacement_latency is not None and replacement_ready != (
                previous.release.fields["cycle"] + replacement_latency
            ):
                raise LifecycleError(f"slot {slot}: replacement ready mismatch")
    return completed


def distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "mean": 0.0, "max": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def summarize(lifecycles: list[Lifecycle]) -> dict[str, object]:
    by_kernel: dict[int, list[Lifecycle]] = defaultdict(list)
    for lifecycle in lifecycles:
        by_kernel[lifecycle.admit.fields["kernel_uid"]].append(lifecycle)

    kernels: dict[str, object] = {}
    for kernel_uid, records in sorted(by_kernel.items()):
        by_sm: dict[int, list[Lifecycle]] = defaultdict(list)
        for record in records:
            by_sm[record.admit.fields["sid"]].append(record)
        ctas_per_sm = [len(sm_records) for sm_records in by_sm.values()]
        active_spans = []
        last_releases = []
        for sm_records in by_sm.values():
            first_admit = min(r.admit.fields["cycle"] for r in sm_records)
            last_release = max(r.release.fields["cycle"] for r in sm_records)
            active_spans.append(last_release - first_admit)
            last_releases.append(last_release)
        replacement_records = [
            r for r in records if r.admit.fields["slot_reuse"]
        ]
        replacement_gaps = [
            r.admit.fields["release_to_admit"] for r in replacement_records
        ]
        pending_drains = [
            r.release.fields["exit_to_release"]
            for r in records
            if r.release.fields["pending_tma"]
        ]
        logical_ids = [r.admit.fields["logical_cta"] for r in records]
        duplicate_logical_ids = len(logical_ids) - len(set(logical_ids))
        kernels[str(kernel_uid)] = {
            "ctas": len(records),
            "sms": len(by_sm),
            "slot_initial_admits": sum(
                1 for r in records if not r.admit.fields["slot_reuse"]
            ),
            "slot_replacement_admits": len(replacement_records),
            "duplicate_logical_cta_ids": duplicate_logical_ids,
            "ctas_per_sm": distribution(ctas_per_sm),
            "cta_active_cycles": distribution(
                [r.release.fields["active_cycles"] for r in records]
            ),
            "release_to_admit": distribution(replacement_gaps),
            "pending_tma_drain": distribution(pending_drains),
            "sm_active_span": distribution(active_spans),
            "sm_last_release": distribution(last_releases),
            "sm_last_release_spread": (
                max(last_releases) - min(last_releases) if last_releases else 0
            ),
        }
    return {"lifecycles": len(lifecycles), "kernels": kernels}


def print_distribution(stream: TextIO, label: str, values: object) -> None:
    assert isinstance(values, dict)
    print(
        f"  {label}: count={values['count']} min={values['min']} "
        f"mean={values['mean']:.2f} max={values['max']}",
        file=stream,
    )


def print_summary(summary: dict[str, object], stream: TextIO = sys.stdout) -> None:
    kernels = summary["kernels"]
    assert isinstance(kernels, dict)
    for kernel_uid, values in kernels.items():
        assert isinstance(values, dict)
        print(
            f"kernel_uid={kernel_uid} ctas={values['ctas']} sms={values['sms']} "
            f"slot_initial={values['slot_initial_admits']} "
            f"slot_replacements={values['slot_replacement_admits']} "
            f"duplicate_logical_ids={values['duplicate_logical_cta_ids']}",
            file=stream,
        )
        print_distribution(stream, "ctas_per_sm", values["ctas_per_sm"])
        print_distribution(stream, "cta_active_cycles", values["cta_active_cycles"])
        print_distribution(stream, "release_to_admit", values["release_to_admit"])
        print_distribution(stream, "pending_tma_drain", values["pending_tma_drain"])
        print_distribution(stream, "sm_active_span", values["sm_active_span"])
        print_distribution(stream, "sm_last_release", values["sm_last_release"])
        print(
            f"  sm_last_release_spread: {values['sm_last_release_spread']}",
            file=stream,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="simulator case log")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()
    try:
        with args.log.open(encoding="utf-8", errors="replace") as stream:
            summary = summarize(read_lifecycles(stream))
    except (OSError, LifecycleError) as error:
        print(f"CTA lifecycle analysis failed: {error}", file=sys.stderr)
        return 2
    if args.json:
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
