#!/usr/bin/env python3
"""Render failed CI test results as a compact GitHub job summary."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET


ANSI_ESCAPE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))"
)
GTEST_RUN = re.compile(r"^\[\s*RUN\s*\]\s+(\S+)")
GTEST_FINISH = re.compile(r"^\[\s*(?:OK|FAILED|SKIPPED)\s*\]\s+(\S+)")
MAX_FAILURES = 20
MAX_DETAIL_LINES = 30
MAX_DETAIL_CHARS = 4096


@dataclass(frozen=True)
class CaseResult:
    group: str
    name: str
    result: str
    location: str
    detail: str


def _clean_detail(value: str) -> str:
    cleaned = ANSI_ESCAPE.sub("", value).replace("\r", "").strip()
    lines = cleaned.splitlines()
    truncated = len(lines) > MAX_DETAIL_LINES
    cleaned = "\n".join(lines[:MAX_DETAIL_LINES])
    if len(cleaned) > MAX_DETAIL_CHARS:
        cleaned = cleaned[:MAX_DETAIL_CHARS]
        truncated = True
    if truncated:
        cleaned += "\n... truncated; see the artifact for complete output."
    return cleaned


def _clean_log_tail(value: str) -> str:
    cleaned = ANSI_ESCAPE.sub("", value).replace("\r", "").strip()
    lines = cleaned.splitlines()
    truncated = len(lines) > MAX_DETAIL_LINES
    cleaned = "\n".join(lines[-MAX_DETAIL_LINES:])
    if len(cleaned) > MAX_DETAIL_CHARS:
        cleaned = cleaned[-MAX_DETAIL_CHARS:]
        truncated = True
    if truncated:
        cleaned = "... earlier output omitted; see the artifact.\n" + cleaned
    return cleaned


def _group_name(
    xml_file: Path,
    xml_root: Path,
    job: str,
    context: dict[str, str],
) -> str:
    relative = xml_file.relative_to(xml_root)
    label = relative.parts[0] if len(relative.parts) > 1 else xml_file.stem
    if label == context.get("phase") and context.get("group"):
        return context["group"]
    prefix = f"{job}-"
    return label[len(prefix) :] if label.startswith(prefix) else label


def _read_context(log_root: Path) -> dict[str, str]:
    context: dict[str, str] = {}
    path = log_root / "failure-context.txt"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return context
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            context[key] = value
    return context


def _collect_xml_failures(
    log_root: Path, job: str, context: dict[str, str]
) -> tuple[list[CaseResult], list[str]]:
    xml_root = log_root / "xml"
    failures: list[CaseResult] = []
    parse_errors: list[str] = []

    for xml_file in sorted(xml_root.glob("**/*.xml")):
        try:
            root = ET.parse(xml_file).getroot()
        except (ET.ParseError, OSError) as error:
            parse_errors.append(f"{xml_file.name}: {error}")
            continue

        group = _group_name(xml_file, xml_root, job, context)
        for testcase in root.iter("testcase"):
            problems = list(testcase.findall("failure"))
            problems.extend(testcase.findall("error"))
            if not problems:
                continue

            classname = testcase.get("classname", "").strip()
            case_name = testcase.get("name", "unknown").strip()
            full_name = f"{classname}.{case_name}" if classname else case_name
            source = testcase.get("file", "").strip()
            line = testcase.get("line", "").strip()
            location = f"{source}:{line}" if source and line else source
            details = []
            for problem in problems:
                text = (problem.text or problem.get("message", "")).strip()
                if text:
                    details.append(text)
            failures.append(
                CaseResult(
                    group=group,
                    name=full_name,
                    result="Failed",
                    location=location,
                    detail=_clean_detail("\n\n".join(details)),
                )
            )

    return failures, parse_errors


def _collect_aborted_case(
    log_root: Path, context: dict[str, str]
) -> CaseResult | None:
    phase = context.get("phase", "").strip()
    exit_code = context.get("exit_code", "").strip()
    if not phase or not exit_code or exit_code == "0":
        return None

    log_file = log_root / "logs" / f"{phase}.log"
    try:
        log_text = log_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    raw_lines = log_text.splitlines()
    active_case = ""
    active_line_index = 0
    for line_index, raw_line in enumerate(raw_lines):
        line = ANSI_ESCAPE.sub("", raw_line).strip()
        started = GTEST_RUN.match(line)
        if started:
            active_case = started.group(1)
            active_line_index = line_index
            continue
        finished = GTEST_FINISH.match(line)
        if finished and finished.group(1).rstrip(",") == active_case:
            active_case = ""

    if not active_case:
        return None

    return CaseResult(
        group=context.get("group", "").strip() or "—",
        name=active_case,
        result="Aborted",
        location="",
        detail=_clean_log_tail("\n".join(raw_lines[active_line_index:])),
    )


def _table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def render_failure_report(log_root: Path, job: str) -> str:
    context = _read_context(log_root)
    cases, parse_errors = _collect_xml_failures(log_root, job, context)
    aborted_case = _collect_aborted_case(log_root, context)
    if aborted_case and all(
        (case.group, case.name) != (aborted_case.group, aborted_case.name)
        for case in cases
    ):
        cases.append(aborted_case)
    phase = context.get("phase", "")
    exit_code = context.get("exit_code", "")
    lines: list[str] = []

    if not cases and phase:
        phase_text = f"Failed during `{phase}`"
        if exit_code:
            phase_text += f" with exit code `{exit_code}`"
        lines.extend([f"{phase_text}.", ""])

    if cases:
        lines.extend(
            [
                "| Group | Test | Result |",
                "|---|---|---|",
            ]
        )
        for case in cases[:MAX_FAILURES]:
            lines.append(
                f"| {_table_cell(case.group)} | "
                f"`{_table_cell(case.name)}` | {_table_cell(case.result)} |"
            )
        if len(cases) > MAX_FAILURES:
            lines.extend(
                [
                    "",
                    f"{len(cases) - MAX_FAILURES} additional failed or aborted "
                    "cases are available in the artifact.",
                ]
            )

        for case in cases[:MAX_FAILURES]:
            detail_parts = []
            if case.location:
                source = case.location.rsplit(":", 1)[0]
                if not case.detail or source not in case.detail:
                    detail_parts.append(case.location)
            if case.detail:
                detail_parts.append(case.detail)
            detail = "\n".join(detail_parts) or "No failure message was recorded."
            lines.extend(
                [
                    "",
                    "<details>",
                    f"<summary><code>{html.escape(case.name)}</code></summary>",
                    "",
                    f"<pre>{html.escape(detail)}</pre>",
                    "</details>",
                ]
            )
    else:
        lines.extend(
            [
                "No failed GoogleTest case was found in the available XML.",
                "The failure may have occurred during build, timeout, a "
                "post-check, or before XML output completed.",
            ]
        )

    if parse_errors:
        lines.extend(["", "Some XML files could not be parsed:"])
        for error in parse_errors[:5]:
            lines.append(f"- `{_table_cell(error)}`")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_root", type=Path)
    parser.add_argument("--job", default=os.environ.get("CI_JOB", "run-tests"))
    args = parser.parse_args()
    print(
        render_failure_report(args.log_root, args.job),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
