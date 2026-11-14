#!/usr/bin/env python3

"""Run a declarative set of GTest filters through strict executable SASS."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
PASSED_COUNT = re.compile(r"\[  PASSED  \]\s+(\d+) tests?\.")


def load_suite(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        suite = json.load(source)
    if not isinstance(suite, dict):
        raise ValueError("suite root must be an object")
    for field in ("architecture", "group", "cases"):
        if field not in suite:
            raise ValueError(f"suite is missing {field!r}")
    if not isinstance(suite["cases"], list) or not suite["cases"]:
        raise ValueError("suite cases must be a non-empty array")
    return suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="JSON suite description")
    arguments = parser.parse_args()

    repository = Path(__file__).resolve().parent.parent
    suite_path = arguments.suite.expanduser().resolve()
    suite = load_suite(suite_path)
    architecture = suite["architecture"]
    group = suite["group"]
    suite_config = suite.get("config")
    suite_profile = suite.get("profile")
    suite_mode = suite.get("mode")
    for field, value in (
        ("config", suite_config),
        ("profile", suite_profile),
        ("mode", suite_mode),
    ):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"suite {field} must be a non-empty string")
    total_expected = 0

    for index, case in enumerate(suite["cases"], start=1):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be an object")
        try:
            gtest_filter = case["gtest_filter"]
            expected_tests = int(case["expected_tests"])
        except KeyError as error:
            raise ValueError(f"case {index} is missing {error.args[0]!r}") from error
        if expected_tests <= 0:
            raise ValueError(f"case {index} expected_tests must be positive")
        if "manifest" in case:
            raise ValueError(
                f"case {index} must not reference a generated manifest"
            )
        profile = case.get("profile", suite_profile)
        mode = case.get("mode", suite_mode)
        config = case.get("config", suite_config)
        binary = case.get("binary")
        for field, value in (
            ("config", config),
            ("profile", profile),
            ("mode", mode),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(
                    f"case {index} {field} must be a non-empty string"
                )
        if binary is not None and (not isinstance(binary, str) or not binary):
            raise ValueError(
                f"case {index} binary must be a non-empty string"
            )

        print(
            f"[{index}/{len(suite['cases'])}] {gtest_filter}"
            f"{f' [{mode}]' if mode is not None else ''} "
            f"{f'[{binary}] ' if binary is not None else ''}"
            "-> auto SASS cache",
            flush=True,
        )
        command = [
            sys.executable,
            str(repository / "tests" / "run_tests.py"),
            "run",
            "--arch",
            architecture,
            "--group",
            group,
        ]
        if profile is not None:
            command.extend(("--profile", profile))
        if mode is not None:
            command.extend(("--mode", mode))
        if config is not None:
            command.extend(("--config", config))
        if binary is not None:
            command.extend(("--binary", binary))
        command.extend(("--gtest-filter", gtest_filter, "--sass"))
        completed = subprocess.run(
            command,
            cwd=repository,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        output = ANSI_ESCAPE.sub("", completed.stdout)
        counts = PASSED_COUNT.findall(output)
        passed = int(counts[-1]) if counts else -1
        executed_sass = (
            'FlashGPU-Sim SASS: Performing strict functional execution of ' in output
            or 'mode=execution-driven SASS timing simulation' in output
        )
        if completed.returncode != 0 or passed != expected_tests or not executed_sass:
            print(output, end="" if output.endswith("\n") else "\n")
            print(
                f"case {index} failed: exit={completed.returncode}, "
                f"passed={passed}, expected={expected_tests}, sass_execution={executed_sass}",
                file=sys.stderr,
            )
            return 1
        print(f"  passed {passed}/{expected_tests}", flush=True)
        total_expected += expected_tests

    declared_total = int(suite.get("expected_tests", total_expected))
    if total_expected != declared_total:
        raise ValueError(
            f"case total {total_expected} does not match suite total {declared_total}"
        )
    print(f"strict SASS suite passed: {total_expected}/{declared_total} tests")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
