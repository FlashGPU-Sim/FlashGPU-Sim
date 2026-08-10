#!/usr/bin/env python3
"""Read, validate, and expand the TOML-owned CI job plan."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Sequence

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in the Ubuntu 22.04 CI image.
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as error:
        raise SystemExit(
            "TOML support requires Python 3.11+ or the 'tomli' package"
        ) from error


CI_DIR = Path(__file__).resolve().parent
TEST_DIR = CI_DIR.parent
REPO_ROOT = TEST_DIR.parent
JOBS_FILE = CI_DIR / "jobs.toml"

sys.path.insert(0, str(TEST_DIR))

from runner.errors import RunnerError  # noqa: E402
from runner.make import MakeInterface  # noqa: E402
from runner.selection import SelectionResolver  # noqa: E402


NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
ARCH_RE = re.compile(r"^sm[0-9]+$")
GROUP_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
SELECTOR_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")
TRANSPORT_DELIMITER = "|"


class CiPlanError(ValueError):
    """Raised when the CI job plan violates its schema."""


@dataclass(frozen=True)
class CiTest:
    name: str
    group: str
    profile: str = ""
    mode: str = ""
    gtest_filter: str = ""
    post_check: str = ""


@dataclass(frozen=True)
class CiJob:
    name: str
    arch: str
    pre_checks: tuple[str, ...]
    tests: tuple[CiTest, ...]


def _check_exact_keys(
    table: dict[str, Any], allowed: set[str], required: set[str], label: str
) -> None:
    actual = set(table)
    missing = required - actual
    extra = actual - allowed
    if missing or extra:
        missing_text = ", ".join(sorted(missing)) or "none"
        extra_text = ", ".join(sorted(extra)) or "none"
        raise CiPlanError(
            f"{JOBS_FILE}: {label} keys do not match the schema "
            f"(missing: {missing_text}; extra: {extra_text})"
        )


def _string(table: dict[str, Any], key: str, label: str) -> str:
    value = table.get(key, "")
    if not isinstance(value, str):
        raise CiPlanError(f"{JOBS_FILE}: {label}.{key} must be a string")
    if "\n" in value or "\r" in value or TRANSPORT_DELIMITER in value:
        raise CiPlanError(
            f"{JOBS_FILE}: {label}.{key} contains a reserved character"
        )
    return value


def _script_path(value: str, label: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise CiPlanError(f"{JOBS_FILE}: {label} must be a repository-relative path")
    if not path.parts or path.parts[0] != "tests" or path.suffix != ".py":
        raise CiPlanError(f"{JOBS_FILE}: {label} must name a Python script under tests/")
    resolved = REPO_ROOT.joinpath(*path.parts)
    if not resolved.is_file():
        raise CiPlanError(f"{JOBS_FILE}: {label} does not exist: {value}")
    return value


def _string_list(table: dict[str, Any], key: str, label: str) -> tuple[str, ...]:
    value = table.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CiPlanError(f"{JOBS_FILE}: {label}.{key} must be an array of strings")
    result = tuple(_script_path(item, f"{label}.{key}") for item in value)
    if len(result) != len(set(result)):
        raise CiPlanError(f"{JOBS_FILE}: {label}.{key} contains duplicates")
    return result


def _parse_test(data: Any, job_name: str, index: int) -> CiTest:
    label = f"job {job_name} test {index}"
    if not isinstance(data, dict):
        raise CiPlanError(f"{JOBS_FILE}: {label} must be an inline table")
    _check_exact_keys(
        data,
        {"name", "group", "profile", "mode", "gtest_filter", "post_check"},
        {"name", "group"},
        label,
    )
    name = _string(data, "name", label)
    group = _string(data, "group", label)
    profile = _string(data, "profile", label)
    mode = _string(data, "mode", label)
    gtest_filter = _string(data, "gtest_filter", label)
    post_check = _string(data, "post_check", label)

    if not NAME_RE.fullmatch(name):
        raise CiPlanError(f"{JOBS_FILE}: {label}.name must use lowercase kebab-case")
    if not GROUP_RE.fullmatch(group):
        raise CiPlanError(f"{JOBS_FILE}: {label}.group is invalid: {group}")
    for key, value in (("profile", profile), ("mode", mode)):
        if value and not SELECTOR_RE.fullmatch(value):
            raise CiPlanError(f"{JOBS_FILE}: {label}.{key} is invalid: {value}")
    if mode and not profile:
        raise CiPlanError(f"{JOBS_FILE}: {label}.mode requires profile")
    if post_check:
        post_check = _script_path(post_check, f"{label}.post_check")

    return CiTest(name, group, profile, mode, gtest_filter, post_check)


def load_jobs(path: Path = JOBS_FILE) -> tuple[CiJob, ...]:
    global JOBS_FILE
    original_jobs_file = JOBS_FILE
    JOBS_FILE = path
    try:
        try:
            with path.open("rb") as jobs_file:
                data = tomllib.load(jobs_file)
        except tomllib.TOMLDecodeError as error:
            raise CiPlanError(f"{path}: invalid TOML: {error}") from error

        if not isinstance(data, dict):
            raise CiPlanError(f"{path}: root must be a TOML table")
        _check_exact_keys(data, {"jobs"}, {"jobs"}, "top-level")
        raw_jobs = data["jobs"]
        if not isinstance(raw_jobs, dict) or not raw_jobs:
            raise CiPlanError(f"{path}: [jobs.<name>] must define at least one job")

        jobs: list[CiJob] = []
        for name, raw_job in raw_jobs.items():
            label = f"job {name}"
            if not isinstance(raw_job, dict):
                raise CiPlanError(f"{path}: {label} must be a table")
            _check_exact_keys(
                raw_job,
                {"arch", "pre_checks", "tests"},
                {"arch", "tests"},
                label,
            )
            arch = _string(raw_job, "arch", label)
            if not NAME_RE.fullmatch(name):
                raise CiPlanError(f"{path}: job key must use lowercase kebab-case: {name}")
            if not ARCH_RE.fullmatch(arch):
                raise CiPlanError(f"{path}: {label}.arch must match sm<digits>")

            raw_tests = raw_job["tests"]
            if not isinstance(raw_tests, list) or not raw_tests:
                raise CiPlanError(f"{path}: {label}.tests must be a non-empty array")
            tests = tuple(
                _parse_test(raw_test, name, test_index)
                for test_index, raw_test in enumerate(raw_tests, start=1)
            )
            test_names = [test.name for test in tests]
            if len(test_names) != len(set(test_names)):
                raise CiPlanError(f"{path}: job {name} contains duplicate test names")
            jobs.append(
                CiJob(
                    name=name,
                    arch=arch,
                    pre_checks=_string_list(raw_job, "pre_checks", label),
                    tests=tests,
                )
            )
        return tuple(jobs)
    finally:
        JOBS_FILE = original_jobs_file


def select_jobs(jobs: tuple[CiJob, ...], name: str) -> tuple[CiJob, ...]:
    if name == "all":
        return jobs
    selected = tuple(job for job in jobs if job.name == name)
    if selected:
        return selected
    available = " ".join(job.name for job in jobs)
    raise CiPlanError(f"Unknown CI job '{name}'; available jobs: {available}")


def validate_selections(jobs: tuple[CiJob, ...]) -> None:
    resolver = SelectionResolver(MakeInterface(TEST_DIR))
    for job in jobs:
        architecture = resolver.architecture(job.arch)
        for test in job.tests:
            selection = resolver.selection(
                architecture,
                test.group,
                profile=test.profile,
                mode=test.mode,
            )
            if selection.executor == "build-only":
                raise CiPlanError(
                    f"{job.name}/{test.name} resolves to a build-only selection"
                )
            if test.gtest_filter and selection.executor == "trace":
                raise CiPlanError(
                    f"{job.name}/{test.name} applies a GoogleTest filter to trace"
                )


def _selected(
    jobs: tuple[CiJob, ...], args: argparse.Namespace
) -> tuple[CiJob, ...]:
    return select_jobs(jobs, args.job)


def _print_plan(jobs: tuple[CiJob, ...]) -> None:
    validate_selections(jobs)
    for job in jobs:
        for test in job.tests:
            print(
                TRANSPORT_DELIMITER.join(
                    (
                        job.name,
                        job.arch,
                        test.name,
                        test.group,
                        test.profile,
                        test.mode,
                        test.gtest_filter,
                        test.post_check,
                    )
                )
            )


def _matrix_label(job: CiJob) -> str:
    prefix = f"{job.arch}-"
    suite = job.name[len(prefix) :] if job.name.startswith(prefix) else job.name
    return f"{job.arch}({suite})"


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("matrix", help="emit the GitHub Actions matrix JSON")
    subparsers.add_parser("list-jobs", help="list configured CI job names")
    for command, help_text in (
        ("plan", "emit expanded test selections"),
        ("checks", "emit pre-check scripts"),
        ("architectures", "emit selected architectures"),
        ("validate", "validate selected jobs and runner selections"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--job", default="all", help="job name or 'all'")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        jobs = load_jobs()
        if args.command == "matrix":
            validate_selections(jobs)
            print(
                json.dumps(
                    {
                        "include": [
                            {"job": job.name, "label": _matrix_label(job)}
                            for job in jobs
                        ]
                    },
                    separators=(",", ":"),
                )
            )
        elif args.command == "list-jobs":
            for job in jobs:
                print(job.name)
        elif args.command == "plan":
            _print_plan(_selected(jobs, args))
        elif args.command == "checks":
            seen: set[str] = set()
            for job in _selected(jobs, args):
                for script in job.pre_checks:
                    if script not in seen:
                        print(script)
                        seen.add(script)
        elif args.command == "architectures":
            seen_arches: set[str] = set()
            for job in _selected(jobs, args):
                if job.arch not in seen_arches:
                    print(job.arch)
                    seen_arches.add(job.arch)
        elif args.command == "validate":
            selected = _selected(jobs, args)
            validate_selections(selected)
            for job in selected:
                print(f"{job.name}: {job.arch}, {len(job.tests)} tests")
        return 0
    except (CiPlanError, OSError, RunnerError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
