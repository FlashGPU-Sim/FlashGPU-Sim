#!/usr/bin/env python3
"""Plan, run, and report informational CI cycle validation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import time
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


PERF_DIR = Path(__file__).resolve().parent
REPO_ROOT = PERF_DIR.parents[2]
MANIFEST_FILE = PERF_DIR / "cases.toml"
LOG_ROOT = REPO_ROOT / "tests" / "logs" / "ci" / "perf"
README_FILE = REPO_ROOT / "README.md"

CONFIG_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
CASE_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
LAUNCHER_RE = re.compile(r"^[_A-Za-z0-9]+_launch[1-9][0-9]*$")
SIM_CYCLES_RE = re.compile(r"^gpu_tot_sim_cycle\s*=\s*([0-9]+)\s*$", re.M)
README_HEADER = (
    "Config",
    "Workload",
    "Shape",
    "NCU cycles",
    "Sim cycles",
    "Difference",
)


class PerfError(ValueError):
    """Raised when the performance manifest or generated results are invalid."""


@dataclass(frozen=True)
class PerfCase:
    id: str
    label: str
    scripts: tuple[str, ...]
    launcher: str | None
    ncu_cycles: float
    readme: bool


@dataclass(frozen=True)
class PerfJob:
    id: str
    config: str
    label: str
    cases: tuple[PerfCase, ...]


def _exact_keys(
    table: dict[str, Any],
    allowed: set[str],
    required: set[str],
    label: str,
    manifest_path: Path,
) -> None:
    actual = set(table)
    missing = required - actual
    extra = actual - allowed
    if missing or extra:
        missing_text = ", ".join(sorted(missing)) or "none"
        extra_text = ", ".join(sorted(extra)) or "none"
        raise PerfError(
            f"{manifest_path}: {label} keys do not match the schema "
            f"(missing: {missing_text}; extra: {extra_text})"
        )


def _nonempty_string(value: Any, label: str, manifest_path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PerfError(f"{manifest_path}: {label} must be a non-empty string")
    if "\n" in value or "\r" in value:
        raise PerfError(f"{manifest_path}: {label} must be a single line")
    return value.strip()


def _script_path(
    value: Any, label: str, manifest_path: Path, repo_root: Path
) -> str:
    if not isinstance(value, str):
        raise PerfError(f"{manifest_path}: {label} must be a string")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise PerfError(
            f"{manifest_path}: {label} must be a repository-relative path"
        )
    if path.suffix != ".sh":
        raise PerfError(f"{manifest_path}: {label} must name a shell script")

    root = repo_root.resolve()
    resolved = root.joinpath(*path.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PerfError(f"{manifest_path}: {label} escapes the repository") from error
    if not resolved.is_file():
        raise PerfError(f"{manifest_path}: {label} does not exist: {value}")
    return path.as_posix()


def _launcher_path(
    value: Any,
    label: str,
    manifest_path: Path,
    repo_root: Path,
    config: str,
) -> str:
    if not isinstance(value, str):
        raise PerfError(f"{manifest_path}: {label} must be a string")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise PerfError(
            f"{manifest_path}: {label} must be a repository-relative path"
        )
    if not LAUNCHER_RE.fullmatch(path.name):
        raise PerfError(
            f"{manifest_path}: {label} must end with a generated launcher name"
        )

    root = repo_root.resolve()
    resolved = root.joinpath(*path.parts).resolve()
    fixture_root = (repo_root / "tests" / "ci" / "perf" / "traces" / config).resolve()
    try:
        resolved.relative_to(fixture_root)
    except ValueError as error:
        raise PerfError(
            f"{manifest_path}: {label} must be under "
            f"tests/ci/perf/traces/{config}"
        ) from error

    required_suffixes = (
        "_Makefile",
        "_harness.cu",
        "_kernel.cubin",
        "_kernel.ptx",
        "_kernel.ptxinfo",
    )
    missing = [
        suffix
        for suffix in required_suffixes
        if not resolved.with_name(resolved.name + suffix).is_file()
    ]
    if missing:
        missing_text = ", ".join(resolved.name + suffix for suffix in missing)
        raise PerfError(
            f"{manifest_path}: {label} replay is incomplete; missing: {missing_text}"
        )
    return path.as_posix()


def load_manifest(
    path: Path = MANIFEST_FILE, repo_root: Path = REPO_ROOT
) -> tuple[PerfJob, ...]:
    try:
        with path.open("rb") as manifest_file:
            data = tomllib.load(manifest_file)
    except tomllib.TOMLDecodeError as error:
        raise PerfError(f"{path}: invalid TOML: {error}") from error

    if not isinstance(data, dict) or not data:
        raise PerfError(f"{path}: define at least one [<config>] table")

    jobs: list[PerfJob] = []
    for config, raw_config in data.items():
        if not CONFIG_RE.fullmatch(config):
            raise PerfError(
                f"{path}: config key must use uppercase letters, digits, and "
                f"underscores: {config}"
            )
        if not isinstance(raw_config, dict) or not raw_config:
            raise PerfError(f"{path}: [{config}] must be a table")

        config_file = repo_root / "configs" / config / "gpgpusim.config"
        if not config_file.is_file():
            raise PerfError(
                f"{path}: [{config}] has no simulator configuration at "
                f"{config_file}"
            )
        case_ids: set[str] = set()
        labels: set[str] = set()
        for job_id, raw_cases in raw_config.items():
            job_path = f"[{config}.{job_id}]"
            if not CASE_RE.fullmatch(job_id):
                raise PerfError(
                    f"{path}: job key must use lowercase kebab-case: {job_id}"
                )
            if not isinstance(raw_cases, dict) or not raw_cases:
                raise PerfError(f"{path}: {job_path} must define at least one case")

            cases: list[PerfCase] = []
            for case_id, raw_case in raw_cases.items():
                case_path = f"[{config}.{job_id}.{case_id}]"
                if not CASE_RE.fullmatch(case_id):
                    raise PerfError(
                        f"{path}: case key must use lowercase kebab-case: {case_id}"
                    )
                if case_id in case_ids:
                    raise PerfError(
                        f"{path}: [{config}] contains duplicate case key: {case_id}"
                    )
                case_ids.add(case_id)
                if not isinstance(raw_case, dict):
                    raise PerfError(f"{path}: {case_path} must be a table")
                _exact_keys(
                    raw_case,
                    {"label", "scripts", "launcher", "ncu_cycles", "readme"},
                    {"label", "ncu_cycles"},
                    case_path,
                    path,
                )
                label = _nonempty_string(
                    raw_case["label"], f"{case_path}.label", path
                )
                if label in labels:
                    raise PerfError(
                        f"{path}: [{config}] contains duplicate case label: {label}"
                    )
                labels.add(label)

                execution_keys = {"scripts", "launcher"} & set(raw_case)
                if len(execution_keys) != 1:
                    raise PerfError(
                        f"{path}: {case_path} must define exactly one of "
                        "scripts or launcher"
                    )

                scripts: tuple[str, ...] = ()
                launcher: str | None = None
                if "scripts" in raw_case:
                    raw_scripts = raw_case["scripts"]
                    if not isinstance(raw_scripts, list) or not raw_scripts:
                        raise PerfError(
                            f"{path}: {case_path}.scripts must be a non-empty array"
                        )
                    scripts = tuple(
                        _script_path(
                            script,
                            f"{case_path}.scripts[{index}]",
                            path,
                            repo_root,
                        )
                        for index, script in enumerate(raw_scripts)
                    )
                    if len(scripts) != len(set(scripts)):
                        raise PerfError(
                            f"{path}: {case_path}.scripts contains duplicates"
                        )
                else:
                    launcher = _launcher_path(
                        raw_case["launcher"],
                        f"{case_path}.launcher",
                        path,
                        repo_root,
                        config,
                    )

                raw_cycles = raw_case["ncu_cycles"]
                if (
                    isinstance(raw_cycles, bool)
                    or not isinstance(raw_cycles, (int, float))
                    or not math.isfinite(float(raw_cycles))
                    or raw_cycles <= 0
                ):
                    raise PerfError(
                        f"{path}: {case_path}.ncu_cycles must be a finite "
                        "positive number"
                    )

                readme = raw_case.get("readme", False)
                if not isinstance(readme, bool):
                    raise PerfError(f"{path}: {case_path}.readme must be a boolean")
                cases.append(
                    PerfCase(
                        id=case_id,
                        label=label,
                        scripts=scripts,
                        launcher=launcher,
                        ncu_cycles=float(raw_cycles),
                        readme=readme,
                    )
                )
            architecture = config.split("_", 1)[0].lower()
            jobs.append(
                PerfJob(
                    id=job_id,
                    config=config,
                    label=f"{architecture}({job_id})",
                    cases=tuple(cases),
                )
            )

    return tuple(jobs)


def matrix_json(jobs: tuple[PerfJob, ...]) -> str:
    return json.dumps(
        {
            "include": [
                {"config": job.config, "job": job.id, "label": job.label}
                for job in jobs
            ]
        },
        separators=(",", ":"),
    )


def select_job(jobs: tuple[PerfJob, ...], config: str, job_id: str) -> PerfJob:
    for job in jobs:
        if job.config == config and job.id == job_id:
            return job
    available = " ".join(f"{job.config}/{job.id}" for job in jobs)
    raise PerfError(
        f"unknown performance job '{config}/{job_id}'; available: {available}"
    )


def _result_file(log_root: Path, job: PerfJob) -> Path:
    return log_root / "results" / f"{job.config}-{job.id}.json"


def _initial_results(job: PerfJob, log_root: Path) -> dict[str, Any]:
    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    return {
        "config": job.config,
        "job": job.id,
        "cases": [
            {
                "id": case.id,
                "status": "not-run",
                "sim_cycles": None,
                "exit_code": None,
                "seconds": 0.0,
                "log": display_path(
                    log_root / job.config / job.id / f"{case.id}.log"
                ),
            }
            for case in job.cases
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def initialize_results(job: PerfJob, log_root: Path = LOG_ROOT) -> Path:
    path = _result_file(log_root, job)
    _write_json(path, _initial_results(job, log_root))
    return path


def _stream_command(
    command: Sequence[str],
    cwd: Path,
    environment: dict[str, str],
    log_file: Any,
) -> int:
    banner = f"\n$ {shlex.join(command)}\n"
    print(banner, end="", flush=True)
    log_file.write(banner)
    log_file.flush()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as error:
        message = f"Unable to start {command[0]}: {error}\n"
        print(message, end="", flush=True)
        log_file.write(message)
        return 127

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_file.write(line)
    process.stdout.close()
    return process.wait()


def _run_replay(
    case: PerfCase,
    config: str,
    job_id: str,
    repo_root: Path,
    log_root: Path,
    environment: dict[str, str],
    log_file: Any,
) -> int:
    assert case.launcher is not None
    if environment.get("GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN") != "1":
        message = "FlashGPU-Sim environment is not active for replay\n"
        print(message, end="", flush=True)
        log_file.write(message)
        return 2

    launcher_path = repo_root.joinpath(*PurePosixPath(case.launcher).parts)
    launcher_name = launcher_path.name
    work_root = (log_root / "work").resolve()
    work_dir = work_root / config / job_id / case.id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree(launcher_path.parent, work_dir)
    shutil.copytree(
        repo_root / "configs" / config,
        work_dir,
        dirs_exist_ok=True,
    )

    exit_code = _stream_command(
        [
            "make",
            "--no-print-directory",
            "-C",
            str(work_dir),
            "-f",
            f"{launcher_name}_Makefile",
        ],
        repo_root,
        environment,
        log_file,
    )
    if exit_code != 0:
        return exit_code
    return _stream_command(
        [f"./{launcher_name}"],
        work_dir,
        environment,
        log_file,
    )


def run_job(
    job: PerfJob,
    repo_root: Path = REPO_ROOT,
    log_root: Path = LOG_ROOT,
) -> int:
    payload = _initial_results(job, log_root)
    result_path = _result_file(log_root, job)
    _write_json(result_path, payload)
    failed = False

    environment = os.environ.copy()
    environment["PERF_SIM_CONFIG"] = job.config
    environment["PERF_JOB"] = job.id

    for case, result in zip(job.cases, payload["cases"]):
        log_path = log_root / job.config / job.id / f"{case.id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Running performance case: {job.config}/{job.id}/{case.id}",
            flush=True,
        )
        started = time.monotonic()
        exit_code = 0
        with log_path.open("w", encoding="utf-8") as log_file:
            if case.launcher is not None:
                exit_code = _run_replay(
                    case,
                    job.config,
                    job.id,
                    repo_root,
                    log_root,
                    environment,
                    log_file,
                )
            else:
                for script in case.scripts:
                    exit_code = _stream_command(
                        ["bash", script],
                        repo_root,
                        environment,
                        log_file,
                    )
                    if exit_code != 0:
                        break

        elapsed = time.monotonic() - started
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        cycle_matches = SIM_CYCLES_RE.findall(log_text)
        sim_cycles = int(cycle_matches[-1]) if cycle_matches else None

        if exit_code != 0:
            status = "failed"
        elif sim_cycles is None:
            status = "missing-cycles"
        else:
            status = "compared"
        failed = failed or status != "compared"
        result.update(
            {
                "status": status,
                "sim_cycles": sim_cycles,
                "exit_code": exit_code,
                "seconds": round(elapsed, 3),
            }
        )
        _write_json(result_path, payload)
        print(
            f"Completed performance case: "
            f"{job.config}/{job.id}/{case.id} status={status}",
            flush=True,
        )

    return 1 if failed else 0


def _table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _read_results(results_dir: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str]]:
    results: dict[tuple[str, str], dict[str, Any]] = {}
    warnings: list[str] = []
    for path in sorted(results_dir.rglob("*.json")) if results_dir.is_dir() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            config = payload["config"]
            raw_cases = payload["cases"]
            if not isinstance(config, str) or not isinstance(raw_cases, list):
                raise ValueError("invalid config or cases field")
            for raw_case in raw_cases:
                case_id = raw_case["id"]
                if not isinstance(case_id, str):
                    raise ValueError("case id is not a string")
                key = (config, case_id)
                if key in results:
                    warnings.append(f"duplicate result for {config}/{case_id}")
                results[key] = raw_case
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            warnings.append(f"could not parse {path.name}: {error}")
    return results, warnings


def _split_markdown_row(line: str) -> tuple[str, ...]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return ()
    return tuple(cell.strip() for cell in stripped[1:-1].split("|"))


def _readme_errors(
    readme_path: Path,
    jobs: tuple[PerfJob, ...],
    results: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    try:
        lines = readme_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return [f"could not read {readme_path}: {error}"]

    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if _split_markdown_row(line) == README_HEADER
        ),
        None,
    )
    if header_index is None:
        return ["README Cycle Validation table was not found"]

    expected: dict[tuple[str, str], tuple[PerfCase, dict[str, Any] | None]] = {}
    for job in jobs:
        for case in job.cases:
            if not case.readme:
                continue
            expected[(job.config, case.label)] = (
                case,
                results.get((job.config, case.id)),
            )

    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    for line in lines[header_index + 2 :]:
        cells = _split_markdown_row(line)
        if not cells:
            break
        if len(cells) != len(README_HEADER):
            errors.append(f"README row has {len(cells)} columns instead of 6")
            continue
        config, label, _shape, ncu_text, sim_text, difference_text = cells
        key = (config, label)
        if key in seen:
            errors.append(f"README contains duplicate row for {config}/{label}")
            continue
        seen.add(key)
        if key not in expected:
            errors.append(f"README row has no performance case: {config}/{label}")
            continue
        case, result = expected[key]
        if result is None or result.get("status") != "compared":
            errors.append(f"no compared result is available for {config}/{label}")
            continue
        try:
            readme_ncu = float(ncu_text.replace(",", ""))
            readme_sim = int(sim_text.replace(",", ""))
            readme_difference = float(difference_text.rstrip("%"))
        except ValueError:
            errors.append(f"README row contains an invalid number: {config}/{label}")
            continue

        sim_cycles = result.get("sim_cycles")
        if not isinstance(sim_cycles, int):
            errors.append(f"result contains invalid cycles for {config}/{label}")
            continue
        difference = (sim_cycles - case.ncu_cycles) / case.ncu_cycles * 100.0
        if not math.isclose(readme_ncu, case.ncu_cycles, abs_tol=0.005):
            errors.append(
                f"README NCU cycles differ for {config}/{label} "
                f"(README: {readme_ncu:,.2f}; manifest: {case.ncu_cycles:,.2f})"
            )
        if readme_sim != sim_cycles:
            errors.append(
                f"README simulator cycles differ for {config}/{label} "
                f"(README: {readme_sim:,}; current: {sim_cycles:,})"
            )
        if not math.isclose(readme_difference, difference, abs_tol=0.005):
            errors.append(
                f"README percentage differs for {config}/{label} "
                f"(README: {readme_difference:+.2f}%; current: {difference:+.2f}%)"
            )

    for config, label in expected.keys() - seen:
        errors.append(f"README row is missing for {config}/{label}")
    return errors


def render_report(
    jobs: tuple[PerfJob, ...],
    results_dir: Path,
    readme_path: Path | None = README_FILE,
) -> tuple[str, bool]:
    results, warnings = _read_results(results_dir)
    lines = [
        "## Performance Test",
        "",
        "NCU metric: `sm__cycles_elapsed.avg`; simulator metric: "
        "`gpu_tot_sim_cycle`; difference: `(Sim - NCU) / NCU`.",
        "",
    ]
    counts = {"compared": 0, "failed": 0, "missing": 0}

    configs = tuple(dict.fromkeys(job.config for job in jobs))
    for config in configs:
        lines.extend(
            [
                f"### `{config}`",
                "",
                "| Result | Case | NCU cycles | Sim cycles | Δ cycles | Difference |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for job in (job for job in jobs if job.config == config):
            for case in job.cases:
                result = results.get((job.config, case.id))
                status = result.get("status") if result else "missing-result"
                sim_cycles = result.get("sim_cycles") if result else None
                if status == "compared" and isinstance(sim_cycles, int):
                    result_text = "Compared"
                    delta = sim_cycles - case.ncu_cycles
                    difference = delta / case.ncu_cycles * 100.0
                    sim_text = f"{sim_cycles:,}"
                    delta_text = f"{delta:+,.2f}"
                    difference_text = f"{difference:+.2f}%"
                    counts["compared"] += 1
                else:
                    result_text = {
                        "failed": "Simulation failed",
                        "missing-cycles": "Missing cycles",
                        "not-run": "Not run",
                    }.get(str(status), "Missing result")
                    sim_text = "—"
                    delta_text = "—"
                    difference_text = "—"
                    counts["failed" if status == "failed" else "missing"] += 1
                lines.append(
                    f"| {result_text} | {_table_cell(case.label)} | "
                    f"{case.ncu_cycles:,.2f} | {sim_text} | {delta_text} | "
                    f"{difference_text} |"
                )
        lines.append("")

    readme_errors = (
        _readme_errors(readme_path, jobs, results) if readme_path is not None else []
    )
    if readme_path is not None:
        lines.extend(
            [
                "### Status",
                "",
                "- README Cycle Validation: "
                + ("matches" if not readme_errors else "does **NOT** match"),
            ]
        )
    if readme_errors:
        lines.extend(
            [
                "",
                "### Suggested follow-up",
                "",
                "The current results do not match the documented Cycle "
                "Validation values.",
                "",
                "- Inspect whether the cycle change is expected or a regression.",
                "- If expected, update the top-level README table.",
                "- For broader validation, also run Triton FlashAttention offline: "
                "`bash tutorials/triton-flash-attention/capture.sh`, then "
                "`bash tutorials/triton-flash-attention/run.sh`.",
            ]
        )

    success = (
        counts["failed"] == 0
        and counts["missing"] == 0
        and not warnings
        and not readme_errors
    )
    return "\n".join(lines) + "\n", success


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_FILE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate the performance manifest")
    subparsers.add_parser("matrix", help="emit the config job matrix JSON")

    for command in ("init", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
        child.add_argument("--job", required=True)
        child.add_argument("--log-root", type=Path, default=LOG_ROOT)

    report = subparsers.add_parser("report", help="render the combined job summary")
    report.add_argument("--results-dir", type=Path, default=LOG_ROOT / "results")
    report.add_argument("--readme", type=Path, default=README_FILE)
    report.add_argument("--no-readme", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        jobs = load_manifest(args.manifest)
        if args.command == "validate":
            for job in jobs:
                print(f"{job.config}/{job.id}: {len(job.cases)} cases")
        elif args.command == "matrix":
            print(matrix_json(jobs))
        elif args.command == "init":
            path = initialize_results(
                select_job(jobs, args.config, args.job), args.log_root
            )
            print(path)
        elif args.command == "run":
            return run_job(
                select_job(jobs, args.config, args.job), log_root=args.log_root
            )
        elif args.command == "report":
            report, success = render_report(
                jobs,
                args.results_dir,
                None if args.no_readme else args.readme,
            )
            print(report, end="")
            return 0 if success else 1
        return 0
    except (OSError, PerfError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
