"""Command-line interface for the FlashGPU-Sim test runner."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import sys
from typing import Sequence

from .build import BuildManager
from .errors import RunnerError
from .executors import TestExecutors
from .gtest import GTest
from .make import MakeInterface
from .model import Settings
from .selection import SelectionResolver
from .ui import UI


TEST_DIR = Path(__file__).resolve().parents[1]
POSITIVE_INTEGER_RE = re.compile(r"^[1-9][0-9]*$")


@dataclass
class Arguments:
    action: str = ""
    positional: list[str] = field(default_factory=list)
    verbose: bool = False
    debug: bool = False
    timeout: int | None = None
    config: str = ""
    architecture: str = ""
    test_group: str = ""
    profile: str = ""
    mode: str = ""
    gtest_filter: str = ""


def usage(program: str, stream: object = sys.stdout) -> None:
    print(
        f"""FlashGPU-Sim Test Runner
Usage: {program} [OPTIONS] ACTION [FILTER]

Actions:
  build --arch ARCH --group NAME [--profile NAME] [--mode NAME|all]
  build --arch ARCH --group all
  run   --arch ARCH --group NAME [--profile NAME] [--mode NAME] [filter]
  list-cases --arch ARCH --group NAME [--profile NAME] [--mode NAME]
             [--gtest-filter EXPR]
  list [--arch ARCH [--group NAME [--profile NAME]]]

Use '{program} list' for the manifest-derived architecture/test-group hierarchy.

Other commands:
  clean              Clean build artifacts
  setup              Setup test environment
  refresh            Refresh run directory and configuration
  list               List the manifest-derived test hierarchy
  list-cases         Build and list registered GTest cases
  list-configs       List available GPU configurations
  help               Show this help

Options:
  -v, --verbose      Verbose output
  -d, --debug        Enable debug mode
  -t, --timeout SEC  Set test timeout
  -c, --config NAME  Use specific GPU configuration
  --arch NAME        Select the hardware architecture
  --group NAME       Select one source/binary group; build also accepts all
  --profile NAME     Select a profile within a complex test group
  --mode NAME        Select a compile-time analysis mode
  --gtest-filter EXPR Filter GoogleTest cases for run or list-cases
  -h, --help         Show this help
  Options may appear before or after the command.

Examples:
  {program} build --arch sm120 --group all
  {program} list-cases --arch sm120 --group integration
  {program} list-cases --arch sm120 --group integration --gtest-filter '*VectorAdd*'
  {program} run --arch sm120 --group integration CudaVectorAdd
  {program} run --arch sm90 --group wgmma --gtest-filter 'WgmmaF16*'
  {program} run --arch sm90 --group fa2 --profile smoke
  {program} run --arch sm90 --group fa2 --profile breakdown --mode only_mma
  {program} run --arch sm90 --group fa3 --profile scaling --mode baseline
  {program} build --arch sm120 --group microbench --profile memory
  {program} run --arch sm120 --group trace --profile gpt2 flash_attn

Standalone calibration groups are build-only; use their local Makefiles
for benchmark-specific runtime arguments. Mode 'all' is also build-only.""",
        file=stream,
    )


def _require_value(argv: Sequence[str], index: int, option: str) -> str:
    if index + 1 >= len(argv) or not argv[index + 1]:
        raise RunnerError(f"{option} requires a value")
    return argv[index + 1]


def _set_once(arguments: Arguments, field_name: str, value: str, option: str) -> None:
    if getattr(arguments, field_name):
        raise RunnerError(f"{option} may only be specified once")
    setattr(arguments, field_name, value)


def parse_arguments(argv: Sequence[str]) -> Arguments:
    arguments = Arguments()
    index = 0
    positional_only = False
    while index < len(argv):
        token = argv[index]
        if not positional_only and token == "--":
            positional_only = True
            index += 1
            continue
        if not positional_only and token in {"-h", "--help"}:
            arguments.action = "help"
            arguments.positional.clear()
            return arguments
        if not positional_only and token in {"-v", "--verbose"}:
            arguments.verbose = True
            index += 1
            continue
        if not positional_only and token in {"-d", "--debug"}:
            arguments.debug = True
            index += 1
            continue
        if not positional_only and token in {"-t", "--timeout"}:
            value = _require_value(argv, index, "--timeout")
            if not POSITIVE_INTEGER_RE.fullmatch(value):
                raise RunnerError("--timeout requires a positive integer")
            arguments.timeout = int(value)
            index += 2
            continue
        if not positional_only and token in {"-c", "--config"}:
            arguments.config = _require_value(argv, index, "--config")
            index += 2
            continue
        if not positional_only and token == "--arch":
            _set_once(
                arguments,
                "architecture",
                _require_value(argv, index, "--arch"),
                "--arch",
            )
            index += 2
            continue
        if not positional_only and token == "--group":
            _set_once(
                arguments,
                "test_group",
                _require_value(argv, index, "--group"),
                "--group",
            )
            index += 2
            continue
        if not positional_only and token == "--profile":
            _set_once(
                arguments,
                "profile",
                _require_value(argv, index, "--profile"),
                "--profile",
            )
            index += 2
            continue
        if not positional_only and token == "--mode":
            _set_once(
                arguments,
                "mode",
                _require_value(argv, index, "--mode"),
                "--mode",
            )
            index += 2
            continue
        if not positional_only and token == "--gtest-filter":
            _set_once(
                arguments,
                "gtest_filter",
                _require_value(argv, index, "--gtest-filter"),
                "--gtest-filter",
            )
            index += 2
            continue
        if not positional_only and token.startswith("-"):
            raise RunnerError(f"Unknown option: {token}")

        if not arguments.action:
            arguments.action = token
        else:
            arguments.positional.append(token)
        index += 1

    return arguments


def validate_arguments(arguments: Arguments) -> None:
    action = arguments.action
    if not action:
        return
    if action not in {
        "build",
        "run",
        "setup",
        "refresh",
        "clean",
        "list",
        "list-cases",
        "list-configs",
        "help",
    }:
        raise RunnerError(f"Unknown command: {action}")

    if action == "build" and arguments.positional:
        raise RunnerError("build does not accept positional arguments")
    if action == "run" and len(arguments.positional) > 1:
        raise RunnerError("run accepts at most one filter")
    if action in {
        "setup",
        "refresh",
        "clean",
        "list",
        "list-cases",
        "list-configs",
        "help",
    } and arguments.positional:
        raise RunnerError(f"{action} does not accept arguments")

    selectors = arguments.test_group or arguments.profile or arguments.mode
    selection_actions = {"build", "run", "list", "list-cases"}
    if selectors and action not in selection_actions:
        raise RunnerError(
            "--group, --profile, and --mode are only valid with "
            "build, run, list, or list-cases"
        )
    if arguments.architecture and action not in selection_actions:
        raise RunnerError(
            "--arch is only valid with build, run, list, or list-cases"
        )
    if arguments.gtest_filter and action not in {"run", "list-cases"}:
        raise RunnerError(
            "--gtest-filter is only valid with run or list-cases"
        )
    if arguments.gtest_filter and arguments.positional:
        raise RunnerError(
            "--gtest-filter cannot be combined with a positional filter"
        )
    if arguments.test_group and not arguments.architecture:
        raise RunnerError("--group requires --arch")
    if arguments.profile and not arguments.test_group:
        raise RunnerError("--profile requires --group")
    if arguments.test_group == "all":
        if action != "build":
            raise RunnerError("--group all is only valid with build")
        if arguments.profile or arguments.mode:
            raise RunnerError(
                "--group all does not accept --profile or --mode"
            )
    if arguments.mode and action == "list":
        raise RunnerError("list does not accept --mode")


def _read_test_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _positive_setting(value: str | None, fallback: int) -> int:
    if value and POSITIVE_INTEGER_RE.fullmatch(value):
        return int(value)
    return fallback


def _nonnegative_setting(value: str | None, fallback: int) -> int:
    if value and value.isdigit():
        return int(value)
    return fallback


def load_settings(arguments: Arguments) -> Settings:
    config = _read_test_config(TEST_DIR / "test.config")
    default_config = os.environ.get("DEFAULT_GPU_CONFIG", "SM120_RTX5090")
    explicit_config = os.environ.get("GPU_CONFIG", "")
    settings = Settings(
        timeout=_positive_setting(config.get("TEST_TIMEOUT"), 3600),
        verbose=_nonnegative_setting(config.get("TEST_VERBOSE"), 1),
        debug=config.get("DEBUG_TESTS", "0") == "1",
        test_build_jobs=_positive_setting(
            os.environ.get("TEST_BUILD_JOBS")
            or os.environ.get("FA2_BUILD_JOBS"),
            4,
        ),
        simulator_build_jobs=_positive_setting(
            os.environ.get("GPGPUSIM_BUILD_JOBS"), 4
        ),
        gpu_config=explicit_config or default_config,
        gpu_config_explicit=bool(explicit_config),
    )
    if arguments.verbose:
        settings.verbose = 2
    if arguments.debug:
        settings.debug = True
    if arguments.timeout is not None:
        settings.timeout = arguments.timeout
    if arguments.config:
        settings.gpu_config = arguments.config
        settings.gpu_config_explicit = True
    return settings


def list_hierarchy(
    arguments: Arguments,
    make: MakeInterface,
    ui: UI,
) -> None:
    architectures = make.architectures()
    if arguments.architecture and arguments.architecture not in architectures:
        raise RunnerError(
            f"Unknown architecture: {arguments.architecture}\n"
            f"Available architectures: {' '.join(architectures)}"
        )
    if arguments.test_group:
        groups = make.test_groups(arguments.architecture)
        if arguments.test_group not in groups:
            raise RunnerError(
                f"Unsupported test group '{arguments.test_group}' for "
                f"'{arguments.architecture}'\nAvailable test groups: "
                f"{' '.join(groups)}"
            )
    if arguments.profile:
        profiles = make.profiles(
            arguments.architecture, arguments.test_group
        )
        if arguments.profile not in profiles:
            raise RunnerError(
                f"Unknown profile '{arguments.profile}' for "
                f"{arguments.architecture}/{arguments.test_group}\n"
                f"Available profiles: {' '.join(profiles)}"
            )

    ui.info("Supported architecture / test-group hierarchy:")
    for architecture in architectures:
        if arguments.architecture and architecture != arguments.architecture:
            continue
        ui.plain(architecture)
        for test_group in make.test_groups(architecture):
            if arguments.test_group and test_group != arguments.test_group:
                continue
            ui.plain(f"  {test_group}")
            for profile in make.profiles(architecture, test_group):
                if arguments.profile and profile != arguments.profile:
                    continue
                ui.plain(f"    {profile}")
                if arguments.profile:
                    for mode in make.modes(
                        architecture, test_group, profile
                    ):
                        ui.plain(f"      {mode}")


def execute(arguments: Arguments, program: str) -> int:
    diagnostics_only = arguments.action == "list-cases"
    ui = UI(sys.stderr if diagnostics_only else sys.stdout)
    settings = load_settings(arguments)
    make = MakeInterface(TEST_DIR)
    resolver = SelectionResolver(make)
    builder = BuildManager(TEST_DIR, settings, make, ui)
    gtest = GTest(ui, settings.timeout)
    executors = TestExecutors(TEST_DIR, settings, make, builder, gtest, ui)

    if not arguments.action:
        usage(program)
        return 0
    if arguments.action == "help":
        usage(program)
        return 0
    if arguments.action == "setup":
        builder.setup_environment()
        return 0
    if arguments.action == "refresh":
        ui.info("Refreshing run directory and configuration...")
        builder.setup_run_directory()
        ui.success("Run directory refreshed!")
        return 0
    if arguments.action == "clean":
        builder.clean()
        return 0
    if arguments.action == "list-configs":
        builder.list_configs(program)
        return 0
    if arguments.action == "list":
        list_hierarchy(arguments, make, ui)
        return 0

    architecture = resolver.architecture(arguments.architecture)
    if arguments.action == "build" and arguments.test_group == "all":
        builder.configure_architecture(architecture)
        builder.setup_run_directory()
        builder.build_all(architecture, resolver)
        return 0

    selection = resolver.selection(
        architecture,
        arguments.test_group,
        profile=arguments.profile,
        mode=arguments.mode,
    )
    builder.configure_architecture(architecture)

    if arguments.action == "build":
        builder.setup_run_directory()
        builder.build(selection)
        return 0
    if arguments.action == "list-cases":
        for test_name in executors.list_cases(
            selection, arguments.gtest_filter
        ):
            print(test_name, flush=True)
        return 0
    if arguments.action == "run":
        if arguments.gtest_filter and selection.executor == "trace":
            raise RunnerError("--gtest-filter is not valid for trace tests")
        if selection.mode == "all":
            raise RunnerError(
                "--mode all is build-only; run requires one concrete mode"
            )
        builder.setup_run_directory()
        return executors.run(
            selection,
            arguments.positional[0] if arguments.positional else "",
            arguments.gtest_filter,
        )
    raise RunnerError(f"Unknown command: {arguments.action}")


def main(argv: Sequence[str] | None = None) -> int:
    arguments_list = list(sys.argv[1:] if argv is None else argv)
    program = sys.argv[0]
    try:
        arguments = parse_arguments(arguments_list)
        validate_arguments(arguments)
        return execute(arguments, program)
    except RunnerError as error:
        UI().error(str(error))
        if arguments_list and arguments_list[0] not in {
            "build",
            "run",
            "setup",
            "refresh",
            "clean",
            "list",
            "list-cases",
            "list-configs",
            "help",
        } and not arguments_list[0].startswith("-"):
            usage(program, stream=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
