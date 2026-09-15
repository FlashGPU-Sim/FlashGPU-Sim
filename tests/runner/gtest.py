"""GoogleTest discovery, filtering, and execution."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
import shutil
import subprocess
from typing import Mapping, Sequence

from .errors import DiscoveryError, RunnerError
from .ui import UI


def parse_gtest_list(output: str) -> list[str]:
    """Convert --gtest_list_tests output into fully qualified case names."""
    cases: list[str] = []
    suite = ""
    for raw_line in output.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if not raw_line.startswith((" ", "\t")):
            if line.endswith("."):
                suite = line[:-1]
            continue
        if suite:
            cases.append(f"{suite}.{line}")
    return cases


def name_matches_filter(test_name: str, gtest_filter: str) -> bool:
    """Apply GoogleTest's positive[:...] - negative[:...] filter grammar."""
    positive, separator, negative = gtest_filter.partition("-")
    positive_patterns = [item for item in positive.split(":") if item] or ["*"]
    negative_patterns = [item for item in negative.split(":") if item]

    if not any(fnmatch.fnmatchcase(test_name, pattern) for pattern in positive_patterns):
        return False
    if separator and any(
        fnmatch.fnmatchcase(test_name, pattern) for pattern in negative_patterns
    ):
        return False
    return True


class GTest:
    def __init__(self, ui: UI, timeout: int) -> None:
        self.ui = ui
        self.timeout = timeout

    def list_cases(self, binary: Path, config_dir: Path) -> list[str]:
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise DiscoveryError(f"GTest executable not found: {binary}")

        environment = os.environ.copy()
        environment.pop("GTEST_OUTPUT", None)
        environment.pop("LD_PRELOAD", None)
        # Listing tests must not initialize the simulator. CUDA test binaries
        # register their fatbins before main(), which can make
        # --gtest_list_tests enter PTX extraction when setup_environment has
        # put the simulator's libcudart first. Use the toolkit runtime for this
        # metadata-only subprocess; actual test execution retains the caller's
        # simulator environment.
        simulator_root = environment.get("GPGPUSIM_ROOT")
        library_paths = [
            entry
            for entry in environment.get("LD_LIBRARY_PATH", "").split(":")
            if entry
        ]
        if simulator_root:
            simulator_lib = Path(simulator_root) / "lib"
            library_paths = [
                entry
                for entry in library_paths
                if not Path(entry).is_relative_to(simulator_lib)
            ]
        cuda_root = environment.get("CUDA_INSTALL_PATH")
        if cuda_root:
            cuda_lib = str(Path(cuda_root) / "lib64")
            library_paths = [
                entry for entry in library_paths if entry != cuda_lib
            ]
            library_paths.insert(0, cuda_lib)
            # Some tests link the driver API directly. GPU-less CI has the
            # toolkit stub but no libcuda.so.1 loader entry. Preloading the
            # stub satisfies its SONAME for discovery only; never export this
            # environment to actual test execution.
            driver_stub = Path(cuda_lib) / "stubs/libcuda.so"
            if driver_stub.is_file():
                environment["LD_PRELOAD"] = str(driver_stub)
        if library_paths:
            environment["LD_LIBRARY_PATH"] = ":".join(library_paths)
        else:
            environment.pop("LD_LIBRARY_PATH", None)
        completed = subprocess.run(
            [str(binary), "--gtest_color=no", "--gtest_list_tests"],
            cwd=config_dir,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise DiscoveryError(
                f"Failed to list tests in: {binary} (exit {completed.returncode})\n"
                f"{completed.stderr.strip()}"
            )
        return parse_gtest_list(completed.stdout)

    def matches(self, binary: Path, config_dir: Path, gtest_filter: str) -> bool:
        return any(
            name_matches_filter(test_name, gtest_filter)
            for test_name in self.list_cases(binary, config_dir)
        )

    def intersection(
        self,
        binary: Path,
        config_dir: Path,
        group_filter: str,
        user_filter: str,
    ) -> str | None:
        matches = [
            test_name
            for test_name in self.list_cases(binary, config_dir)
            if name_matches_filter(test_name, group_filter)
            and name_matches_filter(test_name, user_filter)
        ]
        return ":".join(matches) if matches else None

    def group_cases(
        self,
        binaries: Sequence[Path],
        config_dir: Path,
        default_filter: str = "*",
        requested_filter: str = "*",
    ) -> list[str]:
        if not binaries:
            raise RunnerError("Binary group contains no GTest binaries")

        cases: list[str] = []
        for binary in binaries:
            for test_name in self.list_cases(binary, config_dir):
                if name_matches_filter(
                    test_name, default_filter
                ) and name_matches_filter(test_name, requested_filter):
                    cases.append(test_name)
        if not cases:
            if requested_filter != "*":
                raise RunnerError(
                    f"No GTest cases matched filter: {requested_filter}"
                )
            raise RunnerError("No GTest cases matched the selected profile")
        return cases

    def run_binary(
        self,
        binary: Path,
        config_dir: Path,
        gtest_filter: str,
        *,
        env: Mapping[str, str] | None = None,
    ) -> int:
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RunnerError(f"Test executable not found or not executable: {binary}")
        if not self.matches(binary, config_dir, gtest_filter):
            raise RunnerError(
                f"No tests in {binary.name} matched filter: {gtest_filter}"
            )

        command = [str(binary), f"--gtest_filter={gtest_filter}"]
        timeout = shutil.which("timeout")
        if timeout is not None:
            command = [timeout, str(self.timeout), *command]
        return self.ui.run(command, cwd=config_dir, env=env)
