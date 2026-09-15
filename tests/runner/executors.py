"""Runtime executors for resolved test selections."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from .build import BuildManager
from .errors import RunnerError
from .gtest import GTest
from .make import MakeInterface
from .model import Selection, Settings
from .ui import RED, UI


class TestExecutors:
    def __init__(
        self,
        test_dir: Path,
        settings: Settings,
        make: MakeInterface,
        builder: BuildManager,
        gtest: GTest,
        ui: UI,
    ) -> None:
        self.test_dir = test_dir
        self.settings = settings
        self.make = make
        self.builder = builder
        self.gtest = gtest
        self.ui = ui

    def _binary_paths(self, binary_group: str) -> list[Path]:
        try:
            binaries = self.make.binaries(binary_group)
        except RunnerError as error:
            raise RunnerError(
                f"Unable to resolve binary group: {binary_group}\n{error}"
            ) from error
        paths: list[Path] = []
        for binary in binaries:
            path = Path(binary)
            paths.append(path if path.is_absolute() else self.test_dir / path)
        if self.settings.binary:
            selected = [
                path for path in paths if path.name == self.settings.binary
            ]
            if not selected:
                available = " ".join(path.name for path in paths) or "<none>"
                raise RunnerError(
                    f"Binary '{self.settings.binary}' is not in group "
                    f"'{binary_group}'\nAvailable binaries: {available}"
                )
            return selected
        return paths

    def _gtest_environment(
        self,
        selection: Selection,
        binary: Path,
        **updates: str,
    ) -> dict[str, str]:
        # Reconstruct frontend selection for this invocation; a previous
        # shell experiment must not silently select a different frontend.
        environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith("FLASHGPU_SASS_")
        }
        environment.pop("PTX_SIM_MODE_FUNC", None)
        environment["GTEST_COLOR"] = "yes"
        if self.settings.verbose == 2:
            environment["GTEST_VERBOSITY"] = "1"
        if self.settings.sassir:
            environment["FLASHGPU_SASS_IR"] = self.settings.sassir
        elif self.settings.sass_auto:
            cuda_root = environment.get("CUDA_INSTALL_PATH")
            if not cuda_root:
                raise RunnerError(
                    "--sass requires CUDA_INSTALL_PATH for nvdisasm and cuobjdump"
                )
            cuda_bin = Path(cuda_root) / "bin"
            nvdisasm = cuda_bin / "nvdisasm"
            cuobjdump = cuda_bin / "cuobjdump"
            decode_tool = (
                self.test_dir.parent
                / "src/gpgpu-sim/flash/sass/tools/dump_kernel_sassir.py"
            )
            for label, path in (
                ("nvdisasm", nvdisasm),
                ("cuobjdump", cuobjdump),
                ("SASS decode tool", decode_tool),
            ):
                if not path.is_file():
                    raise RunnerError(f"{label} not found: {path}")
            environment.pop("FLASHGPU_SASS_IR", None)
            environment.update(
                {
                    "FLASHGPU_SASS_AUTO": "1",
                    "FLASHGPU_SASS_BINARY": str(binary.resolve()),
                    "FLASHGPU_SASS_ARCH": selection.architecture.name,
                    "FLASHGPU_SASS_DECODE_TOOL": str(decode_tool.resolve()),
                    "FLASHGPU_SASS_PYTHON": str(Path(sys.executable).resolve()),
                    "FLASHGPU_SASS_NVDISASM": str(nvdisasm.resolve()),
                    "FLASHGPU_SASS_CUOBJDUMP": str(cuobjdump.resolve()),
                }
            )
        if self.settings.sass_timing:
            environment["FLASHGPU_SASS_TIMING"] = "1"
        environment.update(updates)
        return environment

    def list_cases(
        self, selection: Selection, requested_filter: str = ""
    ) -> list[str]:
        if selection.executor == "trace":
            raise RunnerError(
                f"{selection.architecture.name}/{selection.test_group} "
                "does not use GoogleTest"
            )
        if selection.executor == "build-only" and selection.binary_group == "none":
            raise RunnerError(
                f"{selection.label} has no GTest binary manifest"
            )
        if selection.executor not in {
            "gtest-single",
            "gtest-multi",
            "fa3-profile",
            "build-only",
        }:
            raise RunnerError(
                f"Unknown executor '{selection.executor}' for "
                f"{selection.architecture.name}/{selection.test_group}"
            )

        self.builder.setup_run_directory()
        self.builder.build(selection)
        self.ui.info(f"GTest cases for {selection.label}:")
        return self.gtest.group_cases(
            self._binary_paths(selection.binary_group),
            self.builder.config_dir,
            selection.default_filter,
            requested_filter or "*",
        )

    def run(
        self,
        selection: Selection,
        positional_filter: str = "",
        requested_gtest_filter: str = "",
    ) -> int:
        if selection.executor == "gtest-single":
            return self._run_single(
                selection, positional_filter, requested_gtest_filter
            )
        if selection.executor == "gtest-multi":
            return self._run_multi(
                selection, positional_filter, requested_gtest_filter
            )
        if selection.executor == "fa3-profile":
            return self._run_fa3(
                selection, positional_filter, requested_gtest_filter
            )
        if selection.executor == "trace":
            return self._run_trace(selection, positional_filter)
        if selection.executor == "build-only":
            raise RunnerError(
                f"{selection.label} is build-only and cannot be run with the "
                "generic runner\nUse its test-specific interface with explicit "
                "runtime arguments"
            )
        raise RunnerError(
            f"Unknown executor '{selection.executor}' for "
            f"{selection.architecture.name}/{selection.test_group}"
        )

    def _run_single(
        self,
        selection: Selection,
        positional_filter: str,
        requested_gtest_filter: str,
    ) -> int:
        if not self.builder.config_dir.is_dir():
            raise RunnerError(
                f"Configuration directory not found: {self.builder.config_dir}"
            )
        self.builder.build(selection)
        binaries = self._binary_paths(selection.binary_group)
        if len(binaries) != 1:
            raise RunnerError(
                f"Expected one binary in group '{selection.binary_group}', "
                f"found {len(binaries)}"
            )
        binary = binaries[0]

        gtest_filter = selection.default_filter
        user_filter = requested_gtest_filter
        if not user_filter and positional_filter:
            user_filter = f"*{positional_filter}*"
        if user_filter:
            if selection.default_filter == "*":
                gtest_filter = user_filter
            else:
                intersection = self.gtest.intersection(
                    binary,
                    self.builder.config_dir,
                    selection.default_filter,
                    user_filter,
                )
                if intersection is None:
                    raise RunnerError(
                        f"No tests in {selection.label} matched: {user_filter}"
                    )
                gtest_filter = intersection

        shown_filter = requested_gtest_filter or positional_filter or "all"
        self.ui.info(
            f"Running {selection.label}: {shown_filter} "
            f"(config: {self.settings.gpu_config})"
        )
        if self.settings.sass_timing:
            self.ui.info(
                "Requested SASS timing (explicit config mode takes precedence)"
            )
        elif self.settings.sass_auto:
            self.ui.info(
                "Requested SASS functional (explicit config mode takes precedence)"
            )
        elif self.settings.sassir:
            self.ui.info(
                "Requested SASSIR debug image (explicit config mode takes precedence): "
                f"{self.settings.sassir}"
            )
        return_code = self.gtest.run_binary(
            binary,
            self.builder.config_dir,
            gtest_filter,
            env=self._gtest_environment(selection, binary),
        )
        if return_code == 0:
            self.ui.success(f"✓ {selection.label} passed!")
        else:
            self.ui.color(
                RED, f"✗ {selection.label} failed (exit code: {return_code})"
            )
        return return_code

    @staticmethod
    def _multi_requested_filter(
        positional_filter: str, requested_gtest_filter: str
    ) -> str:
        if requested_gtest_filter:
            return requested_gtest_filter
        if not positional_filter:
            return "*"
        result = f"*{positional_filter}*"
        if "." in positional_filter and "/" not in positional_filter:
            suite, case = positional_filter.split(".", 1)
            result += f":*{suite}/*.{case}*"
        return result

    def _select_multi_binaries(
        self,
        selection: Selection,
        requested_filter: str,
    ) -> list[tuple[Path, str]]:
        selected: list[tuple[Path, str]] = []
        for binary in self._binary_paths(selection.binary_group):
            if not binary.is_file():
                continue
            gtest_filter = selection.default_filter
            if requested_filter != "*":
                if selection.default_filter == "*":
                    gtest_filter = requested_filter
                else:
                    intersection = self.gtest.intersection(
                        binary,
                        self.builder.config_dir,
                        selection.default_filter,
                        requested_filter,
                    )
                    if intersection is None:
                        continue
                    gtest_filter = intersection
            if self.gtest.matches(binary, self.builder.config_dir, gtest_filter):
                selected.append((binary, gtest_filter))
        return selected

    def _run_multi(
        self,
        selection: Selection,
        positional_filter: str,
        requested_gtest_filter: str,
    ) -> int:
        if not self.builder.config_dir.is_dir():
            raise RunnerError(
                f"Configuration directory not found: {self.builder.config_dir}"
            )
        self.builder.build(selection)
        requested_filter = self._multi_requested_filter(
            positional_filter, requested_gtest_filter
        )
        shown_filter = requested_gtest_filter or positional_filter or "all"
        self.ui.info(
            f"Running {selection.label}: {shown_filter} "
            f"(config: {self.settings.gpu_config})"
        )
        selected = self._select_multi_binaries(selection, requested_filter)
        if not selected:
            raise RunnerError(
                f"No {selection.label} binaries matched filter: {shown_filter}"
            )

        exit_code = 0
        for binary, gtest_filter in selected:
            self.ui.info(f"Running {binary.name}...")
            return_code = self.gtest.run_binary(
                binary,
                self.builder.config_dir,
                gtest_filter,
                env=self._gtest_environment(selection, binary),
            )
            if return_code != 0:
                exit_code = return_code
        if exit_code == 0:
            self.ui.success(f"✓ {selection.label} passed!")
        else:
            self.ui.color(
                RED, f"✗ {selection.label} failed (exit code: {exit_code})"
            )
        return exit_code

    def _run_fa3(
        self,
        selection: Selection,
        positional_filter: str,
        requested_gtest_filter: str,
    ) -> int:
        if not self.builder.config_dir.is_dir():
            raise RunnerError(
                f"Configuration directory not found: {self.builder.config_dir}"
            )
        requested_filter = (
            requested_gtest_filter
            or (f"*{positional_filter}*" if positional_filter else "*")
        )
        self.builder.build(selection, parallel=True)

        matched = 0
        exit_code = 0
        last_filter = selection.default_filter
        for binary in self._binary_paths(selection.binary_group):
            if not binary.is_file():
                raise RunnerError(f"FA3 gtest executable not found: {binary}")
            gtest_filter = selection.default_filter
            if requested_filter != "*":
                intersection = self.gtest.intersection(
                    binary,
                    self.builder.config_dir,
                    selection.default_filter,
                    requested_filter,
                )
                if intersection is None:
                    continue
                gtest_filter = intersection
            last_filter = gtest_filter
            if not self.gtest.matches(
                binary, self.builder.config_dir, gtest_filter
            ):
                continue

            matched += 1
            self.ui.info(
                f"Running {selection.label}: {binary.name} "
                f"(config: {self.settings.gpu_config})"
            )
            if self.settings.sass_timing:
                self.ui.info(
                    "Requested SASS timing (explicit config mode takes precedence)"
                )
            elif self.settings.sass_auto:
                self.ui.info(
                    "Requested SASS functional (explicit config mode takes precedence)"
                )
            return_code = self.gtest.run_binary(
                binary,
                self.builder.config_dir,
                gtest_filter,
                env=self._gtest_environment(
                    selection,
                    binary,
                    FA3_H1D128_PROFILE_CASE_LIST=selection.case_list
                ),
            )
            if return_code != 0:
                exit_code = return_code

        if matched == 0:
            raise RunnerError(f"No FA3 binaries matched filter: {last_filter}")
        if exit_code == 0:
            self.ui.success(f"✓ {selection.label} passed!")
        else:
            self.ui.color(
                RED, f"✗ {selection.label} failed (exit code: {exit_code})"
            )
        return exit_code

    def _run_trace(self, selection: Selection, test_name: str) -> int:
        self.builder.build(selection, parallel=False)
        self.ui.info("Running trace tests...")
        binary_dir = self.test_dir / "build" / "trace" / "bin"
        tests: list[tuple[str, list[str]]] = []
        if not test_name or test_name in "embedding":
            tests.append(
                ("gpt2_embedding", [str(binary_dir / "gpt2_embedding_test")])
            )
        for name in ("gelu", "flash_attn", "layernorm", "residual_add", "linear"):
            if test_name and test_name not in name:
                continue
            tests.append(
                (
                    f"gpt2_{name}",
                    [str(binary_dir / "gpt2_data_driven_test"), name],
                )
            )
        if not tests:
            raise RunnerError(f"No trace tests matched pattern: {test_name}")

        exit_code = 0
        for label, command in tests:
            self.ui.info(f"--- {label} ---")
            if self.ui.run(command, cwd=binary_dir) != 0:
                exit_code = 1
                self.ui.color(RED, f"FAILED: {label}")
        if exit_code == 0:
            self.ui.success("✓ Trace tests passed!")
        else:
            self.ui.color(RED, "✗ Trace tests failed!")
        return exit_code
