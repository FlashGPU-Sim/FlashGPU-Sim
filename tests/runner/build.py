"""Configuration setup and Make build orchestration."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

from .errors import RunnerError
from .make import MakeInterface
from .model import Architecture, Selection, Settings
from .selection import SelectionResolver
from .ui import UI


class BuildManager:
    def __init__(
        self,
        test_dir: Path,
        settings: Settings,
        make: MakeInterface,
        ui: UI,
    ) -> None:
        self.test_dir = test_dir
        self.repo_root = test_dir.parent
        self.settings = settings
        self.make = make
        self.ui = ui

    @property
    def config_dir(self) -> Path:
        return self.test_dir / "run" / self.settings.gpu_config

    def configure_architecture(self, architecture: Architecture) -> None:
        if not self.settings.gpu_config_explicit:
            self.settings.gpu_config = architecture.default_config
        self.validate_config(architecture)

    def validate_config(self, architecture: Architecture) -> None:
        config_file = (
            self.repo_root
            / "configs"
            / self.settings.gpu_config
            / "gpgpusim.config"
        )
        if not config_file.is_file():
            raise RunnerError(
                f"Configuration '{self.settings.gpu_config}' is missing {config_file}"
            )

        major = ""
        minor = ""
        for line in config_file.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 2:
                continue
            if fields[0] == "-gpgpu_compute_capability_major":
                major = fields[1]
            elif fields[0] == "-gpgpu_compute_capability_minor":
                minor = fields[1]
        if not major.isdigit() or not minor.isdigit():
            raise RunnerError(f"Unable to read compute capability from {config_file}")

        actual = f"{int(major)}.{int(minor)}"
        if actual != architecture.compute_capability:
            raise RunnerError(
                f"{architecture.name} requires compute capability "
                f"{architecture.compute_capability} ({architecture.nvcc_target}), "
                f"but {self.settings.gpu_config} declares {actual}"
            )

    def setup_run_directory(self) -> None:
        config_name = self.settings.gpu_config
        source = self.repo_root / "configs" / config_name
        destination = self.test_dir / "run" / config_name
        self.ui.warning(
            f"Setting up test run directory for config: {config_name}..."
        )
        if not source.is_dir():
            raise RunnerError(
                f"Configuration '{config_name}' not found at {source}"
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        self.ui.warning(f"⚡ Running: cp -r {source} {destination.parent}/")
        shutil.copytree(source, destination, dirs_exist_ok=True)
        self.ui.success(f"Synced {config_name} configuration to run directory")

    def setup_environment(self) -> None:
        self.ui.info("Setting up FlashGPU-Sim test environment...")
        for prerequisite in ("g++", "make", "git"):
            if shutil.which(prerequisite) is None:
                raise RunnerError(f"Error: {prerequisite} not found")
        self.ui.info("Setting up Google Test...")
        if self.ui.run(["make", "setup-gtest"], cwd=self.test_dir) != 0:
            raise RunnerError("GoogleTest setup failed")
        self.setup_run_directory()
        self.ui.success("Environment setup complete!")

    def is_native_mode(self) -> bool:
        if os.environ.get("GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN"):
            return False
        simulator_lib = f"{self.repo_root}/lib/"
        if any(
            simulator_lib in entry
            for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":")
        ):
            self.ui.warning(
                "Warning: LD_LIBRARY_PATH contains simulator paths but "
                "GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN is not set"
            )
            self.ui.warning(
                "Treating as simulator mode to avoid contamination"
            )
            return False
        return True

    def build_gpgpusim(self) -> None:
        if self.is_native_mode():
            self.ui.info(
                "Native GPU mode detected - skipping FlashGPU-Sim library build"
            )
            return

        self.ui.info("Building FlashGPU-Sim library...")
        libraries = sorted((self.repo_root / "lib").rglob("libcudart.so"))
        library = libraries[0] if libraries else None
        needs_rebuild = library is None or library.stat().st_size == 0
        if not needs_rebuild and library is not None:
            library_mtime = library.stat().st_mtime
            needs_rebuild = any(
                source.stat().st_mtime > library_mtime
                for source in (self.repo_root / "src").rglob("*.cc")
            )

        if needs_rebuild:
            self.ui.warning("FlashGPU-Sim library needs rebuild...")
            command = [
                "make",
                "FLASH=1",
                f"-j{self.settings.simulator_build_jobs}",
            ]
            if self.ui.run(command, cwd=self.repo_root) != 0:
                raise RunnerError("FlashGPU-Sim library build failed")
            self.ui.success("FlashGPU-Sim library built successfully")
        else:
            self.ui.success("FlashGPU-Sim library is up to date")

    def build(self, selection: Selection, *, parallel: bool | None = None) -> None:
        self.build_gpgpusim()
        if parallel is None:
            parallel = selection.architecture.compute_capability == "9.0"

        self.ui.info(
            f"Building {selection.architecture.name}/{selection.test_group} via "
            f"'{selection.build_target}' (NVCC target: "
            f"{selection.architecture.nvcc_target})..."
        )
        command = ["make"]
        if parallel:
            command.append(f"-j{self.settings.test_build_jobs}")
        if self.settings.debug:
            command.append(
                "CXXFLAGS=-std=c++17 -Wall -Wextra -pthread -g -O0 -DDEBUG"
            )
        command.extend(
            [
                f"ARCH={selection.architecture.name}",
                f"GPU_CONFIG={self.settings.gpu_config}",
                selection.build_target,
            ]
        )
        if self.ui.run(command, cwd=self.test_dir) != 0:
            raise RunnerError("Build failed!")
        self.ui.success("Build successful!")

    def build_all(
        self, architecture: Architecture, resolver: SelectionResolver
    ) -> list[Selection]:
        plan = resolver.aggregate_plan(architecture)
        self.ui.info(f"Building all test groups for {architecture.name}...")
        for selection in plan:
            self.ui.info(
                f"Aggregate selection: {selection.label} -> "
                f"{selection.build_target}"
            )
            self.build(selection)
        self.ui.success(
            f"Built {len(plan)} unique targets for {architecture.name}"
        )
        return plan

    def clean(self) -> None:
        self.ui.info("Cleaning test build artifacts...")
        if self.ui.run(["make", "clean"], cwd=self.test_dir) != 0:
            raise RunnerError("Clean failed!")
        self.ui.success("Clean complete!")

    def list_configs(self, program: str) -> None:
        self.ui.info("Available GPU configurations:")
        configs_dir = self.repo_root / "configs"
        if not configs_dir.is_dir():
            raise RunnerError(
                f"Error: configs directory not found at {configs_dir}"
            )

        found = False
        for config_path in sorted(configs_dir.iterdir()):
            if not config_path.is_dir() or not (
                config_path / "gpgpusim.config"
            ).is_file():
                continue
            found = True
            if config_path.name == self.settings.gpu_config:
                self.ui.success(f"  ✓ {config_path.name} (default)")
            else:
                self.ui.plain(f"    {config_path.name}")
        if not found:
            self.ui.warning(f"No GPU configurations found in {configs_dir}")

        self.ui.plain()
        self.ui.info("To use a config:")
        self.ui.plain(
            f"  {program} -c CONFIG_NAME run --arch sm120 "
            "--group integration"
        )
        self.ui.plain(
            f"  {program} --config CONFIG_NAME run --arch sm90 "
            "--group wgmma"
        )
