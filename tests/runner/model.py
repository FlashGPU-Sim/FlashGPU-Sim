"""Shared data models for test selection and execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Architecture:
    name: str
    default_config: str
    nvcc_target: str
    compute_capability: str


@dataclass(frozen=True)
class Selection:
    architecture: Architecture
    test_group: str
    profile: str = ""
    mode: str = ""
    build_target: str = ""
    binary_group: str = ""
    executor: str = ""
    default_filter: str = "*"
    case_list: str = ""

    @property
    def label(self) -> str:
        parts = [self.architecture.name, self.test_group]
        if self.profile:
            parts.append(self.profile)
        if self.mode:
            parts.append(self.mode)
        return "/".join(parts)


@dataclass
class Settings:
    timeout: int = 3600
    verbose: int = 1
    debug: bool = False
    test_build_jobs: int = 4
    simulator_build_jobs: int = 4
    gpu_config: str = "SM120_RTX5090"
    gpu_config_explicit: bool = False
