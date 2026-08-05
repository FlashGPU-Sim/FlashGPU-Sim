"""Resolve public architecture/test-group selections into build metadata."""

from __future__ import annotations

import re

from .errors import RunnerError
from .make import MakeInterface
from .model import Architecture, Selection


NVCC_TARGET_RE = re.compile(r"^sm_([0-9]+)[af]?$")


def compute_capability_from_nvcc_target(nvcc_target: str) -> str:
    match = NVCC_TARGET_RE.fullmatch(nvcc_target)
    if match is None or len(match.group(1)) < 2:
        raise RunnerError(
            f"Invalid nvcc_target in architecture manifest: {nvcc_target}"
        )
    digits = match.group(1)
    return f"{int(digits[:-1])}.{int(digits[-1])}"


class SelectionResolver:
    def __init__(self, make: MakeInterface) -> None:
        self.make = make

    @staticmethod
    def _available(label: str, values: list[str]) -> str:
        return f"{label}: {' '.join(values)}"

    def architecture(self, name: str) -> Architecture:
        if not name:
            raise RunnerError("--arch is required")
        architectures = self.make.architectures()
        if name not in architectures:
            raise RunnerError(
                f"Unknown architecture: {name}\n"
                + self._available("Available architectures", architectures)
            )
        default_config, nvcc_target = self.make.architecture_metadata(name)
        return Architecture(
            name=name,
            default_config=default_config,
            nvcc_target=nvcc_target,
            compute_capability=compute_capability_from_nvcc_target(nvcc_target),
        )

    def selection(
        self,
        architecture: Architecture,
        test_group: str,
        profile: str = "",
        mode: str = "",
    ) -> Selection:
        if not test_group:
            raise RunnerError("--test-group is required")

        test_groups = self.make.test_groups(architecture.name)
        if test_group not in test_groups:
            raise RunnerError(
                f"Unsupported test group '{test_group}' for '{architecture.name}'\n"
                + self._available("Available test groups", test_groups)
            )

        profiles = self.make.profiles(architecture.name, test_group)
        if profiles and not profile:
            raise RunnerError(
                f"{architecture.name}/{test_group} requires --profile\n"
                + self._available("Available profiles", profiles)
            )
        if not profiles and profile:
            raise RunnerError(
                f"{architecture.name}/{test_group} does not accept --profile"
            )
        if profile and profile not in profiles:
            raise RunnerError(
                f"Unknown profile '{profile}' for {architecture.name}/{test_group}\n"
                + self._available("Available profiles", profiles)
            )

        modes = (
            self.make.modes(architecture.name, test_group, profile)
            if profile
            else []
        )
        if modes and not mode:
            raise RunnerError(
                f"{architecture.name}/{test_group}/{profile} requires --mode\n"
                + self._available("Available modes", modes)
            )
        if not modes and mode:
            raise RunnerError(
                f"{architecture.name}/{test_group}/{profile or 'default'} "
                "does not accept --mode"
            )
        if mode and mode not in modes:
            raise RunnerError(
                f"Unknown mode '{mode}' for "
                f"{architecture.name}/{test_group}/{profile}\n"
                + self._available("Available modes", modes)
            )

        metadata = self.make.selection_metadata(
            architecture.name, test_group, profile, mode
        )
        return Selection(
            architecture=architecture,
            test_group=test_group,
            profile=profile,
            mode=mode,
            build_target=metadata[0],
            binary_group=metadata[1],
            executor=metadata[2],
            default_filter=metadata[3],
            case_list=metadata[4],
        )

    def aggregate_plan(self, architecture: Architecture) -> list[Selection]:
        """Expand an architecture into unique, buildable Make targets."""
        candidates: list[tuple[str, str, str]] = []
        for test_group in self.make.test_groups(architecture.name):
            profiles = self.make.profiles(architecture.name, test_group)
            if not profiles:
                candidates.append((test_group, "", ""))
                continue
            for profile in profiles:
                modes = self.make.modes(architecture.name, test_group, profile)
                if not modes:
                    candidates.append((test_group, profile, ""))
                elif "all" in modes:
                    candidates.append((test_group, profile, "all"))
                else:
                    candidates.extend(
                        (test_group, profile, mode) for mode in modes
                    )

        plan: list[Selection] = []
        seen_targets: set[str] = set()
        for test_group, profile, mode in candidates:
            selection = self.selection(
                architecture, test_group, profile=profile, mode=mode
            )
            if selection.build_target in seen_targets:
                continue
            seen_targets.add(selection.build_target)
            plan.append(selection)

        if not plan:
            raise RunnerError(
                f"Architecture '{architecture.name}' has no buildable selections"
            )
        return plan
