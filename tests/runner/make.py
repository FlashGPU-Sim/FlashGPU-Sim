"""Machine-readable and build-facing Make integration."""

from __future__ import annotations

from pathlib import Path
import subprocess

from .errors import RunnerError


class MakeInterface:
    def __init__(self, test_dir: Path) -> None:
        self.test_dir = test_dir

    def query(self, target: str, **variables: str) -> str:
        command = ["make", "-s", "--no-print-directory", target]
        command.extend(f"{key}={value}" for key, value in variables.items())
        completed = subprocess.run(
            command,
            cwd=self.test_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            if detail:
                raise RunnerError(detail)
            raise RunnerError(f"Make query failed: {target}")
        return completed.stdout.rstrip("\n")

    def lines(self, target: str, **variables: str) -> list[str]:
        return [
            line.strip()
            for line in self.query(target, **variables).splitlines()
            if line.strip()
        ]

    def architectures(self) -> list[str]:
        return self.lines("list-architectures")

    def test_groups(self, architecture: str) -> list[str]:
        return self.lines("list-test-groups", ARCH=architecture)

    def profiles(self, architecture: str, test_group: str) -> list[str]:
        return self.lines(
            "list-test-group-profiles", ARCH=architecture, TEST_GROUP=test_group
        )

    def modes(
        self, architecture: str, test_group: str, profile: str
    ) -> list[str]:
        return self.lines(
            "list-test-group-modes",
            ARCH=architecture,
            TEST_GROUP=test_group,
            PROFILE=profile,
        )

    def architecture_metadata(self, architecture: str) -> tuple[str, str]:
        fields = self.query(
            "print-architecture-metadata", ARCH=architecture
        ).split("|", 1)
        if len(fields) != 2 or not all(fields):
            raise RunnerError(f"Incomplete architecture manifest: {architecture}")
        return fields[0], fields[1]

    def selection_metadata(
        self,
        architecture: str,
        test_group: str,
        profile: str,
        mode: str,
    ) -> tuple[str, str, str, str, str]:
        fields = self.query(
            "print-test-group-metadata",
            ARCH=architecture,
            TEST_GROUP=test_group,
            PROFILE=profile,
            MODE=mode,
        ).split("|", 4)
        if len(fields) != 5 or not all(fields[:4]):
            raise RunnerError("Incomplete test-group build metadata")
        return fields[0], fields[1], fields[2], fields[3], fields[4]

    def binaries(self, binary_group: str) -> list[str]:
        return self.lines("print-binary-group", BINARY_GROUP=binary_group)
