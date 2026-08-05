#!/usr/bin/env python3
"""Regression checks for architecture-wide runner builds."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = REPO_ROOT / "test"
RUNNER = TEST_DIR / "run_tests.py"
sys.path.insert(0, str(TEST_DIR))

from runner.make import MakeInterface  # noqa: E402
from runner.selection import SelectionResolver  # noqa: E402


def aggregate_plan(architecture: str) -> list[str]:
    resolver = SelectionResolver(MakeInterface(TEST_DIR))
    resolved_architecture = resolver.architecture(architecture)
    return [
        "|".join(
            [
                selection.architecture.name,
                selection.test_group,
                selection.profile,
                selection.mode,
                selection.build_target,
            ]
        )
        for selection in resolver.aggregate_plan(resolved_architecture)
    ]


class RunnerAllBuildTest(unittest.TestCase):
    def assert_plan(self, architecture: str, expected: list[str]) -> None:
        plan = aggregate_plan(architecture)
        self.assertEqual(plan, expected)
        targets = [row.rsplit("|", 1)[1] for row in plan]
        self.assertEqual(len(targets), len(set(targets)))

    def test_sm120_builds_every_manifest_selection(self) -> None:
        self.assert_plan(
            "sm120",
            [
                "sm120|unit|||build-sm120-unit",
                "sm120|integration|||build-sm120-integration",
                "sm120|barrier|||build-sm120-barrier",
                "sm120|tma|||build-sm120-tma",
                "sm120|mma|||build-sm120-mma",
                "sm120|fa2|smoke||fa2-sm120-smoke",
                "sm120|fa2|small||fa2-sm120-small",
                "sm120|fa2|medium||fa2-sm120-medium",
                "sm120|fa2|large||fa2-sm120-large",
                "sm120|fa2|breakdown|all|fa2-sm120-breakdown",
                "sm120|fa2|scaling|all|fa2-sm120-scaling",
                "sm120|fa2|concurrency|all|fa2-sm120-concurrency",
                "sm120|microbench|mbarrier||microbench-sm120-mbarrier",
                "sm120|microbench|mma||microbench-sm120-mma",
                "sm120|microbench|memory||microbench-sm120-memory",
                "sm120|trace|gpt2||trace-sm120-gpt2",
            ],
        )

    def test_sm90_builds_every_manifest_selection(self) -> None:
        self.assert_plan(
            "sm90",
            [
                "sm90|integration|||build-sm90-integration",
                "sm90|barrier|||build-sm90-barrier",
                "sm90|wgmma|||build-sm90-wgmma",
                "sm90|fa2|smoke||fa2-sm90-smoke",
                "sm90|fa2|small||fa2-sm90-small",
                "sm90|fa2|medium||fa2-sm90-medium",
                "sm90|fa2|large||fa2-sm90-large",
                "sm90|fa2|breakdown|all|fa2-sm90-breakdown",
                "sm90|fa2|scaling|all|fa2-sm90-scaling",
                "sm90|fa2|concurrency|all|fa2-sm90-concurrency",
                "sm90|fa3|smoke||fa3-standard",
                "sm90|fa3|packgqa||fa3-packgqa",
                "sm90|fa3|breakdown|all|fa3-modes",
                "sm90|microbench|cp-async||microbench-sm90-cp-async",
                "sm90|microbench|mma||microbench-sm90-mma",
                "sm90|microbench|tma||microbench-sm90-tma",
                "sm90|microbench|wgmma||microbench-sm90-wgmma",
            ],
        )

    def test_all_rejects_narrowing_and_non_build_actions(self) -> None:
        for arguments, expected_error in [
            (
                [
                    "build",
                    "--arch",
                    "sm120",
                    "--test-group",
                    "all",
                    "--profile",
                    "smoke",
                ],
                "does not accept --profile or --mode",
            ),
            (
                ["run", "--arch", "sm120", "--test-group", "all"],
                "only valid with build",
            ),
        ]:
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [str(RUNNER), *arguments],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn(expected_error, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
