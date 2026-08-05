#!/usr/bin/env python3
"""Regression checks for architecture/test-set CI planning."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
CI_RUNNER = REPO_ROOT / "test" / "ci" / "run_ci_tests.sh"


def planner(architecture: str, test_set: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        CI_ARCH=architecture,
        CI_TEST_SET=test_set,
        CI_LIST_JOBS="1",
    )
    return subprocess.run(
        [str(CI_RUNNER)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


class CiPlannerTest(unittest.TestCase):
    def assert_plan(
        self, architecture: str, test_set: str, expected: list[str]
    ) -> None:
        result = planner(architecture, test_set)
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertEqual(result.stdout.splitlines(), expected)
        self.assertEqual(result.stderr, "")

    def test_core_is_intersected_with_each_architecture_manifest(self) -> None:
        self.assert_plan(
            "sm120",
            "core",
            [
                "sm120|core|unit",
                "sm120|core|integration",
                "sm120|core|barrier",
                "sm120|core|tma",
                "sm120|core|mma",
                "sm120|core|trace",
            ],
        )
        self.assert_plan(
            "sm90",
            "core",
            [
                "sm90|core|integration",
                "sm90|core|barrier",
                "sm90|core|wgmma",
            ],
        )

    def test_test_sets_are_intersected_with_architecture_manifests(self) -> None:
        self.assert_plan(
            "all", "fa2", ["sm120|fa2|fa2", "sm90|fa2|fa2"]
        )
        self.assert_plan("all", "fa3", ["sm90|fa3|fa3"])

    def test_explicit_incompatible_pair_is_rejected(self) -> None:
        self.assert_plan("sm120", "fa2", ["sm120|fa2|fa2"])
        result = planner("sm120", "fa3")
        self.assertEqual(result.returncode, 2)
        self.assertIn("has no jobs", result.stdout)


if __name__ == "__main__":
    unittest.main()
