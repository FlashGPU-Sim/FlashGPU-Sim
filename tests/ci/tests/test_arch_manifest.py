#!/usr/bin/env python3
"""Regression checks for TOML architecture manifests and build metadata."""

from __future__ import annotations

import re
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_DIR = REPO_ROOT / "tests"
sys.path.insert(0, str(TEST_DIR / "scripts"))

from arch_manifest import load_manifests  # noqa: E402


VALID_EXECUTORS = {
    "build-only",
    "fa3-profile",
    "gtest-multi",
    "gtest-single",
    "trace",
}


def make_lines(target: str, **variables: str) -> list[str]:
    command = [
        "make",
        "-s",
        "--no-print-directory",
        "-C",
        str(TEST_DIR),
        target,
        *(f"{name}={value}" for name, value in variables.items()),
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


class ArchitectureManifestTest(unittest.TestCase):
    def test_toml_is_the_source_of_architecture_and_source_membership(self) -> None:
        manifests = {manifest.name: manifest for manifest in load_manifests()}
        self.assertEqual(make_lines("list-architectures"), sorted(manifests))

        for architecture, manifest in manifests.items():
            with self.subTest(architecture=architecture):
                metadata = make_lines(
                    "print-architecture-metadata", ARCH=architecture
                )
                self.assertEqual(
                    metadata, [f"{manifest.config}|{manifest.nvcc_target}"]
                )
                self.assertRegex(manifest.nvcc_target, r"^sm_[0-9]+[af]?$")
                self.assertTrue(
                    (REPO_ROOT / "configs" / manifest.config / "gpgpusim.config").is_file()
                )

                test_groups = make_lines("list-test-groups", ARCH=architecture)
                self.assertEqual(test_groups, list(manifest.test_groups))
                self.assertEqual(len(test_groups), len(set(test_groups)))

                for test_group in test_groups:
                    sources = make_lines(
                        "print-test-group-sources",
                        ARCH=architecture,
                        TEST_GROUP=test_group,
                    )
                    expected = [
                        f"src/{source}"
                        for source in manifest.sources_for(test_group)
                    ]
                    self.assertEqual(sources, expected)
                    for source in sources:
                        self.assertTrue((TEST_DIR / source).is_file(), source)

    def test_every_public_test_group_selection_is_complete(self) -> None:
        binary_groups = set(make_lines("list-binary-groups"))
        referenced_binary_groups: set[str] = set()
        selections = 0

        for architecture in make_lines("list-architectures"):
            test_groups = make_lines("list-test-groups", ARCH=architecture)
            self.assertTrue(test_groups)

            for test_group in test_groups:
                profiles = make_lines(
                    "list-test-group-profiles",
                    ARCH=architecture,
                    TEST_GROUP=test_group,
                )
                if not profiles:
                    selection_variants: list[tuple[str, str]] = [("", "")]
                else:
                    self.assertEqual(len(profiles), len(set(profiles)))
                    selection_variants = []
                    for profile in profiles:
                        modes = make_lines(
                            "list-test-group-modes",
                            ARCH=architecture,
                            TEST_GROUP=test_group,
                            PROFILE=profile,
                        )
                        self.assertEqual(len(modes), len(set(modes)))
                        if modes:
                            selection_variants.extend((profile, mode) for mode in modes)
                        else:
                            selection_variants.append((profile, ""))

                for profile, mode in selection_variants:
                    with self.subTest(
                        architecture=architecture,
                        test_group=test_group,
                        profile=profile,
                        mode=mode,
                    ):
                        rows = make_lines(
                            "print-test-group-metadata",
                            ARCH=architecture,
                            TEST_GROUP=test_group,
                            PROFILE=profile,
                            MODE=mode,
                        )
                        self.assertEqual(len(rows), 1)
                        fields = rows[0].split("|")
                        self.assertEqual(len(fields), 5)
                        build_target, binary_group, executor, default_filter, _ = fields
                        self.assertTrue(build_target)
                        self.assertIn(binary_group, binary_groups)
                        self.assertIn(executor, VALID_EXECUTORS)
                        self.assertTrue(default_filter)
                        referenced_binary_groups.add(binary_group)
                        selections += 1

        self.assertGreater(selections, 0)
        for binary_group in referenced_binary_groups - {"none"}:
            with self.subTest(binary_group=binary_group):
                binaries = make_lines(
                    "print-binary-group", BINARY_GROUP=binary_group
                )
                self.assertTrue(binaries)
                for binary in binaries:
                    self.assertTrue(re.match(r"^build[^/]*/bin/", binary), binary)


if __name__ == "__main__":
    unittest.main()
