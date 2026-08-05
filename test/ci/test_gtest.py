#!/usr/bin/env python3
"""Regression tests for the runner's GoogleTest integration."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = REPO_ROOT / "test"
sys.path.insert(0, str(TEST_DIR))

from runner.executors import TestExecutors  # noqa: E402
from runner.gtest import GTest  # noqa: E402
from runner.model import Architecture, Selection, Settings  # noqa: E402
from runner.ui import UI  # noqa: E402


FAKE_GTEST = r"""#!/usr/bin/env bash
set -euo pipefail

mode=run
for arg in "$@"; do
  if [[ "$arg" == "--gtest_list_tests" ]]; then
    mode=list
    break
  fi
done

if [[ "$mode" == list ]]; then
  printf '%s\n' \
    'PositiveSuite.' \
    '  Passes' \
    'Values/ParameterizedSuite.' \
    '  Works/0  # GetParam() = 42' \
    'NegativeSuite.' \
    '  Hidden'
  if [[ -n "${GTEST_OUTPUT:-}" ]]; then
    output_dir="${GTEST_OUTPUT#xml:}"
    mkdir -p "$output_dir"
    printf '<testsuites tests="1" name="Discovery" />\n' \
      > "${output_dir}discovery.xml"
  fi
  exit 0
fi

[[ "${GTEST_OUTPUT:-}" == xml:* ]]
output_dir="${GTEST_OUTPUT#xml:}"
mkdir -p "$output_dir"
printf '%s\n' \
  '<?xml version="1.0" encoding="UTF-8"?>' \
  '<testsuites tests="1" failures="0" errors="0" name="AllTests">' \
  '  <testsuite name="PositiveSuite" tests="1" failures="0" errors="0">' \
  '    <testcase name="Passes" status="run" result="completed" classname="PositiveSuite" />' \
  '  </testsuite>' \
  '</testsuites>' > "${output_dir}fake_gtest.xml"
"""

FAKE_GTEST_SECOND = r"""#!/usr/bin/env bash
set -euo pipefail

for arg in "$@"; do
  if [[ "$arg" == "--gtest_list_tests" ]]; then
    printf 'SecondSuite.\n  Other\n'
    exit 0
  fi
done
exit 1
"""


class FakeMake:
    def __init__(self, binaries: list[Path]) -> None:
        self._binaries = binaries

    def binaries(self, _binary_group: str) -> list[str]:
        return [str(binary) for binary in self._binaries]


class FakeBuilder:
    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir
        self.setup_calls = 0
        self.build_calls = 0

    def setup_run_directory(self) -> None:
        self.setup_calls += 1

    def build(self, _selection: Selection) -> None:
        self.build_calls += 1


class GTestDiscoveryOutputTest(unittest.TestCase):
    def test_discovery_suppresses_xml_and_execution_writes_one_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            xml_dir = root / "xml"
            fake_gtest = root / "fake_gtest"
            fake_gtest_second = root / "fake_gtest_second"
            config_dir.mkdir()
            xml_dir.mkdir()
            fake_gtest.write_text(FAKE_GTEST, encoding="utf-8")
            fake_gtest.chmod(0o755)
            fake_gtest_second.write_text(FAKE_GTEST_SECOND, encoding="utf-8")
            fake_gtest_second.chmod(0o755)

            gtest = GTest(UI(), timeout=10)
            with mock.patch.dict(
                os.environ, {"GTEST_OUTPUT": f"xml:{xml_dir}/"}
            ):
                self.assertEqual(
                    gtest.list_cases(fake_gtest, config_dir),
                    [
                        "PositiveSuite.Passes",
                        "Values/ParameterizedSuite.Works/0",
                        "NegativeSuite.Hidden",
                    ],
                )
                self.assertTrue(
                    gtest.matches(fake_gtest, config_dir, "PositiveSuite.*")
                )
                self.assertEqual(list(xml_dir.glob("*.xml")), [])
                self.assertEqual(
                    gtest.intersection(
                        fake_gtest,
                        config_dir,
                        "PositiveSuite.*",
                        "*Passes*",
                    ),
                    "PositiveSuite.Passes",
                )
                self.assertEqual(
                    gtest.group_cases(
                        [fake_gtest, fake_gtest_second],
                        config_dir,
                        "PositiveSuite.*:SecondSuite.*",
                    ),
                    ["PositiveSuite.Passes", "SecondSuite.Other"],
                )
                self.assertEqual(
                    gtest.group_cases(
                        [fake_gtest, fake_gtest_second],
                        config_dir,
                        "PositiveSuite.*:SecondSuite.*",
                        "*Other",
                    ),
                    ["SecondSuite.Other"],
                )
                self.assertEqual(list(xml_dir.glob("*.xml")), [])
                environment = os.environ.copy()
                return_code = gtest.run_binary(
                    fake_gtest,
                    config_dir,
                    "PositiveSuite.Passes",
                    env=environment,
                )
                self.assertEqual(return_code, 0)

            xml_files = sorted(xml_dir.glob("*.xml"))
            self.assertEqual([path.name for path in xml_files], ["fake_gtest.xml"])
            result = ET.parse(xml_files[0]).getroot()
            self.assertEqual(result.tag, "testsuites")
            self.assertEqual(result.attrib["tests"], "1")
            self.assertEqual(result.attrib["failures"], "0")
            testcase = result.find("./testsuite/testcase")
            self.assertIsNotNone(testcase)
            assert testcase is not None
            self.assertEqual(testcase.attrib["status"], "run")
            self.assertEqual(testcase.attrib["result"], "completed")

    def test_list_cases_uses_the_selected_binary_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "run" / "config"
            fake_gtest = root / "fake_gtest"
            config_dir.mkdir(parents=True)
            fake_gtest.write_text(FAKE_GTEST, encoding="utf-8")
            fake_gtest.chmod(0o755)

            diagnostics = io.StringIO()
            ui = UI(diagnostics)
            settings = Settings(gpu_config="config")
            fake_builder = FakeBuilder(config_dir)
            selection = Selection(
                architecture=Architecture(
                    "sm120", "config", "sm_120a", "12.0"
                ),
                test_group="integration",
                build_target="fake-build",
                binary_group="fake-group",
                executor="gtest-single",
                default_filter="PositiveSuite.*",
            )
            executors = TestExecutors(
                TEST_DIR,
                settings,
                FakeMake([fake_gtest]),  # type: ignore[arg-type]
                fake_builder,  # type: ignore[arg-type]
                GTest(ui, timeout=10),
                ui,
            )

            self.assertEqual(
                executors.list_cases(selection, "*Passes"),
                ["PositiveSuite.Passes"],
            )
            self.assertEqual(fake_builder.setup_calls, 1)
            self.assertEqual(fake_builder.build_calls, 1)
            self.assertIn(
                "GTest cases for sm120/integration:", diagnostics.getvalue()
            )


if __name__ == "__main__":
    unittest.main()
