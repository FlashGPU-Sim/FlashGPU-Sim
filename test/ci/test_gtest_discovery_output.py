#!/usr/bin/env python3
"""Regression test for gtest discovery with CI XML output enabled."""

from __future__ import annotations

import subprocess
import tempfile
import textwrap
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


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
  printf 'PositiveSuite.\n  Passes\n'
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


class GTestDiscoveryOutputTest(unittest.TestCase):
    def test_discovery_suppresses_xml_and_execution_writes_one_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            xml_dir = root / "xml"
            fake_gtest = root / "fake_gtest"
            config_dir.mkdir()
            xml_dir.mkdir()
            fake_gtest.write_text(FAKE_GTEST, encoding="utf-8")
            fake_gtest.chmod(0o755)

            command = textwrap.dedent(
                r"""
                source "$1/test/run_tests.sh"
                TEST_TIMEOUT=10
                export GTEST_OUTPUT="xml:$2/"
                gtest_binary_matches_filter "$3" "$4" 'PositiveSuite.*'
                shopt -s nullglob
                discovery_xml=("$2"/*.xml)
                [[ "${#discovery_xml[@]}" -eq 0 ]]
                resolved_filter="$(gtest_binary_intersection_filter \
                  "$3" "$4" 'PositiveSuite.*' 'Passes')"
                [[ "$resolved_filter" == 'PositiveSuite.Passes' ]]
                discovery_xml=("$2"/*.xml)
                [[ "${#discovery_xml[@]}" -eq 0 ]]
                run_binary_with_filter "$3" "$4" "$resolved_filter"
                """
            )
            subprocess.run(
                [
                    "bash",
                    "-c",
                    command,
                    "gtest-discovery-regression",
                    str(REPO_ROOT),
                    str(xml_dir),
                    str(fake_gtest),
                    str(config_dir),
                ],
                cwd=REPO_ROOT,
                check=True,
            )

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


if __name__ == "__main__":
    unittest.main()
