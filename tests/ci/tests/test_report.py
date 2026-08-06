#!/usr/bin/env python3
"""Tests for compact CI failure reporting."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


CI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CI_DIR))

from report import render_failure_report  # noqa: E402


FAILED_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<testsuites tests="3" failures="2" errors="0">
  <testsuite name="ExampleSuite" tests="3" failures="2" errors="0">
    <testcase name="Passes" classname="ExampleSuite" time="0.1" />
    <testcase name="FirstFailure" classname="ExampleSuite" time="0.2"
              file="tests/example_test.cu" line="12">
      <failure>tests/example_test.cu:13
Expected cudaSuccess but got cudaErrorInvalidValue</failure>
    </testcase>
    <testcase name="SecondFailure" classname="ExampleSuite" time="0.3">
      <failure>first assertion</failure>
      <failure>second assertion</failure>
    </testcase>
  </testsuite>
</testsuites>
"""


class FailureReportTest(unittest.TestCase):
    def test_report_lists_only_failed_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            xml_dir = root / "xml" / "sm90-core-integration-smoke"
            xml_dir.mkdir(parents=True)
            (xml_dir / "integration_tests.xml").write_text(
                FAILED_XML, encoding="utf-8"
            )
            (root / "failure-context.txt").write_text(
                "phase=sm90-core-integration-smoke\n"
                "exit_code=1\n"
                "group=integration\n",
                encoding="utf-8",
            )

            report = render_failure_report(root, "sm90-core")

            self.assertIn("| integration |", report)
            self.assertIn("ExampleSuite.FirstFailure", report)
            self.assertIn("ExampleSuite.SecondFailure", report)
            self.assertNotIn("ExampleSuite.Passes", report)
            self.assertIn(
                "| integration | `ExampleSuite.FirstFailure` | Failed |", report
            )
            self.assertIn("tests/example_test.cu:13", report)
            self.assertNotIn("tests/example_test.cu:12", report)

    def test_report_recovers_case_aborted_before_xml_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_dir = root / "logs"
            log_dir.mkdir()
            (root / "failure-context.txt").write_text(
                "phase=sm120-core-mbarrier-sanity\n"
                "exit_code=250\n"
                "group=barrier\n",
                encoding="utf-8",
            )
            (log_dir / "sm120-core-mbarrier-sanity.log").write_text(
                "\x1b[0;32m[ RUN      ] \x1b[mEarlierTest.Passes\n"
                "\x1b[0;32m[       OK ] \x1b[mEarlierTest.Passes (1 ms)\n"
                "\x1b[0;32m[ RUN      ] \x1b[mMBarrierSanityTest.Phase\n"
                "GPGPU-Sim uArch: ERROR ** deadlock detected\n"
                "/usr/bin/timeout: the monitored command dumped core\n",
                encoding="utf-8",
            )

            report = render_failure_report(root, "sm120-core")

            self.assertIn(
                "| barrier | `MBarrierSanityTest.Phase` | Aborted |", report
            )
            self.assertNotIn("EarlierTest.Passes", report)
            self.assertIn("deadlock detected", report)
            self.assertIn(
                "<summary><code>MBarrierSanityTest.Phase</code></summary>",
                report,
            )
            self.assertNotIn("No failed GoogleTest case", report)
            self.assertNotIn("## `sm120-core` failures", report)
            self.assertNotIn("Failed during `sm120-core-mbarrier-sanity`", report)

    def test_completed_parameterized_case_is_not_reported_as_aborted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_dir = root / "logs"
            log_dir.mkdir()
            (root / "failure-context.txt").write_text(
                "phase=sm120-core-integration\n"
                "exit_code=1\n"
                "group=integration\n",
                encoding="utf-8",
            )
            (log_dir / "sm120-core-integration.log").write_text(
                "[ RUN      ] Values/ExampleTest.Case/0\n"
                "tests/example_test.cu:12: Failure\n"
                "[  FAILED  ] Values/ExampleTest.Case/0, where GetParam() = 42\n",
                encoding="utf-8",
            )

            report = render_failure_report(root, "sm120-core")

            self.assertNotIn("Aborted", report)
            self.assertIn("No failed GoogleTest case", report)

    def test_report_handles_failure_without_xml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "failure-context.txt").write_text(
                "phase=build-flashgpu-sim\nexit_code=137\n", encoding="utf-8"
            )

            report = render_failure_report(root, "sm90-fa2")

            self.assertIn("Failed during `build-flashgpu-sim`", report)
            self.assertIn("exit code `137`", report)
            self.assertIn("No failed GoogleTest case", report)

    def test_report_handles_incomplete_xml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            xml_dir = root / "xml" / "sm90-fa3-smoke"
            xml_dir.mkdir(parents=True)
            (xml_dir / "fa3_tests.xml").write_text(
                "<testsuites>", encoding="utf-8"
            )

            report = render_failure_report(root, "sm90-fa3")

            self.assertIn("Some XML files could not be parsed", report)
            self.assertIn("fa3_tests.xml", report)


if __name__ == "__main__":
    unittest.main()
