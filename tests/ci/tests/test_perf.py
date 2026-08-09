#!/usr/bin/env python3
"""Tests for the informational performance CI framework."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


CI_DIR = Path(__file__).resolve().parents[1]
PERF_DIR = CI_DIR / "perf"
sys.path.insert(0, str(PERF_DIR))

from perf import (  # noqa: E402
    MANIFEST_FILE,
    PerfCase,
    PerfError,
    PerfJob,
    load_manifest,
    matrix_json,
    render_report,
    run_job,
)


class PerfFrameworkTest(unittest.TestCase):
    def make_repo(self, root: Path, *configs: str) -> None:
        for config in configs:
            config_dir = root / "configs" / config
            config_dir.mkdir(parents=True)
            (config_dir / "gpgpusim.config").write_text("# test\n", encoding="utf-8")
        scripts = root / "scripts"
        scripts.mkdir()
        (scripts / "pass.sh").write_text(
            "#!/usr/bin/env bash\necho 'gpu_tot_sim_cycle = 1234'\n",
            encoding="utf-8",
        )

    def write_manifest(self, root: Path, text: str) -> Path:
        path = root / "cases.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_repository_manifest_and_matrix_are_valid(self) -> None:
        jobs = load_manifest(MANIFEST_FILE)

        self.assertEqual([job.config for job in jobs], ["SM120_RTX5090"])
        self.assertEqual(len(jobs[0].cases), 3)
        self.assertEqual(
            json.loads(matrix_json(jobs)),
            {"config": ["SM120_RTX5090"]},
        )

    def test_multiple_configs_keep_order_and_may_repeat_case_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_repo(root, "SM120_TEST", "SM90_TEST")
            manifest = self.write_manifest(
                root,
                """
[SM120_TEST.cases.same-case]
label = "First"
scripts = ["scripts/pass.sh"]
ncu_cycles = 10.5

[SM90_TEST.cases.same-case]
label = "Second"
scripts = ["scripts/pass.sh"]
ncu_cycles = 20
""",
            )

            jobs = load_manifest(manifest, root)

            self.assertEqual([job.config for job in jobs], ["SM120_TEST", "SM90_TEST"])
            self.assertEqual(jobs[0].cases[0].id, jobs[1].cases[0].id)

    def test_unknown_fields_and_invalid_cycles_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_repo(root, "SM120_TEST")
            unknown = self.write_manifest(
                root,
                """
[SM120_TEST]
extra = true
[SM120_TEST.cases.example]
label = "Example"
scripts = ["scripts/pass.sh"]
ncu_cycles = 10
""",
            )
            with self.assertRaisesRegex(PerfError, "extra: extra"):
                load_manifest(unknown, root)

            invalid_cycles = self.write_manifest(
                root,
                """
[SM120_TEST.cases.example]
label = "Example"
scripts = ["scripts/pass.sh"]
ncu_cycles = 0
""",
            )
            with self.assertRaisesRegex(PerfError, "finite positive number"):
                load_manifest(invalid_cycles, root)

    def test_script_paths_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.make_repo(root, "SM120_TEST")
            manifest = self.write_manifest(
                root,
                """
[SM120_TEST.cases.example]
label = "Example"
scripts = ["../outside.sh"]
ncu_cycles = 10
""",
            )
            with self.assertRaisesRegex(PerfError, "repository-relative"):
                load_manifest(manifest, root)

    def test_runner_continues_after_a_failed_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scripts = root / "scripts"
            scripts.mkdir()
            (scripts / "pass.sh").write_text(
                "echo 'gpu_tot_sim_cycle = 1234'\n", encoding="utf-8"
            )
            (scripts / "fail.sh").write_text("exit 7\n", encoding="utf-8")
            job = PerfJob(
                config="SM120_TEST",
                cases=(
                    PerfCase("fails", "Failure", ("scripts/fail.sh",), 100.0),
                    PerfCase("passes", "Success", ("scripts/pass.sh",), 1200.0),
                ),
            )
            log_root = root / "logs"

            status = run_job(job, root, log_root)
            payload = json.loads(
                (log_root / "results" / "SM120_TEST.json").read_text(encoding="utf-8")
            )

            self.assertEqual(status, 1)
            self.assertEqual(payload["cases"][0]["status"], "failed")
            self.assertEqual(payload["cases"][0]["exit_code"], 7)
            self.assertEqual(payload["cases"][1]["status"], "compared")
            self.assertEqual(payload["cases"][1]["sim_cycles"], 1234)

    def test_report_compares_cycles_and_validates_readme(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_dir = root / "results"
            results_dir.mkdir()
            job = PerfJob(
                config="SM120_TEST",
                cases=(
                    PerfCase(
                        "example", "Tutorial - Example", ("scripts/pass.sh",), 100.0
                    ),
                ),
            )
            (results_dir / "SM120_TEST.json").write_text(
                json.dumps(
                    {
                        "config": "SM120_TEST",
                        "cases": [
                            {
                                "id": "example",
                                "status": "compared",
                                "sim_cycles": 105,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            readme = root / "README.md"
            readme.write_text(
                """
| Config | Workload | Shape | NCU cycles | Sim cycles | Difference |
| --- | --- | --- | ---: | ---: | ---: |
| SM120_TEST | Tutorial - Example | any | 100.00 | 105 | +5.00% |
""",
                encoding="utf-8",
            )

            report, success = render_report((job,), results_dir, readme)

            self.assertTrue(success)
            self.assertIn("| Compared | Tutorial - Example | 100.00 | 105 | +5.00", report)
            self.assertIn("README Cycle Validation: matches", report)

    def test_report_marks_missing_results_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job = PerfJob(
                config="SM120_TEST",
                cases=(PerfCase("example", "Example", ("script.sh",), 100.0),),
            )

            report, success = render_report((job,), root / "missing", None)

            self.assertFalse(success)
            self.assertIn("| Missing result | Example |", report)

    def test_report_rejects_a_missing_readme_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_dir = root / "results"
            results_dir.mkdir()
            job = PerfJob(
                config="SM120_TEST",
                cases=(PerfCase("example", "Example", ("script.sh",), 100.0),),
            )
            (results_dir / "SM120_TEST.json").write_text(
                json.dumps(
                    {
                        "config": "SM120_TEST",
                        "cases": [
                            {
                                "id": "example",
                                "status": "compared",
                                "sim_cycles": 100,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            readme = root / "README.md"
            readme.write_text(
                """
| Config | Workload | Shape | NCU cycles | Sim cycles | Difference |
| --- | --- | --- | ---: | ---: | ---: |
""",
                encoding="utf-8",
            )

            report, success = render_report((job,), results_dir, readme)

            self.assertFalse(success)
            self.assertIn("README row is missing for SM120_TEST/Example", report)


if __name__ == "__main__":
    unittest.main()
