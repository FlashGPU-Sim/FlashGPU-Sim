"""Negative controls for the numerical and timing gates, independent of a model revision."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "ci/check_sass_medium.py"
spec = importlib.util.spec_from_file_location("sass_medium_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
CASE = "H32D64FullB16S512"


class SassMediumGateTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.baseline = self.root / "baseline.csv"
        self.baseline.write_text(f"profile,case,sim_cycles\nmedium,{CASE},100000\n")
        self.log = self.root / "run.log"

    def valid(self, cycles=100000):
        return (f"[ RUN      ] Fa3PrefillFp16MediumTest.{CASE}\n"
                "mode=execution-driven SASS timing simulation\n"
                f"gpu_kernel_sim_cycle[1] = {cycles}\n"
                f"FA3 reference PASS: {CASE} O and LSE\n"
                "[  PASSED  ] 1 test.\n")

    def check(self, text):
        self.log.write_text(text)
        with contextlib.redirect_stdout(io.StringIO()):
            gate.check(self.log, baseline=self.baseline)

    def test_accepts_valid_reference_and_timing(self):
        self.check(self.valid())

    def test_accepts_gtest_ansi_colors(self):
        self.check(self.valid().replace("[  PASSED  ]", "\x1b[32m[  PASSED  ]\x1b[0m"))

    def test_inclusive_half_percent_boundary(self):
        for cycles in (99500, 100500):
            with self.subTest(cycles=cycles):
                self.check(self.valid(cycles))

    def test_rejects_one_cycle_outside_either_boundary(self):
        for cycles in (99499, 100501):
            with self.subTest(cycles=cycles), self.assertRaisesRegex(SystemExit, "exceeds 0.5%"):
                self.check(self.valid(cycles))

    def test_requires_reference_for_both_o_and_lse(self):
        with self.assertRaisesRegex(SystemExit, "O/LSE"):
            self.check(self.valid().replace(" O and LSE", " O only"))

    def test_rejects_failed_numerical_test(self):
        with self.assertRaisesRegex(SystemExit, "O/LSE"):
            self.check(self.valid() + "[  FAILED  ] 1 test.\n")

    def test_requires_native_timing(self):
        with self.assertRaisesRegex(SystemExit, "actual SASS timing"):
            self.check(self.valid().replace("execution-driven SASS", "PTX"))

    def test_rejects_extra_kernel(self):
        with self.assertRaisesRegex(SystemExit, "exactly one"):
            self.check(self.valid() + "gpu_kernel_sim_cycle[2] = 1\n")

    def test_rejects_extra_or_unknown_case(self):
        for text in (self.valid() + f"[ RUN      ] Fa3PrefillFp16MediumTest.{CASE}\n",
                     self.valid().replace(CASE, "Unknown")):
            with self.subTest(text=text), self.assertRaisesRegex(SystemExit, "exactly one"):
                self.check(text)


if __name__ == "__main__":
    unittest.main()
