"""Offline tracker tests. Requires the CuTe DSL Python environment, no GPU."""
import gc
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import cutlass.cute as cute
from CutedslTrace import Tracker
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests/dsl/cutedsl/examples"))
from example_vector_add import add


class TrackerTest(unittest.TestCase):
    def test_cached_calls_aliases_and_optional_checks(self):
        for checked in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                tracker = Tracker(temp, target="sm_100a", sm_count=148)
                tracker.enable()
                try:
                    a = torch.arange(128, device="cuda", dtype=torch.float32)
                    b = torch.ones_like(a)
                    output = torch.empty_like(a)
                    args = [cute.runtime.from_dlpack(t, enable_tvm_ffi=True) for t in (a, b, output)]
                    compiled = cute.compile(add, *args, cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                                            options="--enable-tvm-ffi")
                    compiled(a, b, output)
                    tracker.disable()
                    self.assertFalse(tracker.session.active)
                    self.assertFalse((Path(temp) / "manifest.json").exists())
                    with self.assertRaisesRegex(RuntimeError, "closed"):
                        compiled(output, b, output)
                    tracker.enable()
                    compiled(output, b, output)
                    if checked:
                        tracker.check(output, expected=torch.arange(128, dtype=torch.float32) + 2)
                        tracker.disable()  # Export also works after explicit stop.
                    work = tracker.session.output
                    tracker.export()
                    self.assertFalse(work.exists())
                finally:
                    tracker.disable()
                manifest = json.loads((Path(temp) / "manifest.json").read_text())
                self.assertEqual(len(manifest["checks"]), int(checked))
                self.assertEqual(len(manifest["modules"]), 1)
                self.assertEqual(len(manifest["calls"]), 2)
                self.assertEqual(len(manifest["storages"]), 3)
                args = manifest["calls"][1]["arguments"]
                self.assertEqual(args[0]["storage"], args[2]["storage"])
                self.assertFalse((Path(temp) / "gpgpusim.config").exists())
                self.assertTrue((Path(temp) / "harness.cc").exists())
                if checked:
                    # A normal Toolkit build must not require simulator-only symbols.
                    subprocess.run(["make", "-C", temp], check=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    dynamic = subprocess.check_output(
                        ["readelf", "-d", str(Path(temp) / "replay")], text=True)
                    self.assertRegex(dynamic, r"Shared library: \[libcudart\.so\.[0-9]+\]")
                    self.assertNotIn("BIND_NOW", dynamic)
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    compiled(a, b, output)

    def test_failure_restores_hooks_and_preserves_existing_export(self):
        original = cute.compile
        arch = os.environ.get("CUTE_DSL_ARCH")
        with tempfile.TemporaryDirectory() as temp:
            sentinel = Path(temp) / "manifest.json"
            sentinel.write_text("previous export")
            with self.assertRaisesRegex(RuntimeError, "host control flow"):
                tracker = Tracker(temp, target="sm_100a", sm_count=148)
                tracker.enable()
                try:
                    bool(torch.ones((), device="cuda"))
                finally:
                    tracker.disable()
            self.assertIs(cute.compile, original)
            self.assertEqual(os.environ.get("CUTE_DSL_ARCH"), arch)
            self.assertEqual(sentinel.read_text(), "previous export")
            work = tracker.session.output
            del tracker
            gc.collect()
            self.assertFalse(work.exists())

    def test_lifecycle_rejects_overlap_and_empty_export(self):
        original = cute.compile
        from cutlass.cutlass_dsl import BaseDSL
        from cutlass.base_dsl.arch import Arch
        original_arch_query = BaseDSL.get_arch_enum
        with tempfile.TemporaryDirectory() as temp:
            first = Tracker(temp, target="sm_100a", sm_count=148)
            second = Tracker(temp, target="sm_100a", sm_count=148)
            first.enable()
            try:
                first.enable()  # Idempotent; does not stack hooks.
                self.assertEqual(BaseDSL._get_dsl().get_arch_enum(), Arch.sm_100a)
                with self.assertRaisesRegex(RuntimeError, "Concurrent"):
                    second.enable()
                second.disable()  # Must not disturb the active tracker.
                self.assertIsNot(cute.compile, original)
                with self.assertRaisesRegex(RuntimeError, "No CuTe DSL calls"):
                    first.export()
                self.assertIs(cute.compile, original)
            finally:
                first.disable()
            self.assertIs(BaseDSL.get_arch_enum, original_arch_query)
            second.enable()
            second.disable()
            self.assertIs(cute.compile, original)

    def test_storage_views_and_reject_host_mutation(self):
        from CutedslTrace.session import Session
        from CutedslTrace.tensors import OfflineTensor
        with tempfile.TemporaryDirectory() as temp:
            session = Session(Path(temp), "sm_100a", 148)
            tensor = torch.arange(16, dtype=torch.float32).as_subclass(OfflineTensor)
            first = session.tensor(tensor[2:10:2])
            second = session.tensor(tensor[4:12:2])
            self.assertEqual(first["storage"], second["storage"])
            self.assertEqual(first["offset"], 8)
            self.assertEqual(second["offset"], 16)
            self.assertEqual(first["strides"], [2])
            tensor.add_(1)
            with self.assertRaisesRegex(RuntimeError, "CPU mutation"):
                session.tensor(tensor)


if __name__ == "__main__":
    unittest.main()
