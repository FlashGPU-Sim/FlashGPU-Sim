"""Resume decisions must never operate on old supervisor/worker PIDs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/run_fa3_large_forward_queue.py'
SPEC = importlib.util.spec_from_file_location('large_queue', SCRIPT)
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


class LargeQueueResumeTest(unittest.TestCase):
    def test_new_case_is_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'case.json'
            self.assertFalse(QUEUE.skip_completed_case(path, False))
            self.assertFalse(QUEUE.skip_completed_case(path, True))
            source = Path(directory) / 'source.so'
            source.write_bytes(b'original library')
            frozen, digest = QUEUE.freeze_library(source, Path(directory), False)
            self.assertEqual((frozen / 'libcudart.so.13').read_bytes(), source.read_bytes())
            self.assertEqual((frozen / 'libcuda.so.1').resolve(), (frozen / 'libcudart.so.13').resolve())
            source.write_bytes(b'new build')
            self.assertEqual((frozen / 'libcudart.so.13').read_bytes(), b'original library')
            with self.assertRaisesRegex(RuntimeError, 'library mismatch'):
                QUEUE.freeze_library(source, Path(directory), True)

    def test_completed_case_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'case.json'
            path.write_text(json.dumps(dict(state='done', rc=0)))
            before = path.read_bytes()
            self.assertTrue(QUEUE.skip_completed_case(path, True))
            self.assertEqual(path.read_bytes(), before)
            source = Path(directory) / 'source.so'
            source.write_bytes(b'original library')
            result = QUEUE.freeze_library(source, Path(directory), False)
            self.assertEqual(QUEUE.freeze_library(source, Path(directory), True), result)
            self.assertEqual(path.read_bytes(), before)

    def test_running_failed_and_inconsistent_cases_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'case.json'
            for state, rc in [('running', None), ('failed', 1), ('done', 1)]:
                with self.subTest(state=state, rc=rc):
                    path.write_text(json.dumps(dict(state=state, rc=rc)))
                    before = path.read_bytes()
                    with self.assertRaises(RuntimeError):
                        QUEUE.skip_completed_case(path, True)
                    self.assertEqual(path.read_bytes(), before)
            source = Path(directory) / 'source.so'
            source.write_bytes(b'original library')
            with self.assertRaisesRegex(RuntimeError, 'no frozen simulator library'):
                QUEUE.freeze_library(source, Path(directory), True)
            frozen, _ = QUEUE.freeze_library(source, Path(directory), False)
            snapshot = frozen / 'libcudart.so.13'
            snapshot.chmod(0o644)
            snapshot.write_bytes(b'corrupted snapshot')
            with self.assertRaisesRegex(RuntimeError, 'library mismatch'):
                QUEUE.freeze_library(source, Path(directory), True)


if __name__ == '__main__':
    unittest.main()
