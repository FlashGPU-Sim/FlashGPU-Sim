from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from TritonTrace.session import TrackingSession


class TensorSerializationTest(unittest.TestCase):
    def test_bf16_preserves_bits_and_dtype(self):
        with tempfile.TemporaryDirectory() as path:
            session = TrackingSession(path, save_binaries=True, capture_args=True,
                                      mode="online", target_name=None)
            bits = torch.tensor([0x0000, 0x3f80, 0x4001, 0x7fff, 0x8000], dtype=torch.uint16)
            output, metadata = session._serialize_tensor(bits.view(torch.bfloat16), 1, 0, "kernel")
            self.assertEqual(Path(output).read_bytes(), bits.numpy().tobytes())
            self.assertEqual(metadata["dtype"], "bfloat16")
            self.assertEqual(metadata["shape"], [5])
            self.assertEqual(metadata["size_bytes"], 10)

    def test_bf16_changed_output_preserves_bits(self):
        with tempfile.TemporaryDirectory() as path:
            session = TrackingSession(path, save_binaries=True, capture_args=True,
                                      mode="online", target_name=None)
            info = SimpleNamespace(arg_type="tensor", output_file=None)
            launch = SimpleNamespace(kernel_name="kernel", launch_id=1, args_info=[info])
            before = torch.zeros(2, dtype=torch.bfloat16)
            after = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
            with patch("torch.cuda.synchronize"):
                session.capture_outputs(launch, (after,), [before])
            self.assertEqual(Path(info.output_file).read_bytes(), after.view(torch.uint16).numpy().tobytes())
    def test_fp16_metadata_and_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as path:
            session = TrackingSession(path, save_binaries=True, capture_args=True,
                                      mode="online", target_name=None)
            tensor = torch.tensor([1.0, -2.0], dtype=torch.float16)
            output, metadata = session._serialize_tensor(tensor, 1, 0, "kernel")
            self.assertEqual(Path(output).read_bytes(), tensor.numpy().tobytes())
            self.assertEqual(metadata["dtype"], "float16")


if __name__ == "__main__":
    unittest.main()
