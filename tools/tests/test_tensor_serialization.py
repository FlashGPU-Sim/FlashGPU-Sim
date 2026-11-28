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
    def test_numpy_unsupported_dtypes_preserve_bits_and_dtype(self):
        dtypes = [torch.bfloat16]
        for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
            dtype = getattr(torch, name, None)
            if dtype is not None:
                dtypes.append(dtype)

        for dtype in dtypes:
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as path:
                session = TrackingSession(path, save_binaries=True, capture_args=True,
                                          mode="online", target_name=None)
                bits = torch.tensor([0, 1, 0x3f, 0x7f], dtype=torch.uint8)
                tensor = bits.view(dtype)
                output, metadata = session._serialize_tensor(tensor, 1, 0, "kernel")
                self.assertEqual(Path(output).read_bytes(), bits.numpy().tobytes())
                self.assertEqual(metadata["dtype"], str(dtype).removeprefix("torch."))
                self.assertEqual(metadata["shape"], [4])
                self.assertEqual(metadata["size_bytes"], 4)

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

    def test_float8_changed_output_preserves_bits(self):
        dtype = getattr(torch, "float8_e4m3fn", None)
        if dtype is None:
            self.skipTest("PyTorch build does not provide float8_e4m3fn")
        with tempfile.TemporaryDirectory() as path:
            session = TrackingSession(path, save_binaries=True, capture_args=True,
                                      mode="online", target_name=None)
            info = SimpleNamespace(arg_type="tensor", output_file=None)
            launch = SimpleNamespace(kernel_name="kernel", launch_id=1, args_info=[info])
            before = torch.tensor([0, 0], dtype=torch.uint8).view(dtype)
            after_bits = torch.tensor([0x38, 0xb8], dtype=torch.uint8)
            after = after_bits.view(dtype)
            with patch("torch.cuda.synchronize"):
                session.capture_outputs(launch, (after,), [before])
            self.assertEqual(Path(info.output_file).read_bytes(), after_bits.numpy().tobytes())
    def test_fp16_metadata_and_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as path:
            session = TrackingSession(path, save_binaries=True, capture_args=True,
                                      mode="online", target_name=None)
            tensor = torch.tensor([1.0, -2.0], dtype=torch.float16)
            output, metadata = session._serialize_tensor(tensor, 1, 0, "kernel")
            self.assertEqual(Path(output).read_bytes(), tensor.numpy().tobytes())
            self.assertEqual(metadata["dtype"], "float16")

    def test_scalar_tensor_preserves_shape_and_bytes(self):
        with tempfile.TemporaryDirectory() as path:
            session = TrackingSession(path, save_binaries=True, capture_args=True,
                                      mode="online", target_name=None)
            tensor = torch.tensor(1.5, dtype=torch.float32)
            output, metadata = session._serialize_tensor(tensor, 1, 0, "kernel")
            self.assertEqual(Path(output).read_bytes(), tensor.view(torch.uint8).numpy().tobytes())
            self.assertEqual(metadata["shape"], [])
            self.assertEqual(metadata["size_bytes"], 4)

    def test_non_strided_tensor_fails_with_clear_error(self):
        with tempfile.TemporaryDirectory() as path:
            session = TrackingSession(path, save_binaries=True, capture_args=True,
                                      mode="online", target_name=None)
            with self.assertRaisesRegex(TypeError, "dense strided tensors"):
                session._serialize_tensor(torch.ones((2, 2)).to_sparse(), 1, 0, "kernel")


if __name__ == "__main__":
    unittest.main()
