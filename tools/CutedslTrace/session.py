"""Captured module, storage and invocation records for sequential replay."""
import math
import torch
from .artifacts import export_module
from .tensors import OfflineTensor, dtype_info


class Session:
    def __init__(self, output, target, sm_count):
        self.output, self.target, self.sm_count = output, target, sm_count
        self.active = True
        self.modules = []
        self.compiled = []
        self.storages = []
        self.storage_refs = []
        self.calls = []
        self.checks = []
        self.versions = {}
        self.compile_views = []
        (output / "data").mkdir()

    def tensor(self, value):
        if not isinstance(value, OfflineTensor):
            raise TypeError("Runtime tensor must be allocated as CUDA while the tracker is enabled")
        storage = value.untyped_storage()
        index = next((i for i, ref in enumerate(self.storage_refs) if ref._cdata == storage._cdata), None)
        if index is None:
            index = len(self.storages)
            self.storage_refs.append(storage)
            raw = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))
            path = f"data/storage{index}.bin"
            (self.output / path).write_bytes(raw.numpy().tobytes())
            self.storages.append(dict(path=path, bytes=storage.nbytes()))
            self.versions[index] = value._version
        elif self.versions[index] != value._version:
            raise RuntimeError("CPU mutation between captured calls is not supported; prepare inputs before capture")
        code, bits = dtype_info(value.dtype)
        return dict(kind="tensor", storage=index, shape=list(value.shape),
                    strides=list(value.stride()), offset=value.storage_offset() * value.element_size(),
                    code=code, bits=bits)

    def argument(self, value):
        if isinstance(value, torch.Tensor):
            return self.tensor(value)
        if value is None:
            return dict(kind="none")
        if isinstance(value, (bool, int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Nonfinite scalar arguments are not supported")
            if isinstance(value, int) and not -(2**63) <= value < 2**63:
                raise ValueError("Integer argument exceeds int64")
            return dict(kind=type(value).__name__, value=value)
        raise TypeError(f"Unsupported runtime argument: {type(value).__name__}")

    def record(self, compiled, args, options):
        arguments = [self.argument(value) for value in args]
        index = next((i for i, fn in enumerate(self.compiled) if fn is compiled), None)
        if index is None:
            index = len(self.modules)
            name = f"module{index}"
            artifacts = export_module(compiled, self.output / "modules", name, self.target)
            self.modules.append(dict(name=name, artifacts=artifacts, compile_options=options))
            self.compiled.append(compiled)
        self.calls.append(dict(module=index, arguments=arguments))

    def check(self, tensor, expected):
        description = self.tensor(tensor)
        # Optional exact check. Numeric-reference policy remains in the workload.
        if isinstance(expected, torch.Tensor):
            if isinstance(expected, OfflineTensor):
                raise ValueError("Expected values must come from CPU data, not an unexecuted CUDA result")
            value = expected
            if value.device.type != "cpu" or tuple(value.shape) != tuple(tensor.shape) or value.dtype != tensor.dtype:
                raise ValueError("Expected tensor must be CPU data with the same shape and dtype")
        else:
            value = torch.full(tuple(tensor.shape), expected, dtype=tensor.dtype, device="cpu")
        path = f"data/expected{len(self.checks)}.bin"
        (self.output / path).write_bytes(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        self.checks.append(dict(tensor=description, expected=path))
