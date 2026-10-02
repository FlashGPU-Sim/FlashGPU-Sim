"""CPU-backed CUDA tensor descriptions for offline Python execution."""
import ctypes as C
from types import SimpleNamespace
import torch
from torch.overrides import TorchFunctionMode


class OfflineTensor(torch.Tensor):
    @property
    def device(self):
        return torch.device("cuda:0")

    @property
    def is_cuda(self):
        return True

    def item(self):
        raise RuntimeError("Offline CUDA tensors cannot drive host scalar computations")

    def __bool__(self):
        raise RuntimeError("Offline CUDA tensors cannot drive host control flow")


class TensorMode(TorchFunctionMode):
    def __init__(self, session):
        super().__init__()
        self.session = session

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        def tensors(value):
            if isinstance(value, OfflineTensor):
                yield value
            elif isinstance(value, (list, tuple)):
                for child in value:
                    yield from tensors(child)
            elif isinstance(value, dict):
                for child in value.values():
                    yield from tensors(child)
        metadata = {"__get__", "size", "stride", "dim", "numel", "nelement", "element_size",
                    "data_ptr", "untyped_storage", "storage_offset", "detach", "view",
                    "transpose", "permute", "t", "is_contiguous", "empty_like", "zeros_like", "ones_like"}
        if getattr(func, "__name__", "") not in metadata:
            for tensor in tensors((args, kwargs)):
                if any(ref._cdata == tensor.untyped_storage()._cdata for ref in self.session.storage_refs):
                    raise RuntimeError("Host computation on a captured CUDA buffer is unsupported offline")
        device = kwargs.get("device")
        redirect = device is not None and str(device).startswith("cuda")
        if redirect:
            if torch.device(device).index not in (None, 0):
                raise ValueError("Offline export currently supports device 0 only")
            kwargs["device"] = "cpu"
        result = func(*args, **kwargs)
        if redirect and isinstance(result, torch.Tensor):
            result = result.as_subclass(OfflineTensor)
        # Empty allocations have no specified contents; initialize deterministically.
        if func in (torch.empty, torch.empty_like, torch.empty_strided) and isinstance(result, OfflineTensor):
            result.zero_()
        return result


class Device(C.Structure):
    _fields_ = [("type", C.c_int), ("id", C.c_int)]


class Dtype(C.Structure):
    _fields_ = [("code", C.c_uint8), ("bits", C.c_uint8), ("lanes", C.c_uint16)]


class DLTensor(C.Structure):
    _fields_ = [("data", C.c_void_p), ("device", Device), ("ndim", C.c_int),
                ("dtype", Dtype), ("shape", C.POINTER(C.c_int64)),
                ("strides", C.POINTER(C.c_int64)), ("offset", C.c_uint64)]


class ManagedTensor(C.Structure):
    _fields_ = [("tensor", DLTensor), ("context", C.c_void_p), ("deleter", C.c_void_p)]


def dtype_info(dtype):
    codes = {torch.float16: 2, torch.float32: 2, torch.float64: 2,
             torch.bfloat16: 4, torch.int8: 0, torch.int16: 0,
             torch.int32: 0, torch.int64: 0, torch.uint8: 1, torch.bool: 6}
    if dtype not in codes:
        raise TypeError(f"Unsupported export dtype: {dtype}")
    return codes[dtype], torch.empty((), dtype=dtype).element_size() * 8


class CompileTensor:
    """A session-owned DLPack descriptor, never used to execute CUDA code.

    CuTe's capsule-only input path avoids TVM/PyTorch's real-device queries.
    The session keeps this descriptor and its CPU storage alive through compile.
    """
    def __init__(self, tensor):
        self.tensor = tensor
        self.shape = (C.c_int64 * tensor.ndim)(*tensor.shape)
        self.strides = (C.c_int64 * tensor.ndim)(*tensor.stride())
        code, bits = dtype_info(tensor.dtype)
        self.managed = ManagedTensor(DLTensor(
            tensor.data_ptr(), Device(2, 0), tensor.ndim, Dtype(code, bits, 1),
            self.shape, self.strides, 0), None, None)
        make = C.pythonapi.PyCapsule_New
        make.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p]
        make.restype = C.py_object
        self.capsule = make(C.addressof(self.managed), b"dltensor", None)

    def __dlpack_device__(self):
        return self.capsule


def target_properties(target, sm_count):
    number = int(target.removeprefix("sm_").rstrip("af"))
    return SimpleNamespace(major=number // 10, minor=number % 10,
                           multi_processor_count=sm_count)
