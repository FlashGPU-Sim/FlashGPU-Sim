"""Scoped hooks on CuTe DSL compilation; no workload imports or selectors."""
from contextlib import ExitStack
import shlex


class CapturedFunction:
    def __init__(self, session, compiled, options):
        self.session, self.compiled, self.options = session, compiled, options

    def __call__(self, *args, **kwargs):
        if not self.session.active:
            raise RuntimeError("This cached function belongs to a closed offline export session")
        if kwargs:
            raise TypeError("Offline replay currently requires positional runtime arguments")
        self.session.record(self.compiled, args, self.options)


class CompilerHooks:
    def __init__(self, session):
        self.session = session
        self.stack = ExitStack()
        self.views = session.compile_views

    def __enter__(self):
        import torch
        import cutlass.cute as cute
        import cutlass.utils
        from cutlass.cutlass_dsl import BaseDSL
        from cutlass.base_dsl.arch import Arch
        from cutlass.cutlass_dsl.tvm_ffi_provider import TVMFFIJitCompiledFunctionBase
        from .tensors import CompileTensor, OfflineTensor, TensorMode, target_properties
        original_compile = cute.compile
        original_tensor_init = cute.runtime._Tensor.__init__
        original_create = TVMFFIJitCompiledFunctionBase._create_tvm_ffi_function
        session = self.session

        def compile_function(*args, **kwargs):
            kwargs["no_jit_engine"] = True
            # Preserve computation/ABI options; explicitly request target and artifacts.
            options = kwargs.get("options", "")
            if not isinstance(options, str):
                raise TypeError("Offline export currently expects string compile options")
            tokens = shlex.split(options)
            if any(token.startswith("--gpu-arch") for token in tokens):
                raise ValueError("Set the target on Tracker, not a second --gpu-arch option")
            kwargs["options"] = options + f" --gpu-arch={session.target} --keep-ptx --keep-cubin --dump-dir={session.output / 'dump'}"
            # Preserve the workload's compiler options, including TVM FFI.
            compiled = original_compile(*args, **kwargs)
            if not isinstance(compiled, TVMFFIJitCompiledFunctionBase):
                raise TypeError("Initial offline replay supports CuTe DSL's TVM FFI ABI; compile with --enable-tvm-ffi")
            return CapturedFunction(session, compiled, options)

        def tensor_init(instance, tensor, *args, **kwargs):
            if isinstance(tensor, OfflineTensor):
                view = CompileTensor(tensor)
                self.views.append(view)
                tensor = view
            original_tensor_init(instance, tensor, *args, **kwargs)

        def disabled_call(*args, **kwargs):
            raise RuntimeError("Attempted to execute a JIT function during offline export")

        def create_function(instance):
            # CuTe DSL 4.5.2's kwargs wrapper requires a callable even without an engine.
            return disabled_call if instance.engine is None else original_create(instance)

        class HardwareInfo:
            def get_device_multiprocessor_count(self):
                return session.sm_count

        properties = target_properties(session.target, session.sm_count)
        target_arch = Arch.from_string(session.target)
        try:
            for owner, name, value in (
                (cute, "compile", compile_function),
                # The DSL instance may cache the host architecture before enable().
                (BaseDSL, "get_arch_enum", lambda self: target_arch),
                (cute.runtime._Tensor, "__init__", tensor_init),
                (TVMFFIJitCompiledFunctionBase, "_create_tvm_ffi_function", create_function),
                (cutlass.utils, "HardwareInfo", HardwareInfo),
                (torch.cuda, "get_device_properties", lambda *a: properties),
                (torch.cuda, "get_device_capability", lambda *a: (properties.major, properties.minor)),
            ):
                previous = getattr(owner, name)
                setattr(owner, name, value)
                self.stack.callback(setattr, owner, name, previous)
            self.stack.enter_context(TensorMode(session))
            self.stack.enter_context(torch.no_grad())
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *exc):
        self.stack.close()
