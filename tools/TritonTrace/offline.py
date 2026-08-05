from __future__ import annotations

import re
from typing import Any, Dict

import triton

from .harness import HarnessGenerator
from .session import TrackingSession


class OfflineTracker:
    """Compile intercepted Triton launches for an explicit CUDA target."""

    def __init__(
        self,
        tracker,
        session: TrackingSession,
        harness: HarnessGenerator,
        target,
    ):
        self.tracker = tracker
        self.session = session
        self.harness = harness
        self.target = target
        self.target_name = f"sm{target.arch}"
        self.binders: Dict[tuple, Any] = {}

    @staticmethod
    def parse_target(target):
        from triton.backends.compiler import GPUTarget

        if target is None:
            raise ValueError(
                "mode='offline' requires an explicit CUDA target such as 'sm90'"
            )
        if not isinstance(target, str):
            raise TypeError("offline target must be a string such as 'sm90'")

        match = re.fullmatch(r"sm_?(\d+)(?:a)?", target.lower())
        if match is None:
            raise ValueError(
                f"Invalid CUDA target '{target}'; expected a value such as 'sm90'"
            )
        return GPUTarget("cuda", int(match.group(1)), 32)

    def install(self) -> None:
        self._patch_jit_run()
        self._patch_autotuner_run()
        print("[TritonTracker] Offline compilation interception installed")

    def _patch_jit_run(self) -> None:
        from triton.runtime.jit import JITFunction

        offline_tracker = self

        def patched_run(jit_function, *args, **kwargs):
            if not offline_tracker.tracker.enabled:
                return None
            return offline_tracker.compile(jit_function, args, kwargs)

        JITFunction.run = patched_run
        print("[TritonTracker] Patched JITFunction.run")

    def _patch_autotuner_run(self) -> None:
        from triton.runtime.autotuner import Autotuner

        offline_tracker = self

        def patched_run(autotuner, *args, **kwargs):
            if not offline_tracker.tracker.enabled:
                return None
            if len(autotuner.configs) > 1:
                raise RuntimeError(
                    "Offline tracking cannot benchmark multiple @triton.autotune "
                    "configurations. Select one fixed triton.Config before invoking "
                    "the kernel."
                )
            config = autotuner.configs[0]
            autotuner.best_config = config
            return autotuner.fn.run(*args, **kwargs, **config.all_kwargs())

        Autotuner.run = patched_run
        print("[TritonTracker] Patched Autotuner.run for offline configuration checks")

    def compile(self, jit_function, args: tuple, kwargs: Dict[str, Any]):
        from triton.compiler import ASTSource, make_backend
        from triton.runtime.jit import create_function_from_signature

        compile_kwargs = dict(kwargs)
        grid = compile_kwargs.pop("grid", None)
        warmup = compile_kwargs.pop("warmup", False)
        if grid is None:
            raise ValueError("Offline tracking requires a grid for each kernel invocation")

        compile_kwargs["debug"] = (
            compile_kwargs.get("debug", jit_function.debug)
            or triton.knobs.runtime.debug
        )

        backend = make_backend(self.target)
        binder_key = (id(jit_function), self.target_name)
        binder_entry = self.binders.get(binder_key)
        if binder_entry is None:
            binder = create_function_from_signature(
                jit_function.signature, jit_function.params, backend
            )
            self.binders[binder_key] = (jit_function, binder)
        else:
            _, binder = binder_entry

        bound_args, specialization, binder_options = binder(*args, **compile_kwargs)
        options, signature, constexprs, attributes = jit_function._pack_args(
            backend, compile_kwargs, bound_args, specialization, binder_options
        )
        source = ASTSource(jit_function, signature, constexprs, attributes)
        compiled = triton.compile(
            source,
            target=self.target,
            options=options.__dict__,
        )

        kernel, created = self.session.record_kernel(
            compiled.name,
            compiled.metadata_group,
            compiled.hash,
            event="compiled offline",
        )
        if kernel is None:
            raise RuntimeError(
                f"Offline compilation produced no binary for '{compiled.name}'"
            )
        if created and self.session.save_binaries:
            self.harness.generate_kernel_template(kernel)

        if warmup:
            return compiled

        if callable(grid):
            grid = grid(bound_args)
        runtime_args = tuple(
            bound_args[name]
            for name, argument_type in signature.items()
            if argument_type != "constexpr"
        )
        launch, _ = self.session.record_launch(
            kernel,
            grid,
            runtime_args,
            snapshot_tensors=False,
        )
        if self.session.save_binaries and self.session.capture_args:
            unsupported_arguments = [
                argument.index
                for argument in launch.args_info
                if argument.arg_type not in {"tensor", "scalar"}
                or (argument.arg_type == "tensor" and not argument.data_file)
            ]
            if unsupported_arguments:
                print(
                    "  [WARNING] Launch-specific harness skipped because arguments "
                    f"{unsupported_arguments} cannot be serialized"
                )
            else:
                self.harness.generate_launch_harness(
                    kernel, launch, validate_outputs=False
                )

        return compiled
