from __future__ import annotations

from typing import Any, Dict, Optional

import triton

from .harness import HarnessGenerator
from .session import TrackingSession


class OnlineTracker:
    """Collect kernels and launches from Triton's CUDA runtime hooks."""

    def __init__(self, tracker, session: TrackingSession, harness: HarnessGenerator):
        self.tracker = tracker
        self.session = session
        self.harness = harness
        self.pending_args: Optional[tuple] = None
        self.pending_grid: Optional[tuple] = None
        self.pending_snapshots = None
        self.function_to_hash: Dict[Any, str] = {}

    def install(self) -> None:
        runtime = triton.knobs.runtime
        runtime.kernel_load_end_hook.add(self._on_kernel_load)
        runtime.launch_enter_hook.add(self._on_launch_enter)
        runtime.launch_exit_hook.add(self._on_launch_exit)
        self._patch_jit_run()
        print("[TritonTracker] Runtime hooks installed")

    def _patch_jit_run(self) -> None:
        from triton.runtime.jit import JITFunction

        original_run = JITFunction.run
        online_tracker = self

        def patched_run(jit_function, *args, **kwargs):
            warmup = kwargs.get("warmup", False)
            if online_tracker.tracker.enabled and not warmup:
                online_tracker.pending_args = (
                    args if online_tracker.session.capture_args else ()
                )
                grid = kwargs.get("grid")
                if grid is not None:
                    if callable(grid):
                        metadata = {
                            key: value
                            for key, value in kwargs.items()
                            if key.isupper() or key == "grid"
                        }
                        metadata.pop("grid", None)
                        try:
                            grid = grid(metadata)
                        except Exception as error:
                            raise RuntimeError(
                                f"Failed to evaluate grid function: {error}\n"
                                f"Grid function: {grid}\n"
                                f"Available meta keys: {list(metadata.keys())}\n"
                                "This may indicate missing constexpr values in kwargs."
                            ) from error
                    online_tracker.pending_grid = TrackingSession.normalize_grid(grid)

            result = original_run(jit_function, *args, **kwargs)
            if warmup or not online_tracker.tracker.enabled:
                online_tracker._clear_pending()
            return result

        JITFunction.run = patched_run
        print("[TritonTracker] Patched JITFunction.run")

    def _on_kernel_load(self, module, function, name, metadata_group, hash_value):
        del module
        kernel, created = self.session.record_kernel(
            name, metadata_group, hash_value, event="loaded"
        )
        if kernel is None:
            return
        self.function_to_hash[function] = hash_value
        if created and self.session.save_binaries:
            self.harness.generate_kernel_template(kernel)

    @staticmethod
    def _metadata_dict(launch_metadata):
        metadata = (
            launch_metadata.get()
            if hasattr(launch_metadata, "get")
            else launch_metadata
        )
        return metadata if isinstance(metadata, dict) else None

    def _on_launch_enter(self, launch_metadata):
        if not self.tracker.enabled:
            return

        metadata = self._metadata_dict(launch_metadata)
        if metadata is None:
            return
        kernel_name = metadata.get("name", "unknown")
        function = metadata.get("function")
        if function is None:
            raise RuntimeError(
                f"Cannot match kernel '{kernel_name}': function pointer is None. "
                "Triton launch metadata should contain 'function'."
            )
        if function not in self.function_to_hash:
            raise RuntimeError(
                f"Cannot match kernel '{kernel_name}': function pointer {function} is not "
                "registered by the kernel-load hook. "
                f"Available mappings: {list(self.function_to_hash.keys())}"
            )

        kernel_hash = self.function_to_hash[function]
        kernel = self.session.compiled_kernels.get(kernel_hash)
        if kernel is None:
            raise RuntimeError(
                f"Cannot match kernel '{kernel_name}': hash {kernel_hash} is missing from "
                "the tracking session."
            )
        if self.pending_grid is None:
            raise RuntimeError(
                f"Grid size was not captured for kernel '{kernel_name}'. "
                f"Available launch metadata keys: {list(metadata.keys())}"
            )

        _, self.pending_snapshots = self.session.record_launch(
            kernel,
            self.pending_grid,
            self.pending_args or (),
            snapshot_tensors=True,
        )

    def _on_launch_exit(self, launch_metadata):
        if not self.tracker.enabled:
            return

        metadata = self._metadata_dict(launch_metadata)
        if metadata is None or not self.session.kernel_launches:
            return

        launch = self.session.kernel_launches[-1]
        kernel_name = metadata.get("name", "unknown")
        if launch.kernel_name != kernel_name:
            print(
                "[WARNING] Kernel name mismatch in launch_exit: "
                f"expected {launch.kernel_name}, got {kernel_name}"
            )
            return

        try:
            self.session.capture_outputs(
                launch, self.pending_args or (), self.pending_snapshots
            )
            if (
                self.session.save_binaries
                and self.session.capture_args
                and self.pending_snapshots is not None
            ):
                kernel = self.session.compiled_kernels.get(launch.kernel_hash)
                if kernel is not None:
                    print("  Generating harness with validation code...")
                    self.harness.generate_launch_harness(
                        kernel, launch, validate_outputs=True
                    )
        finally:
            self._clear_pending()

    def _clear_pending(self) -> None:
        self.pending_args = None
        self.pending_grid = None
        self.pending_snapshots = None
