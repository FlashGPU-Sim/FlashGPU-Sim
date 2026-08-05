from __future__ import annotations

from typing import Optional

from .harness import HarnessGenerator
from .offline import OfflineTracker
from .online import OnlineTracker
from .session import TrackingSession


class Tracker:
    """Track Triton compilation and launch data for standalone replay."""

    def __init__(
        self,
        output_dir,
        save_binaries: bool = True,
        capture_args: bool = True,
        enabled: bool = True,
        mode: str = "online",
        target: Optional[str] = None,
    ):
        self.mode = mode.lower()
        if self.mode not in {"online", "offline"}:
            raise ValueError(
                f"Unsupported tracking mode '{mode}'; expected 'online' or 'offline'"
            )
        if self.mode == "online" and target is not None:
            raise ValueError("target is only valid when mode='offline'")

        self.offline_target = (
            OfflineTracker.parse_target(target) if self.mode == "offline" else None
        )
        self.target_name = (
            f"sm{self.offline_target.arch}"
            if self.offline_target is not None
            else None
        )
        self.enabled = enabled
        self.session = TrackingSession(
            output_dir,
            save_binaries=save_binaries,
            capture_args=capture_args,
            mode=self.mode,
            target_name=self.target_name,
        )
        self.harness = HarnessGenerator(self.session)
        if self.mode == "online":
            self.backend = OnlineTracker(self, self.session, self.harness)
        else:
            self.backend = OfflineTracker(
                self,
                self.session,
                self.harness,
                self.offline_target,
            )
        self.backend.install()

        print("[TritonTracker] Initialized")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Save binaries: {self.save_binaries}")
        print(f"  Capture arguments: {self.capture_args}")
        print(f"  Tracking enabled: {self.enabled}")
        print(f"  Mode: {self.mode}")
        if self.target_name is not None:
            print(f"  Target: {self.target_name}")

    @property
    def output_dir(self):
        return self.session.output_dir

    @property
    def save_binaries(self):
        return self.session.save_binaries

    @property
    def capture_args(self):
        return self.session.capture_args

    @property
    def compiled_kernels(self):
        return self.session.compiled_kernels

    @property
    def kernel_launches(self):
        return self.session.kernel_launches

    @property
    def binaries_dir(self):
        return self.session.binaries_dir

    @property
    def launchers_dir(self):
        return self.session.launchers_dir

    @property
    def data_dir(self):
        return self.session.data_dir

    def enable(self) -> None:
        self.enabled = True
        print("[TritonTracker] Tracking ENABLED")

    def disable(self) -> None:
        self.enabled = False
        print("[TritonTracker] Tracking DISABLED")

    def is_enabled(self) -> bool:
        return self.enabled

    def save_summary(self) -> None:
        self.session.save()
