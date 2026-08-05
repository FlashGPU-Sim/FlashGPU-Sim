"""Console output and subprocess helpers."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import IO, Mapping, Sequence


RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BLUE = "\033[0;34m"
RESET = "\033[0m"


class UI:
    """Write diagnostics and run visible subprocesses on one output stream."""

    def __init__(self, stream: IO[str] | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdout

    def color(self, color: str, message: str) -> None:
        print(f"{color}{message}{RESET}", file=self.stream, flush=True)

    def info(self, message: str) -> None:
        self.color(BLUE, message)

    def success(self, message: str) -> None:
        self.color(GREEN, message)

    def warning(self, message: str) -> None:
        self.color(YELLOW, message)

    def error(self, message: str) -> None:
        print(f"{RED}{message}{RESET}", file=sys.stderr, flush=True)

    def plain(self, message: str = "") -> None:
        print(message, file=self.stream, flush=True)

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
    ) -> int:
        self.warning(f"⚡ Running: {shlex.join(command)}")

        # list-cases reserves stdout for machine-readable case names. When the
        # UI is attached to another stream, route child diagnostics there too.
        child_stream: IO[str] | None = None
        if self.stream is not sys.stdout:
            child_stream = self.stream

        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=dict(env) if env is not None else os.environ.copy(),
            stdout=child_stream,
            stderr=child_stream,
            check=False,
        )
        return completed.returncode
