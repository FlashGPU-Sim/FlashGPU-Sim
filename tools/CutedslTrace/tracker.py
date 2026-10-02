"""Public, workload-independent offline tracker."""
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import weakref
from contextlib import ExitStack


class Tracker:
    _active = False

    def __init__(self, output_dir, *, target, sm_count):
        if not re.fullmatch(r"sm_\d+[af]?", target):
            raise ValueError("Specify a target such as sm_100a")
        if sm_count <= 0:
            raise ValueError("sm_count must be positive")
        self.output = Path(output_dir).resolve()
        self.target, self.sm_count = target, sm_count
        self.session = None
        self._temporary = None
        self._cleanup = None
        self._enabled = False
        self._exported = False

    def enable(self):
        """Start or resume capture. Only one tracker can be enabled at a time."""
        if self._enabled:
            return
        if self._exported:
            raise RuntimeError("This tracker has already exported; create a new Tracker")
        if Tracker._active:
            raise RuntimeError("Concurrent offline trackers are not supported")
        from .session import Session
        from .compiler import CompilerHooks
        if self.session is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="cute-offline-")
            self._cleanup = weakref.finalize(self, self._temporary.cleanup)
            work = Path(self._temporary.name)
            (work / "dump").mkdir()
            self.session = Session(work, self.target, self.sm_count)
        self.stack = ExitStack()
        Tracker._active = True
        self.session.active = True
        try:
            environment = {
                "CUTE_DSL_ARCH": self.target, "CUTE_DSL_KEEP": "ptx,cubin,ir",
                "CUTE_DSL_DUMP_DIR": str(self.session.output / "dump"),
            }
            for name, value in environment.items():
                previous = os.environ.get(name)
                if previous is None:
                    self.stack.callback(os.environ.pop, name, None)
                else:
                    self.stack.callback(os.environ.__setitem__, name, previous)
                os.environ[name] = value
            self.stack.enter_context(CompilerHooks(self.session))
        except BaseException:
            self.session.active = False
            self.stack.close()
            Tracker._active = False
            raise
        self._enabled = True

    def disable(self):
        """Restore the environment while retaining captured records for export."""
        if not self._enabled:
            return
        self.session.active = False
        try:
            self.stack.close()
        finally:
            self._enabled = False
            Tracker._active = False

    def check(self, tensor, *, expected):
        """Optionally compare the replayed tensor exactly with a scalar/CPU tensor."""
        if self.session is None or not self.session.active:
            raise RuntimeError("check() requires an enabled tracker")
        self.session.check(tensor, expected)

    def export(self):
        """Stop capture, publish one replay for all calls, and release temporary data."""
        from .harness import render
        self.disable()
        if self._exported:
            raise RuntimeError("This tracker has already exported")
        if self.session is None or not self.session.calls:
            raise RuntimeError("No CuTe DSL calls captured")
        render(self.session)
        manifest = dict(schema_version=2, mode="offline", target=self.target,
                        sm_count=self.sm_count, modules=self.session.modules,
                        calls=self.session.calls, storages=self.session.storages,
                        checks=self.session.checks,
                        versions={name: importlib.metadata.version(name) for name in
                                  ("nvidia-cutlass-dsl", "apache-tvm-ffi", "torch")})
        (self.session.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        self.output.mkdir(parents=True, exist_ok=True)
        for name in ("modules", "data", "harness.cc", "Makefile", "manifest.json"):
            source, dest = self.session.output / name, self.output / name
            if source.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(source, dest)
            else:
                shutil.copyfile(source, dest)
        self._exported = True
        self._cleanup()
