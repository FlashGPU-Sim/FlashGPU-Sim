"""Runner-specific exceptions."""


class RunnerError(RuntimeError):
    """A user-facing test runner failure."""


class DiscoveryError(RunnerError):
    """A GoogleTest binary could not be queried."""
