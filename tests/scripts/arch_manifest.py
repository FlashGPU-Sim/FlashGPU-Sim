#!/usr/bin/env python3
"""Read and validate architecture-owned test source manifests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in the Ubuntu 22.04 CI image.
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as error:
        raise SystemExit(
            "TOML support requires Python 3.11+ or the 'tomli' package"
        ) from error


TEST_DIR = Path(__file__).resolve().parents[1]
ARCH_DIR = TEST_DIR / "arch"
SOURCE_DIR = TEST_DIR / "src"
CONFIG_DIR = TEST_DIR.parent / "configs"

ARCH_NAME_RE = re.compile(r"^sm[0-9]+$")
NVCC_TARGET_RE = re.compile(r"^sm_([1-9][0-9]+)([af]?)$")
CONFIG_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
TEST_GROUP_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
SOURCE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SOURCE_SUFFIXES = {".cc", ".cu"}


class ManifestError(ValueError):
    """Raised when an architecture manifest violates the repository schema."""


@dataclass(frozen=True)
class ArchitectureManifest:
    name: str
    nvcc_target: str
    config: str
    sources: tuple[str, ...]

    @property
    def compute_capability(self) -> str:
        match = NVCC_TARGET_RE.fullmatch(self.nvcc_target)
        assert match is not None
        digits = match.group(1)
        return f"{int(digits[:-1])}.{digits[-1]}"

    @property
    def compute_target(self) -> str:
        return self.nvcc_target.replace("sm_", "compute_", 1)

    @property
    def test_groups(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(source.split("/", 1)[0] for source in self.sources))

    def sources_for(self, test_group: str) -> tuple[str, ...]:
        prefix = f"{test_group}/"
        return tuple(source for source in self.sources if source.startswith(prefix))


def _expect_table(data: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ManifestError(f"{path}: [{key}] must be a TOML table")
    return value


def _check_exact_keys(
    table: dict[str, Any], expected: set[str], label: str, path: Path
) -> None:
    actual = set(table)
    if actual != expected:
        missing = ", ".join(sorted(expected - actual)) or "none"
        extra = ", ".join(sorted(actual - expected)) or "none"
        raise ManifestError(
            f"{path}: {label} keys do not match the schema "
            f"(missing: {missing}; extra: {extra})"
        )


def _read_config_compute_capability(config: str, manifest_path: Path) -> str:
    config_file = CONFIG_DIR / config / "gpgpusim.config"
    if not config_file.is_file():
        raise ManifestError(
            f"{manifest_path}: config does not exist: {config_file.relative_to(TEST_DIR.parent)}"
        )

    major: str | None = None
    minor: str | None = None
    for line in config_file.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        if fields[0] == "-gpgpu_compute_capability_major":
            major = fields[1]
        elif fields[0] == "-gpgpu_compute_capability_minor":
            minor = fields[1]

    if major is None or minor is None or not major.isdigit() or not minor.isdigit():
        raise ManifestError(f"{manifest_path}: cannot read compute capability from {config_file}")
    return f"{int(major)}.{int(minor)}"


def load_manifest(path: Path) -> ArchitectureManifest:
    try:
        with path.open("rb") as manifest_file:
            data = tomllib.load(manifest_file)
    except tomllib.TOMLDecodeError as error:
        raise ManifestError(f"{path}: invalid TOML: {error}") from error

    if not isinstance(data, dict):
        raise ManifestError(f"{path}: manifest root must be a TOML table")
    _check_exact_keys(data, {"arch", "workloads"}, "top-level", path)

    name = path.stem
    if not ARCH_NAME_RE.fullmatch(name):
        raise ManifestError(f"{path}: architecture filename must match sm<digits>.toml")

    arch = _expect_table(data, "arch", path)
    _check_exact_keys(arch, {"config", "nvcc_target"}, "[arch]", path)
    nvcc_target = arch["nvcc_target"]
    config = arch["config"]
    if not isinstance(nvcc_target, str) or not NVCC_TARGET_RE.fullmatch(nvcc_target):
        raise ManifestError(f"{path}: arch.nvcc_target must look like sm_90a")
    if not isinstance(config, str) or not CONFIG_NAME_RE.fullmatch(config):
        raise ManifestError(f"{path}: arch.config must be a configuration name")

    target_digits = NVCC_TARGET_RE.fullmatch(nvcc_target).group(1)  # type: ignore[union-attr]
    if name != f"sm{target_digits}":
        raise ManifestError(
            f"{path}: filename {name} does not match nvcc_target {nvcc_target}"
        )

    workloads = _expect_table(data, "workloads", path)
    _check_exact_keys(workloads, {"sources"}, "[workloads]", path)
    raw_sources = workloads["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ManifestError(f"{path}: workloads.sources must be a non-empty array")

    sources: list[str] = []
    seen_sources: set[str] = set()
    for index, source in enumerate(raw_sources):
        label = f"{path}: workloads.sources[{index}]"
        if not isinstance(source, str):
            raise ManifestError(f"{label} must be a string")
        posix_path = PurePosixPath(source)
        if (
            posix_path.is_absolute()
            or "\\" in source
            or any(part in {"", ".", ".."} for part in posix_path.parts)
            or any(
                not SOURCE_COMPONENT_RE.fullmatch(part)
                for part in posix_path.parts
            )
            or len(posix_path.parts) < 2
            or str(posix_path) != source
        ):
            raise ManifestError(
                f"{label} must be a normalized path below tests/src/<test_group>/"
            )
        test_group = posix_path.parts[0]
        if not TEST_GROUP_RE.fullmatch(test_group):
            raise ManifestError(f"{label} has an invalid test_group: {test_group}")
        if posix_path.suffix not in SOURCE_SUFFIXES:
            raise ManifestError(f"{label} must name a .cu or .cc source")
        if source in seen_sources:
            raise ManifestError(f"{label} duplicates {source}")
        if not (SOURCE_DIR / posix_path).is_file():
            raise ManifestError(f"{label} does not exist below tests/src: {source}")
        seen_sources.add(source)
        sources.append(source)

    manifest = ArchitectureManifest(name, nvcc_target, config, tuple(sources))
    config_cc = _read_config_compute_capability(config, path)
    if config_cc != manifest.compute_capability:
        raise ManifestError(
            f"{path}: {nvcc_target} requires compute capability "
            f"{manifest.compute_capability}, but {config} declares {config_cc}"
        )
    return manifest


def load_manifests() -> tuple[ArchitectureManifest, ...]:
    paths = sorted(ARCH_DIR.glob("*.toml"))
    if not paths:
        raise ManifestError(f"no architecture manifests found below {ARCH_DIR}")
    manifests = tuple(load_manifest(path) for path in paths)

    names = [manifest.name for manifest in manifests]
    if len(names) != len(set(names)):
        raise ManifestError("architecture names must be unique")

    integration_sources = {
        path.relative_to(SOURCE_DIR).as_posix()
        for path in (SOURCE_DIR / "integration").iterdir()
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    }
    for manifest in manifests:
        selected = set(manifest.sources_for("integration"))
        if selected != integration_sources:
            missing = ", ".join(sorted(integration_sources - selected)) or "none"
            extra = ", ".join(sorted(selected - integration_sources)) or "none"
            raise ManifestError(
                f"{manifest.name}: integration must contain the complete portable "
                f"source set (missing: {missing}; extra: {extra})"
            )
    return manifests


def _make_list(name: str, values: tuple[str, ...] | list[str]) -> list[str]:
    if not values:
        return [f"{name} :="]
    lines = [f"{name} := \\"]
    for index, value in enumerate(values):
        suffix = " \\" if index + 1 < len(values) else ""
        lines.append(f"\t{value}{suffix}")
    return lines


def render_make(manifests: tuple[ArchitectureManifest, ...]) -> str:
    lines = [
        "# Generated from tests/arch/*.toml by scripts/arch_manifest.py.",
        "# Do not edit this file directly.",
        "",
    ]
    lines.extend(_make_list("ARCHITECTURES", [item.name for item in manifests]))
    lines.append("")
    for manifest in manifests:
        prefix = manifest.name
        lines.extend(
            [
                f"ARCH_DEFAULT_CONFIG_{prefix} := {manifest.config}",
                f"ARCH_NVCC_TARGET_{prefix} := {manifest.nvcc_target}",
                f"ARCH_COMPUTE_TARGET_{prefix} := {manifest.compute_target}",
            ]
        )
        lines.extend(_make_list(f"ARCH_TEST_GROUPS_{prefix}", list(manifest.test_groups)))
        for test_group in manifest.test_groups:
            make_sources = [
                f"$(TEST_SRC_DIR)/{source}"
                for source in manifest.sources_for(test_group)
            ]
            lines.extend(
                _make_list(
                    f"TEST_GROUP_SOURCES_{prefix}_{test_group}", make_sources
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_if_changed(path: Path, content: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate every architecture manifest")
    emit = subparsers.add_parser("emit-make", help="write generated Make variables")
    emit.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        manifests = load_manifests()
    except (ManifestError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if args.command == "validate":
        for manifest in manifests:
            print(
                f"{manifest.name}: {manifest.nvcc_target}, {manifest.config}, "
                f"{len(manifest.sources)} sources, "
                f"{len(manifest.test_groups)} test groups"
            )
    elif args.command == "emit-make":
        write_if_changed(args.output, render_make(manifests))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
