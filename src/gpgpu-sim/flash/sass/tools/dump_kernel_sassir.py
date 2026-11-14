#!/usr/bin/env python3
"""Decode one launched kernel and dump SASSIR for this simulator run."""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from nvdisasm_to_sassir import elf64_sections, export


ELF_FILE = re.compile(r"^ELF file\s+\d+:\s+(.+)$")


def run_tool(command: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{' '.join(command[:2])} failed: {detail}")
    return completed.stdout


def listed_cubins(cuobjdump: Path, binary: Path) -> list[str]:
    output = run_tool([str(cuobjdump), "--list-elf", str(binary)])
    listed = [
        match.group(1).strip()
        for line in output.splitlines()
        if (match := ELF_FILE.match(line)) is not None
    ]
    if not listed:
        raise RuntimeError(f"no cubin images found in {binary}")
    for name in listed:
        if Path(name).name != name:
            raise RuntimeError(f"unsafe cubin name reported by cuobjdump: {name}")
    # CUDA 13.3 reports the same sm_120a output name twice for cubins that
    # contain both the ordinary and Mercury companion sections.  The name is
    # also the extraction key, so those entries cannot identify distinct
    # images and extracting both merely overwrites the same file.  Preserve
    # order while collapsing exact-name duplicates; genuinely different cubin
    # names still participate in the strict unique-kernel check below.
    return list(dict.fromkeys(listed))


def extract_kernel_cubin(
    cuobjdump: Path,
    binary: Path,
    kernel: str,
    fatbin_handle: int,
    arch: str,
    nvdisasm: Path,
    directory: Path,
) -> Path:
    cubins = listed_cubins(cuobjdump, binary)
    # __cudaRegisterFatBinary handles identify runtime fatbin registrations;
    # they are not cuobjdump's ELF ordinals. In particular, CUDA 13 may emit
    # an empty compatibility ELF before the executable kernel ELF within one
    # registration. Use the handle only as a search-order hint, then identify
    # the image by its exact .text.<mangled kernel> section.
    preferred = fatbin_handle - 1
    order = list(range(len(cubins)))
    if 0 <= preferred < len(cubins):
        order.remove(preferred)
        order.insert(0, preferred)

    matches: list[Path] = []
    for index in order:
        name = cubins[index]
        run_tool(
            [str(cuobjdump), "--extract-elf", name, str(binary)],
            cwd=directory,
        )
        cubin = directory / name
        if not cubin.is_file():
            raise RuntimeError(f"cuobjdump did not create {cubin}")
        sections = elf64_sections(cubin.read_bytes())
        if any(section.name == f".text.{kernel}" for section in sections):
            matches.append(cubin)

    if not matches:
        raise RuntimeError(
            f"registered kernel {kernel!r} is absent from all "
            f"{len(cubins)} cubins in {binary}"
        )
    if len(matches) == 1:
        return matches[0]

    # Separate translation units may instantiate the same externally named
    # CUDA template.  Such a kernel is safe to select only when every detail
    # consumed by the frontend is identical; comparing complete generated
    # sassirs covers its resources, ABI, constant banks, instruction bytes,
    # official decode, and scheduling controls.  Retain the strict ambiguity
    # error when same-named definitions differ in any of those properties.
    sassirs: list[str] = []
    for match in matches:
        output = io.StringIO()
        decoded, unknown = export(
            match, [kernel], arch, nvdisasm, cuobjdump, output
        )
        if decoded == 0 or unknown != 0:
            raise RuntimeError(
                f"official decoder produced {decoded} instructions and "
                f"{unknown} unknowns for duplicate kernel {kernel} in "
                f"{match.name}"
            )
        sassirs.append(output.getvalue())
    if any(sassir != sassirs[0] for sassir in sassirs[1:]):
        names = ", ".join(match.name for match in matches)
        raise RuntimeError(
            f"registered kernel {kernel!r} has differing definitions across "
            f"cubins: {names}"
        )

    preferred_name = cubins[preferred] if 0 <= preferred < len(cubins) else None
    return next(
        (match for match in matches if match.name == preferred_name), matches[0]
    )


def dump_sassir(
    cubin: Path,
    kernel: str,
    arch: str,
    nvdisasm: Path,
    cuobjdump: Path,
    destination: Path,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            decoded, unknown = export(
                cubin, [kernel], arch, nvdisasm, cuobjdump, output
            )
            output.flush()
            os.fsync(output.fileno())
        if decoded == 0 or unknown != 0:
            raise RuntimeError(
                f"official decoder produced {decoded} instructions and "
                f"{unknown} unknowns for {kernel}"
            )
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--fatbin-handle", required=True, type=int)
    parser.add_argument("--arch", required=True, choices=("sm90", "sm120"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--nvdisasm", required=True, type=Path)
    parser.add_argument("--cuobjdump", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    binary = arguments.binary.expanduser().resolve()
    # Extraction runs in a temporary cwd. Resolve tool paths against the
    # caller's cwd before passing them to any subprocess.
    arguments.nvdisasm = arguments.nvdisasm.expanduser().resolve()
    arguments.cuobjdump = arguments.cuobjdump.expanduser().resolve()
    arguments.output = arguments.output.expanduser().resolve()
    if not binary.is_file():
        raise RuntimeError(f"executable not found: {binary}")
    for tool in (arguments.nvdisasm, arguments.cuobjdump):
        if not tool.is_file():
            raise RuntimeError(f"CUDA tool not found: {tool}")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="extract-", dir=arguments.output.parent
    ) as temporary:
        cubin = extract_kernel_cubin(
            arguments.cuobjdump,
            binary,
            arguments.kernel,
            arguments.fatbin_handle,
            arguments.arch,
            arguments.nvdisasm,
            Path(temporary),
        )
        sassir = dump_sassir(
            cubin,
            arguments.kernel,
            arguments.arch,
            arguments.nvdisasm,
            arguments.cuobjdump,
            arguments.output,
        )
    print(sassir.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
