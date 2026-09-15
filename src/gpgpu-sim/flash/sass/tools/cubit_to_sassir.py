#!/usr/bin/env python3
"""Export one cubin kernel to the legacy CuBit-backed SASSIR v1 format.

This reads the ELF code section directly and invokes cubit.decode() once per
128-bit instruction. Prefer nvdisasm_to_sassir.py for current SASSIR correctness
work; this exporter remains useful for bitfield and round-trip cross-checks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

# Shared static-image I/O only; this does not invoke the NVIDIA tools.
# CuBit decode and its independent scheduling-control check stay below.
from nvdisasm_to_sassir import (
    elf64_sections,
    escaped,
    find_kernel_section,
    raw_instructions as instructions,
    write_record,
)


def raw_control(hi: int) -> dict[str, int | bool]:
    packed = (hi >> 41) & 0x1FFFF
    return {
        "stall": packed & 0xF,
        "yield_flag": not bool((packed >> 4) & 1),
        "write_bar": (packed >> 5) & 0x7,
        "read_bar": (packed >> 8) & 0x7,
        "wait_mask": (packed >> 11) & 0x3F,
        "reuse_mask": (hi >> 58) & 0x7,
    }


def export(cubin: Path, kernel_name: str, arch: str, output: TextIO) -> tuple[int, int]:
    try:
        import cubit  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "CuBit's Python extension is required; install it with "
            "`pip install -e /path/to/cubit`"
        ) from error

    if not hasattr(cubit, "select_table"):
        raise RuntimeError("this exporter requires a CuBit build with select_table()")
    cubit.select_table(arch)

    data = cubin.read_bytes()
    section = find_kernel_section(elf64_sections(data), kernel_name)
    write_record(output, "SASSIR", 1)
    write_record(output, "KERNEL", escaped(kernel_name), arch, 16)

    decoded_count = 0
    unknown_count = 0
    for pc, lo, hi in instructions(data, section):
        control = raw_control(hi)
        decoded = None
        try:
            decoded = cubit.decode(lo, hi, pc)
        except ValueError:  # CuBit uses ValueError for an unknown encoding.
            unknown_count += 1

        if decoded is None:
            opcode = key = modifier_group = ""
            fields = []
        else:
            decoded_count += 1
            opcode = decoded["opcode"]
            key = decoded["key"]
            modifier_group = decoded["mod_group"]
            fields = decoded["fields"]
            scheduling = decoded["scheduling"]
            for name in ("stall", "yield_flag", "write_bar", "read_bar", "wait_mask"):
                if scheduling[name] != control[name]:
                    raise RuntimeError(
                        f"CuBit/raw control mismatch at pc 0x{pc:x}: {name} "
                        f"{scheduling[name]!r} != {control[name]!r}"
                    )

        write_record(
            output,
            "INST",
            hex(pc),
            hex(lo),
            hex(hi),
            int(decoded is not None),
            escaped(opcode),
            escaped(key),
            escaped(modifier_group),
            control["stall"],
            int(control["yield_flag"]),
            control["write_bar"],
            control["read_bar"],
            control["wait_mask"],
            control["reuse_mask"],
        )
        for field in fields:
            write_record(
                output,
                "FIELD",
                escaped(field["name"]),
                field["shift"],
                field["bits"],
                field["value"],
                field["token_idx"],
                escaped(field["extraction"]),
            )
        write_record(output, "ENDINST")

    write_record(output, "ENDKERNEL")
    return decoded_count, unknown_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cubin", type=Path)
    parser.add_argument("kernel")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--arch", default="sm120")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output is not None and args.output.resolve() == args.cubin.resolve():
        raise ValueError("output path must not overwrite the input cubin")
    stream: TextIO
    close_stream = False
    if args.output is None:
        stream = sys.stdout
    else:
        stream = args.output.open("w", encoding="utf-8")
        close_stream = True
    try:
        decoded, unknown = export(args.cubin, args.kernel, args.arch, stream)
    finally:
        if close_stream:
            stream.close()
    print(
        f"exported {decoded + unknown} static instructions "
        f"({decoded} decoded, {unknown} unknown)",
        file=sys.stderr,
    )
    return 0 if unknown == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
