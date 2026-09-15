#!/usr/bin/env python3
"""Export a cubin kernel with NVIDIA's versioned nvdisasm JSON interface.

nvdisasm supplies the canonical mnemonic, predicate, and operand spelling.
FlashGPU-Sim independently reads each raw 128-bit instruction from the ELF and
decodes its scheduling control; this is a static image, never a dynamic trace.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Iterator, NamedTuple, TextIO
from urllib.parse import quote


class ElfSection(NamedTuple):
    name: str
    offset: int
    size: int


class KernelParameter(NamedTuple):
    ordinal: int
    offset: int
    size: int


class KernelAbi(NamedTuple):
    parameter_base: int
    parameter_size: int
    parameters: list[KernelParameter]


class KernelResources(NamedTuple):
    registers: int
    static_shared: int
    local_memory: int
    stack_size: int


class KernelConstantBank(NamedTuple):
    index: int
    data: bytes


PARAMETER_BASE = {"sm90": 0x210, "sm120": 0x380}
SHT_NOBITS = 8


def _c_string(data: bytes, offset: int) -> str:
    if offset >= len(data):
        raise ValueError("ELF section-name offset is out of range")
    end = data.find(b"\0", offset)
    if end < 0:
        raise ValueError("unterminated ELF section name")
    return data[offset:end].decode("utf-8", errors="strict")


def elf64_sections(data: bytes) -> list[ElfSection]:
    if data[:4] != b"\x7fELF":
        raise ValueError("input is not an ELF file")
    if len(data) < 64:
        raise ValueError("truncated ELF header")
    if data[4] != 2 or data[5] != 1:
        raise ValueError("only little-endian ELF64 cubins are supported")
    table = struct.unpack_from("<Q", data, 0x28)[0]
    entry_size = struct.unpack_from("<H", data, 0x3A)[0]
    count = struct.unpack_from("<H", data, 0x3C)[0]
    names_index = struct.unpack_from("<H", data, 0x3E)[0]
    if entry_size < 64 or count == 0 or names_index >= count:
        raise ValueError("unsupported ELF section table")
    if table + entry_size * count > len(data):
        raise ValueError("truncated ELF section table")

    headers: list[tuple[int, int, int]] = []
    for index in range(count):
        base = table + index * entry_size
        name = struct.unpack_from("<I", data, base)[0]
        section_type = struct.unpack_from("<I", data, base + 0x04)[0]
        offset = struct.unpack_from("<Q", data, base + 0x18)[0]
        size = struct.unpack_from("<Q", data, base + 0x20)[0]
        # SHT_NOBITS sections reserve memory but intentionally have no bytes in
        # the ELF file. Hopper cubins use them for static storage metadata.
        if section_type != SHT_NOBITS and offset + size > len(data):
            raise ValueError(f"ELF section {index} extends past end of file")
        headers.append((name, offset, size))
    _, names_offset, names_size = headers[names_index]
    names = data[names_offset : names_offset + names_size]
    return [
        ElfSection(_c_string(names, name), offset, size)
        for name, offset, size in headers
    ]


def find_kernel_section(sections: list[ElfSection], kernel_name: str) -> ElfSection:
    exact = [s for s in sections if s.name == f".text.{kernel_name}"]
    if len(exact) == 1:
        return exact[0]
    candidates = [
        s for s in sections if s.name.startswith(".text.") and kernel_name in s.name
    ]
    if len(candidates) != 1:
        names = ", ".join(section.name for section in candidates) or "none"
        raise ValueError(
            f"kernel {kernel_name!r} did not select one text section; matches: {names}"
        )
    return candidates[0]


def find_kernel_constant_banks(
    data: bytes, sections: list[ElfSection], kernel_name: str
) -> list[KernelConstantBank]:
    pattern = re.compile(rf"\.nv\.constant([0-9]+)\.{re.escape(kernel_name)}")
    banks: dict[int, bytes] = {}
    for section in sections:
        match = pattern.fullmatch(section.name)
        if match is None:
            continue
        index = int(match.group(1), 10)
        # Bank zero is populated from the CUDA launch ABI at runtime. Cubin
        # banks above zero contain immutable compiler data such as BRX jump
        # tables and must travel with the static instruction image.
        if index == 0:
            continue
        if index in banks:
            raise RuntimeError(
                f"duplicate constant bank {index} for kernel {kernel_name}"
            )
        banks[index] = data[section.offset : section.offset + section.size]
    return [KernelConstantBank(index, banks[index]) for index in sorted(banks)]


def raw_instructions(
    data: bytes, section: ElfSection
) -> Iterator[tuple[int, int, int]]:
    code = data[section.offset : section.offset + section.size]
    if len(code) % 16 != 0:
        raise ValueError(
            f"kernel section size {len(code)} is not a multiple of 16 bytes"
        )
    for pc in range(0, len(code), 16):
        lo, hi = struct.unpack_from("<QQ", code, pc)
        yield pc, lo, hi


def raw_control(hi: int, arch: str) -> dict[str, int | bool]:
    packed = (hi >> 41) & 0x1FFFF
    return {
        "stall": packed & 0xF,
        "yield_flag": not bool((packed >> 4) & 1),
        "write_bar": (packed >> 5) & 0x7,
        "read_bar": (packed >> 8) & 0x7,
        "wait_mask": (packed >> 11) & 0x3F,
        "reuse_mask": (hi >> 58) & (0xF if arch == "sm90" else 0x7),
    }


def escaped(value: object) -> str:
    return quote(str(value), safe="-_.~")


def write_record(output: TextIO, *columns: object) -> None:
    output.write("\t".join(str(column) for column in columns))
    output.write("\n")


def run_nvdisasm(binary: Path, cubin: Path) -> list[object]:
    command = [str(binary), "--emit-json", "--print-code", str(cubin)]
    try:
        result = subprocess.run(
            command, check=False, text=True, capture_output=True, encoding="utf-8"
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"nvdisasm not found: {binary}") from error
    if result.returncode != 0:
        raise RuntimeError(
            f"nvdisasm exited with status {result.returncode}: {result.stderr.strip()}"
        )
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid nvdisasm JSON: {error}") from error
    if not isinstance(document, list) or len(document) < 2:
        raise RuntimeError("nvdisasm JSON has no metadata/function list")
    return document


def run_cuobjdump(binary: Path, cubin: Path, *options: str) -> str:
    command = [str(binary), *options, str(cubin)]
    try:
        result = subprocess.run(
            command, check=False, text=True, capture_output=True, encoding="utf-8"
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"cuobjdump not found: {binary}") from error
    if result.returncode != 0:
        raise RuntimeError(
            f"cuobjdump exited with status {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def parse_kernel_resources(resource_dump: str, kernel_name: str) -> KernelResources:
    heading = re.search(
        rf"^ Function {re.escape(kernel_name)}:\s*$",
        resource_dump,
        flags=re.MULTILINE,
    )
    if heading is None:
        raise RuntimeError(f"cuobjdump resource usage has no function {kernel_name}")
    section = resource_dump[heading.end() :]
    next_function = re.search(r"^ Function \S+:\s*$", section, flags=re.MULTILINE)
    if next_function is not None:
        section = section[: next_function.start()]
    values: dict[str, int] = {}
    for name in ("REG", "STACK", "SHARED", "LOCAL"):
        match = re.search(rf"\b{name}:(\d+)\b", section)
        if match is None:
            raise RuntimeError(
                f"cuobjdump resource usage for {kernel_name} has no {name} value"
            )
        values[name] = int(match.group(1))
    return KernelResources(
        registers=values["REG"],
        static_shared=values["SHARED"],
        local_memory=values["LOCAL"],
        stack_size=values["STACK"],
    )


def parse_kernel_abi(elf_dump: str, kernel_name: str, arch: str) -> KernelAbi:
    section_name = f".nv.info.{kernel_name}"
    heading = re.search(
        rf"^{re.escape(section_name)}\s*$", elf_dump, flags=re.MULTILINE
    )
    if heading is None:
        raise RuntimeError(f"cuobjdump did not emit {section_name}")
    section = elf_dump[heading.end() :]
    next_section = re.search(r"^\S.*$", section, flags=re.MULTILINE)
    if next_section is not None:
        section = section[: next_section.start()]

    parameters = [
        KernelParameter(*(int(value, 0) for value in match.groups()))
        for match in re.finditer(
            r"Ordinal\s*:\s*(0x[0-9a-fA-F]+|\d+)\s+"
            r"Offset\s*:\s*(0x[0-9a-fA-F]+|\d+)\s+"
            r"Size\s*:\s*(0x[0-9a-fA-F]+|\d+)",
            section,
        )
    ]
    size_match = re.search(
        r"EIATTR_CBANK_PARAM_SIZE.*?Value:\s*(0x[0-9a-fA-F]+|\d+)",
        section,
        flags=re.DOTALL,
    )
    bank_match = re.search(
        r"EIATTR_PARAM_CBANK.*?Value:\s*"
        r"(?:0x[0-9a-fA-F]+|\d+)\s+(0x[0-9a-fA-F]+|\d+)",
        section,
        flags=re.DOTALL,
    )
    if (size_match is None) != (bank_match is None):
        raise RuntimeError("incomplete kernel parameter bank metadata")
    if size_match is None:
        if parameters:
            raise RuntimeError("cuobjdump emitted parameters without their cbank")
        # Parameterless kernels still consume their architecture's fixed
        # launch header in constant bank zero. Record an explicit zero-sized
        # ABI so runtime execution can distinguish it from an old sassir
        # with no ABI.
        return KernelAbi(PARAMETER_BASE[arch], 0, [])
    parameter_size = int(size_match.group(1), 0)
    packed_bank = int(bank_match.group(1), 0)
    parameter_base = packed_bank & 0xFFFF
    packed_size = packed_bank >> 16
    if packed_size != parameter_size:
        raise RuntimeError(
            "cuobjdump parameter size disagrees with EIATTR_PARAM_CBANK"
        )
    parameters.sort(key=lambda parameter: parameter.ordinal)
    if [parameter.ordinal for parameter in parameters] != list(
        range(len(parameters))
    ):
        raise RuntimeError("kernel parameter ordinals are not contiguous")
    occupied: set[int] = set()
    for parameter in parameters:
        if parameter.size <= 0 or parameter.offset + parameter.size > parameter_size:
            raise RuntimeError("kernel parameter is outside its constant bank")
        byte_range = set(range(parameter.offset, parameter.offset + parameter.size))
        if occupied & byte_range:
            raise RuntimeError("kernel parameters overlap")
        occupied |= byte_range
    return KernelAbi(parameter_base, parameter_size, parameters)


def find_json_function_sequence(
    document: list[object], kernel_name: str
) -> list[dict[str, object]]:
    matches: list[tuple[list[object], int]] = []
    for group in document[1:]:
        if not isinstance(group, list):
            continue
        for index, candidate in enumerate(group):
            if (
                isinstance(candidate, dict)
                and candidate.get("function-name") == kernel_name
            ):
                matches.append((group, index))
    if len(matches) != 1:
        raise RuntimeError(
            f"nvdisasm selected {len(matches)} functions named {kernel_name!r}"
        )

    # nvdisasm emits device helper routines as separate, nonzero-start JSON
    # functions immediately after the entry function whose ELF text section
    # contains them. Preserve those official decodes in the kernel image; a
    # later zero-start function begins the next text section.
    group, entry_index = matches[0]
    sequence: list[dict[str, object]] = []
    for candidate in group[entry_index:]:
        if not isinstance(candidate, dict):
            raise RuntimeError("nvdisasm function list contains a non-object")
        start = candidate.get("start")
        if sequence and start == 0:
            break
        sequence.append(candidate)
    return sequence


def json_string(
    record: dict[str, object], field: str, *, pc: int, optional: bool = False
) -> str:
    if optional and field not in record:
        return ""
    value = record.get(field)
    if not isinstance(value, str):
        raise RuntimeError(
            f"nvdisasm {field!r} at pc 0x{pc:x} is not a JSON string"
        )
    return value


def json_attributes(record: dict[str, object], *, pc: int) -> dict[str, str]:
    value = record.get("other-attributes", {})
    if not isinstance(value, dict):
        raise RuntimeError(
            f"nvdisasm 'other-attributes' at pc 0x{pc:x} is not a JSON object"
        )
    result: dict[str, str] = {}
    for name, attribute in value.items():
        if not isinstance(name, str) or not isinstance(attribute, str):
            raise RuntimeError(
                f"nvdisasm attribute at pc 0x{pc:x} is not string-valued"
            )
        result[name] = attribute
    return result


def export_kernel(
    raw_data: bytes,
    sections: list[ElfSection],
    document: list[object],
    resource_dump: str,
    kernel_name: str,
    arch: str,
    cuobjdump: Path,
    cubin: Path,
    output: TextIO,
) -> tuple[int, int]:
    section = find_kernel_section(sections, kernel_name)
    raw = list(raw_instructions(raw_data, section))
    abi = parse_kernel_abi(
        run_cuobjdump(cuobjdump, cubin, "-elf", "-fun", kernel_name), kernel_name, arch
    )
    resources = parse_kernel_resources(resource_dump, kernel_name)
    functions = find_json_function_sequence(document, kernel_name)
    function = functions[0]
    entry_disassembly = function.get("sass-instructions")
    if not isinstance(entry_disassembly, list):
        raise RuntimeError("nvdisasm function has no instruction list")
    if function.get("start") != 0:
        raise RuntimeError(
            "only section-local kernel entry pc 0 is supported; nvdisasm "
            f"reported {function.get('start')!r}"
        )
    length = function.get("length")
    if length != len(raw) * 16:
        raise RuntimeError(
            "ELF/nvdisasm instruction alignment mismatch: "
            f"section={len(raw)}, length={length!r}"
        )

    disassembly = list(entry_disassembly)
    next_pc = len(disassembly) * 16
    for helper in functions[1:]:
        helper_start = helper.get("start")
        helper_length = helper.get("length")
        helper_disassembly = helper.get("sass-instructions")
        if (
            not isinstance(helper_start, int)
            or helper_start != next_pc
            or not isinstance(helper_disassembly, list)
            or helper_length != len(helper_disassembly) * 16
        ):
            raise RuntimeError(
                "nvdisasm helper instruction alignment mismatch: "
                f"expected-start={next_pc}, start={helper_start!r}, "
                f"length={helper_length!r}"
            )
        disassembly.extend(helper_disassembly)
        next_pc += helper_length
    if next_pc != length or len(disassembly) != len(raw):
        raise RuntimeError(
            "ELF/nvdisasm instruction coverage mismatch: "
            f"section={len(raw)}, json={len(disassembly)}, length={length!r}"
        )
    helper_targets: dict[str, int] = {}
    for helper in functions[1:]:
        helper_name = helper.get("function-name")
        helper_start = helper.get("start")
        if not isinstance(helper_name, str) or not isinstance(helper_start, int):
            raise RuntimeError("nvdisasm helper identity is malformed")
        helper_targets[helper_name] = helper_start

    write_record(output, "KERNEL", escaped(kernel_name), arch, 16)
    write_record(
        output,
        "RESOURCES",
        resources.registers,
        resources.static_shared,
        resources.local_memory,
        resources.stack_size,
    )
    write_record(
        output, "PARAMBANK", hex(abi.parameter_base), hex(abi.parameter_size)
    )
    for parameter in abi.parameters:
        write_record(
            output,
            "PARAM",
            parameter.ordinal,
            hex(parameter.offset),
            hex(parameter.size),
        )
    for bank in find_kernel_constant_banks(raw_data, sections, kernel_name):
        write_record(output, "CONSTBANK", bank.index, bank.data.hex())

    decoded_count = 0
    unknown_count = 0
    for (pc, lo, hi), decoded in zip(raw, disassembly):
        if not isinstance(decoded, dict):
            raise RuntimeError(f"nvdisasm instruction at pc 0x{pc:x} is not an object")
        opcode = json_string(decoded, "opcode", pc=pc, optional=True)
        predicate = json_string(decoded, "predicate", pc=pc, optional=True)
        operands = json_string(decoded, "operands", pc=pc, optional=True)
        attributes = json_attributes(decoded, pc=pc)
        if opcode.startswith("CALL.REL") and operands in helper_targets:
            if "target-pc" in attributes:
                raise RuntimeError(
                    f"nvdisasm call at pc 0x{pc:x} already has a target-pc"
                )
            attributes["target-pc"] = hex(helper_targets[operands])
        is_decoded = bool(opcode)
        decoded_count += int(is_decoded)
        unknown_count += int(not is_decoded)
        control = raw_control(hi, arch)
        write_record(
            output,
            "INST",
            hex(pc),
            hex(lo),
            hex(hi),
            int(is_decoded),
            "nvdisasm",
            escaped(predicate),
            escaped(opcode),
            escaped(operands),
            control["stall"],
            int(control["yield_flag"]),
            control["write_bar"],
            control["read_bar"],
            control["wait_mask"],
            control["reuse_mask"],
        )
        for name, value in sorted(attributes.items()):
            write_record(output, "ATTR", escaped(name), escaped(value))
        write_record(output, "ENDINST")
    write_record(output, "ENDKERNEL")
    return decoded_count, unknown_count


def export(
    cubin: Path,
    kernel_names: list[str],
    arch: str,
    nvdisasm: Path,
    cuobjdump: Path,
    output: TextIO,
) -> tuple[int, int]:
    if len(kernel_names) != len(set(kernel_names)):
        raise ValueError("kernel names must be unique")
    raw_data = cubin.read_bytes()
    sections = elf64_sections(raw_data)
    document = run_nvdisasm(nvdisasm, cubin)
    resource_dump = run_cuobjdump(cuobjdump, cubin, "-res-usage")
    metadata = document[0]
    if not isinstance(metadata, dict):
        raise RuntimeError("nvdisasm JSON metadata is not an object")
    sm = metadata.get("SM")
    if not isinstance(sm, dict) or not isinstance(sm.get("version"), dict):
        raise RuntimeError("nvdisasm JSON SM version is missing")
    sm_version = sm["version"]
    if arch not in PARAMETER_BASE:
        raise RuntimeError(f"unsupported SASS architecture: {arch}")
    expected_sm = {"sm90": (9, 0), "sm120": (12, 0)}[arch]
    if (sm_version.get("major"), sm_version.get("minor")) != expected_sm:
        raise RuntimeError(
            f"requested {arch} but nvdisasm reported "
            f"SM {sm_version.get('major')}.{sm_version.get('minor')}"
        )
    schema = metadata.get("SchemaVersion", {})
    if not isinstance(schema, dict):
        raise RuntimeError("nvdisasm JSON schema version is missing")
    schema_components = [schema.get(name) for name in ("major", "minor", "revision")]
    if not all(isinstance(component, int) for component in schema_components):
        raise RuntimeError("nvdisasm JSON schema version is malformed")
    schema_text = ".".join(str(component) for component in schema_components)
    producer = metadata.get("Producer")
    if not isinstance(producer, str) or not producer:
        raise RuntimeError("nvdisasm JSON producer is missing")
    write_record(output, "SASSIR", 4)
    write_record(output, "DECODER", "nvdisasm", escaped(producer), schema_text)
    decoded_count = 0
    unknown_count = 0
    for kernel_name in kernel_names:
        decoded, unknown = export_kernel(
            raw_data,
            sections,
            document,
            resource_dump,
            kernel_name,
            arch,
            cuobjdump,
            cubin,
            output,
        )
        decoded_count += decoded
        unknown_count += unknown
    return decoded_count, unknown_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cubin", type=Path)
    parser.add_argument("kernel", nargs="+")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--arch", default="sm120")
    parser.add_argument(
        "--nvdisasm",
        type=Path,
        default=Path(os.environ.get("NVDISASM", "nvdisasm")),
    )
    parser.add_argument(
        "--cuobjdump",
        type=Path,
        default=None,
        help="CUDA cuobjdump used for official kernel parameter metadata",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output is not None and args.output.resolve() == args.cubin.resolve():
        raise ValueError("output path must not overwrite the input cubin")
    stream: TextIO = sys.stdout
    close_stream = False
    if args.output is not None:
        stream = args.output.open("w", encoding="utf-8")
        close_stream = True
    try:
        cuobjdump = args.cuobjdump
        if cuobjdump is None:
            sibling = args.nvdisasm.parent / "cuobjdump"
            cuobjdump = sibling if sibling.is_file() else Path("cuobjdump")
        decoded, unknown = export(
            args.cubin,
            args.kernel,
            args.arch,
            args.nvdisasm,
            cuobjdump,
            stream,
        )
    finally:
        if close_stream:
            stream.close()
    print(
        f"exported {decoded + unknown} static instructions "
        f"({decoded} decoded, {unknown} unknown) with nvdisasm",
        file=sys.stderr,
    )
    return 0 if unknown == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
