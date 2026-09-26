#!/usr/bin/env python3
"""Compile PTX with source markers and emit an nvdisasm reorder guide."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path


# Match quoted strings first, so comment delimiters inside filenames remain
# literal. Blank comments without removing newlines: .loc refers to the
# original PTX's physical line numbers, including multiline comments.
_PTX_STRING = r'"(?:\\.|[^"\\])*"'
_PTX_COMMENT_OR_STRING = re.compile(
    _PTX_STRING + r"|//[^\r\n]*|/\*.*?\*/", re.DOTALL
)


def without_comments(source: str) -> str:
    return _PTX_COMMENT_OR_STRING.sub(
        lambda match: match.group(0) if match.group(0).startswith('"') else
        re.sub(r"[^\r\n]", " ", match.group(0)),
        source,
    )


def _next_file_index(source: str) -> int:
    indices = [
        int(match.group(1))
        for match in re.finditer(r"^\s*\.file\s+(\d+)\b", source, re.MULTILINE)
    ]
    return max(indices, default=0) + 1


# Tokenize only enough PTX syntax to find statement boundaries. Strings are
# indivisible; operand braces belong to a pending statement, not a new scope.
_PTX_TOKEN = re.compile(_PTX_STRING + r"|[.$%A-Za-z_](?:[\w.$%]|::)*|[^\s]")


def _instruction_spans(source: str) -> list[tuple[int, int]]:
    """Find instruction spans in comment-free PTX, including multiline forms."""
    tokens = list(_PTX_TOKEN.finditer(source))
    spans: list[tuple[int, int]] = []
    awaiting_body = False
    depth = 0
    statement = None
    directive = False

    for index, token in enumerate(tokens):
        value = token.group(0)
        if depth == 0:
            if value in (".entry", ".func"):
                awaiting_body = True
            elif value == ";":
                awaiting_body = False
            elif awaiting_body and value == "{":
                depth = 1
                awaiting_body = False
            continue

        if statement is None:
            # Labels and scope delimiters can share a line with instructions.
            if index + 1 < len(tokens) and tokens[index + 1].group(0) == ":":
                continue
            if value == ":":
                continue
            if value in ("{", "}"):
                depth += 1 if value == "{" else -1
                continue
            statement = token
            directive = value.startswith(".")

        if value == ";":
            if not directive:
                spans.append((statement.start(), token.end()))
            statement = None
        elif directive and statement.group(0) in (".loc", ".file", ".section"):
            # These directives terminate at a newline rather than ';'.
            next_start = tokens[index + 1].start() if index + 1 < len(tokens) else len(source)
            if "\n" in source[token.end():next_start]:
                statement = None

    return spans


def normalize_ptx(source: str) -> str:
    """Put each complete instruction on its own line for parser/guide agreement."""
    source = without_comments(source)
    # Walk backwards so replacements do not invalidate the remaining spans.
    for start, end in reversed(_instruction_spans(source)):
        statement = source[start:end]
        # Preserve string literals; only whitespace outside strings is folded.
        statement = re.sub(_PTX_STRING + r"|\s+",
                           lambda m: m.group(0) if m.group(0).startswith('"') else " ",
                           statement)
        source = source[:start] + "\n" + statement + "\n" + source[end:]
    return source


def add_line_markers(source: str, source_name: str) -> tuple[str, int]:
    """Mark instructions using physical line numbers in the supplied PTX."""
    source = without_comments(source)
    file_index = _next_file_index(source)
    escaped_name = source_name.replace("\\", "\\\\").replace('"', '\\"')
    address = re.search(r"\.address_size\s+\d+", source)
    if not address:
        raise ValueError("PTX has no .address_size directive")
    insertions = [(address.end(), f'\n.file {file_index} "{escaped_name}"\n')]
    spans = _instruction_spans(source)
    if not spans:
        raise ValueError("PTX has no marked function-body instructions")
    for start, end in spans:
        line_number = source.count("\n", 0, end - 1) + 1
        line_start = source.rfind("\n", 0, start) + 1
        if not source[line_start:start].strip():
            start = line_start
        prefix = "" if start == 0 or source[start - 1] == "\n" else "\n"
        insertions.append((start, f"{prefix}.loc {file_index} {line_number} 0\n"))
    for position, text in sorted(insertions, reverse=True):
        source = source[:position] + text + source[position:]
    return source, len(spans)


_SASS_INSTRUCTION = re.compile(r"^\s*/\*[0-9a-fA-F]+\*/\s+.*?;", re.MULTILINE)


def instruction_stream(disassembly: str) -> tuple[str, ...]:
    """Return address-qualified instructions, excluding debug annotations."""
    return tuple(match.group(0).strip() for match in _SASS_INSTRUCTION.finditer(disassembly))


def _disassemble(nvdisasm: Path, cubin: Path, *options: str) -> str:
    result = subprocess.run(
        [str(nvdisasm), *options, str(cubin)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout


def generate_guide(
    ptx: Path,
    output_prefix: Path,
    *,
    ptxas: Path,
    nvdisasm: Path,
    arch: str,
) -> Path:
    """Generate lineinfo PTX, cubin, and guide using the configured toolkit (-O3)."""
    if not ptx.is_file():
        raise FileNotFoundError(f"PTX does not exist: {ptx}")
    for tool in (ptxas, nvdisasm):
        if not tool.is_file():
            raise FileNotFoundError(f"CUDA tool does not exist: {tool}")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    lineinfo_ptx = Path(f"{output_prefix}.ptx")
    cubin = Path(f"{output_prefix}.cubin")
    guide = Path(f"{output_prefix}.sass")
    normalized_ptx = Path(f"{output_prefix}.input.ptx")
    if ptx.resolve() in {path.resolve() for path in
                         (normalized_ptx, lineinfo_ptx, cubin, guide)}:
        raise ValueError("guide output paths must not overwrite the input PTX")
    source = ptx.read_text(encoding="utf-8")
    if not arch:
        target = re.search(r"^\s*\.target\s+(sm_[0-9]+[a-z]*)\b", without_comments(source), re.MULTILINE)
        if not target:
            raise ValueError("PTX has no supported .target directive")
        arch = target.group(1)
    normalized_source = normalize_ptx(source)
    normalized_ptx.write_text(normalized_source, encoding="utf-8")
    marked_source, _ = add_line_markers(normalized_source, normalized_ptx.name)
    lineinfo_ptx.write_text(marked_source, encoding="utf-8")

    with tempfile.TemporaryDirectory(
        prefix=f".{output_prefix.name}.verify-", dir=output_prefix.parent
    ) as temporary_dir:
        plain_cubin = Path(temporary_dir) / "plain.cubin"
        subprocess.run(
            [
                str(ptxas),
                f"-arch={arch}",
                "-O3",
                str(ptx),
                "-o",
                str(plain_cubin),
            ],
            check=True,
        )

        subprocess.run(
            [
                str(ptxas),
                f"-arch={arch}",
                "-lineinfo",
                "-O3",
                str(lineinfo_ptx),
                "-o",
                str(cubin),
            ],
            check=True,
        )

        plain_stream = instruction_stream(_disassemble(nvdisasm, plain_cubin))
        lineinfo_stream = instruction_stream(_disassemble(nvdisasm, cubin))
        if not plain_stream:
            raise RuntimeError(f"nvdisasm found no SASS instructions in {plain_cubin}")
        if plain_stream != lineinfo_stream:
            mismatch = next(
                (
                    index
                    for index, pair in enumerate(zip(plain_stream, lineinfo_stream))
                    if pair[0] != pair[1]
                ),
                min(len(plain_stream), len(lineinfo_stream)),
            )
            plain_value = plain_stream[mismatch] if mismatch < len(plain_stream) else "<end>"
            lineinfo_value = (
                lineinfo_stream[mismatch]
                if mismatch < len(lineinfo_stream)
                else "<end>"
            )
            raise RuntimeError(
                "adding PTX line markers changed the SASS instruction stream "
                f"at index {mismatch}: plain={plain_value!r}, "
                f"lineinfo={lineinfo_value!r}"
            )

    with guide.open("w", encoding="utf-8") as stream:
        subprocess.run(
            [str(nvdisasm), "-gi", str(cubin)], check=True, stdout=stream
        )
    return guide


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("ptx", type=Path)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--ptxas", type=Path, required=True)
    parser.add_argument("--nvdisasm", type=Path, required=True)
    parser.add_argument("--arch", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = generate_guide(
        args.ptx,
        args.output_prefix,
        ptxas=args.ptxas,
        nvdisasm=args.nvdisasm,
        arch=args.arch,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
