#!/usr/bin/env python3
"""Build a hand-auditable PTX order by replaying the SASS B/L/T stream.

This is intentionally not a policy search.  For each target region it takes the
SASS order of:

  B = BAR.SYNC
  L = LDSM / ldmatrix
  T = HMMA / mma.sync

and chooses one PTX instruction of that exact class, preserving RAW dependencies.
Non-B/L/T producer instructions are emitted immediately before the instruction
that needs them.  Branch and label boundaries are kept outside the replayed
regions.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegionSpec:
    name: str
    ptx_start: int
    ptx_end: int
    sass_start: int
    sass_end: int


@dataclass
class Node:
    ridx: int
    local_line: int
    text: str
    op: str
    token: str | None
    defs: frozenset[str]
    uses: frozenset[str]


def load_probe(repo: Path):
    path = repo / "test/scripts/ptx_sass_guided_scheduler_probe.py"
    spec = importlib.util.spec_from_file_location("ptx_probe", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def find_function(lines: list[str], needle: str) -> tuple[int, int]:
    func_re = re.compile(r"\s*(?:\.visible\s+)?\.(?:entry|func)\s+(\S+)\s*\(")
    start = None
    for i, line in enumerate(lines):
        match = func_re.match(line)
        if match and needle in match.group(1):
            start = i
            break
    if start is None:
        raise RuntimeError(f"function containing {needle!r} not found")
    for j in range(start + 1, len(lines)):
        if lines[j].strip() == "}":
            return start, j
    raise RuntimeError("function end not found")


def ptx_token(op: str) -> str | None:
    if op.startswith("bar."):
        return "B"
    if op.startswith("ldmatrix"):
        return "L"
    if op.startswith("mma.sync"):
        return "T"
    return None


def sass_tokens(lines: list[str], start: int, end: int) -> list[tuple[int, str, str]]:
    out = []
    for lineno in range(start, end + 1):
        text = lines[lineno - 1]
        if "BAR.SYNC" in text:
            out.append((lineno, "B", text.strip()))
        elif "LDSM" in text:
            out.append((lineno, "L", text.strip()))
        elif "HMMA" in text:
            out.append((lineno, "T", text.strip()))
    return out


def parse_region(probe, fn_lines: list[str], start: int, end: int) -> list[Node]:
    nodes: list[Node] = []
    for local_line in range(start, end + 1):
        raw = fn_lines[local_line - 1]
        parsed = probe.ptx_op_from_line(raw)
        if parsed is None:
            continue
        op, operands, pred = parsed
        defs, uses = probe.extract_ptx_defs_uses(op, operands, pred)
        nodes.append(
            Node(
                ridx=len(nodes),
                local_line=local_line,
                text=raw.strip(),
                op=op,
                token=ptx_token(op),
                defs=frozenset(defs),
                uses=frozenset(uses),
            )
        )
    return nodes


def build_raw_producers(nodes: list[Node]) -> list[list[int]]:
    last_writer: dict[str, int] = {}
    producers: list[list[int]] = [[] for _ in nodes]
    for node in nodes:
        deps = set()
        for reg in node.uses:
            if reg in last_writer:
                deps.add(last_writer[reg])
        producers[node.ridx] = sorted(deps)
        for reg in node.defs:
            last_writer[reg] = node.ridx
    return producers


def token_hist(seq: str) -> str:
    runs = []
    i = 0
    while i < len(seq):
        j = i + 1
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        runs.append((seq[i], j - i))
        i = j
    hist = Counter(length for tok, length in runs if tok == "T")
    return " ".join(f"{k}:{hist[k]}" for k in sorted(hist))


def tensor_tail_after4(seq: str) -> tuple[int, int, int]:
    total_t = seq.count("T")
    gt4_total = 0
    tail = 0
    i = 0
    while i < len(seq):
        j = i + 1
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        if seq[i] == "T" and j - i > 4:
            gt4_total += j - i
            tail += j - i - 4
        i = j
    return total_t, gt4_total, tail


def schedule_region(
    name: str,
    nodes: list[Node],
    target: list[tuple[int, str, str]],
) -> tuple[list[Node], list[str], list[str]]:
    producers = build_raw_producers(nodes)
    scheduled: set[int] = set()
    emitted: list[Node] = []
    map_rows: list[str] = []
    notes: list[str] = []

    def can_emit_non_blt(idx: int, seen: set[int] | None = None) -> bool:
        if idx in scheduled:
            return True
        if seen is None:
            seen = set()
        if idx in seen:
            return True
        seen.add(idx)
        node = nodes[idx]
        if node.token is not None:
            return False
        return all(can_emit_non_blt(dep, seen) for dep in producers[idx])

    def emit(idx: int, reason: str) -> None:
        if idx in scheduled:
            return
        node = nodes[idx]
        for dep in producers[idx]:
            if dep in scheduled:
                continue
            dep_node = nodes[dep]
            if dep_node.token is not None:
                raise RuntimeError(
                    f"{name}: cannot auto-emit {node.local_line} before "
                    f"{dep_node.token} producer line {dep_node.local_line}"
                )
            emit(dep, f"producer-for-{node.local_line}")
        scheduled.add(idx)
        emitted.append(node)
        if reason.startswith("producer"):
            notes.append(f"{name}\tAUTO\t{node.local_line}\t{node.op}\t{reason}\t{node.text}")

    target_pos = 0
    for sass_line, tok, sass_text in target:
        chosen = None
        blocked = []
        for node in nodes:
            if node.ridx in scheduled or node.token != tok:
                continue
            bad_deps = [
                dep for dep in producers[node.ridx]
                if dep not in scheduled and nodes[dep].token is not None
            ]
            if bad_deps:
                blocked.append((node.local_line, [nodes[d].local_line for d in bad_deps]))
                continue
            if all(dep in scheduled or can_emit_non_blt(dep) for dep in producers[node.ridx]):
                chosen = node
                break
        if chosen is None:
            preview = "; ".join(f"{line}<-{deps}" for line, deps in blocked[:8])
            raise RuntimeError(f"{name}: no ready PTX {tok} for SASS line {sass_line}; blocked {preview}")
        emit(chosen.ridx, f"sass-{target_pos}")
        map_rows.append(
            f"{name}\t{target_pos}\t{tok}\t{sass_line}\t{chosen.local_line}\t"
            f"{chosen.op}\t{sass_text}\t{chosen.text}"
        )
        target_pos += 1

    for node in nodes:
        if node.ridx not in scheduled:
            emit(node.ridx, "drain")

    return emitted, map_rows, notes


def extract_blt_from_ptx_lines(lines: list[str], probe) -> str:
    seq = []
    for line in lines:
        parsed = probe.ptx_op_from_line(line)
        if parsed is None:
            continue
        tok = ptx_token(parsed[0])
        if tok:
            seq.append(tok)
    return "".join(seq)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--src-ptx", type=Path, required=True)
    parser.add_argument("--sass", type=Path, required=True)
    parser.add_argument("--function-needle", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    probe = load_probe(repo)
    full_lines = args.src_ptx.read_text(errors="replace").splitlines()
    sass_lines = args.sass.read_text(errors="replace").splitlines()
    fn_start, fn_end = find_function(full_lines, args.function_needle)
    fn_lines = full_lines[fn_start : fn_end + 1]

    specs = [
        RegionSpec("front", 284, 2123, 65, 926),
        RegionSpec("back", 2161, 3752, 951, 1694),
    ]

    rewritten_fn: list[str] = []
    cursor = 1
    all_maps = ["region\ttarget_idx\ttok\tsass_line\tptx_local_line\tptx_op\tsass_text\tptx_text"]
    all_notes = ["region\tkind\tptx_local_line\tptx_op\treason\tptx_text"]
    summary = []
    for spec in specs:
        rewritten_fn.extend(fn_lines[cursor - 1 : spec.ptx_start - 1])
        nodes = parse_region(probe, fn_lines, spec.ptx_start, spec.ptx_end)
        target = sass_tokens(sass_lines, spec.sass_start, spec.sass_end)
        emitted, rows, notes = schedule_region(spec.name, nodes, target)
        region_lines = ["// manual_sass_replay begin " + spec.name]
        region_lines.extend("    " + node.text if not node.text.endswith(":") else node.text for node in emitted)
        region_lines.append("// manual_sass_replay end " + spec.name)
        rewritten_fn.extend(region_lines)
        all_maps.extend(rows)
        all_notes.extend(notes)

        src_seq = extract_blt_from_ptx_lines(fn_lines[spec.ptx_start - 1 : spec.ptx_end], probe)
        out_seq = extract_blt_from_ptx_lines(region_lines, probe)
        sass_seq = "".join(tok for _, tok, _ in target)
        for label, seq in (("src", src_seq), ("out", out_seq), ("sass", sass_seq)):
            total_t, gt4, tail = tensor_tail_after4(seq)
            summary.append(
                f"{spec.name}\t{label}\tlen={len(seq)}\tL={seq.count('L')}\tT={seq.count('T')}\t"
                f"B={seq.count('B')}\tmatch_sass={seq == sass_seq}\tT={total_t}\t"
                f"T_run_gt4_total={gt4}\tT_tail_after4={tail}\tT_run_hist={token_hist(seq)}"
            )

        cursor = spec.ptx_end + 1
    rewritten_fn.extend(fn_lines[cursor - 1 :])

    out_full = full_lines[:fn_start] + rewritten_fn + full_lines[fn_end + 1 :]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manual_sass_replay.ptx").write_text("\n".join(out_full) + "\n")
    (args.out_dir / "manual_sass_replay_target.ptx").write_text("\n".join(rewritten_fn) + "\n")
    (args.out_dir / "manual_sass_replay_map.tsv").write_text("\n".join(all_maps) + "\n")
    (args.out_dir / "manual_sass_replay_auto_producers.tsv").write_text("\n".join(all_notes) + "\n")
    (args.out_dir / "manual_sass_replay_summary.txt").write_text("\n".join(summary) + "\n")
    print("\n".join(summary))
    print(f"wrote {args.out_dir / 'manual_sass_replay.ptx'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
