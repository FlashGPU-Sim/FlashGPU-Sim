#!/usr/bin/env python3
"""Prototype PTX schedulers and compare them with SASS order.

This is an offline analysis tool.  It does not modify simulator behavior.

The PTX side uses static source order filtered by gpgpu_inst_stats line numbers
when a stats file is provided.  That keeps the sequence small and focused on
instructions observed by the simulator, but it is not a dynamic instruction
trace.
"""

from __future__ import annotations

import argparse
import random
import math
import re
from difflib import SequenceMatcher
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_PTX = (
    "test/run/SIM_FA2_SENSITIVITY_H1D128B1S256_TENSORINIT8_20260616_141113/"
    "work/000_sensitivity_H1D128FullB1S256_baseline_fwd/"
    "fa2_fwd_fp16_test.sm_90a.ptx"
)
DEFAULT_STATS = (
    "test/run/SIM_FA2_SENSITIVITY_H1D128B1S256_TENSORINIT8_20260616_141113/"
    "work/000_sensitivity_H1D128FullB1S256_baseline_fwd/gpgpu_inst_stats.txt"
)
DEFAULT_SASS = "test/run/FA2_SENSITIVITY_OPCODE_CHECK/baseline.sass"

WINDOW_FIELDS = (
    "tensor",
    "ldmatrix",
    "sfu",
    "fp32",
    "shfl",
    "mem",
    "int",
    "control",
    "other",
)

SASS_GUIDE_FILTERS = {
    "all": set(WINDOW_FIELDS),
    "compute": {"tensor", "ldmatrix", "sfu", "fp32", "shfl", "mem"},
    "major": {"tensor", "ldmatrix", "sfu", "fp32", "shfl"},
    "mma-ldmatrix": {"tensor", "ldmatrix"},
    "tensor-softmax": {"tensor", "ldmatrix", "sfu", "fp32", "shfl"},
}

CLASS_TOKENS = {
    "tensor": "T",
    "ldmatrix": "L",
    "sfu": "S",
    "fp32": "F",
    "shfl": "H",
    "mem": "M",
    "int": "I",
    "control": "C",
    "other": ".",
}


def class_token(cls: str) -> str:
    return CLASS_TOKENS.get(cls, "?")


@dataclass(frozen=True)
class Inst:
    idx: int
    line: int
    text: str
    op: str
    cls: str
    defs: frozenset[str]
    uses: frozenset[str]
    count: int = 1


@dataclass(frozen=True)
class ScheduleResult:
    insts: list[Inst]
    chunks: int
    edges_checked: int
    max_chunk_size: int
    estimated_issue_cycles: int = 0


@dataclass(frozen=True)
class IssueEstimate:
    cycles: int
    chunks: int
    edges_checked: int
    max_chunk_size: int


@dataclass(frozen=True)
class DepEdge:
    src: int
    dst: int
    latency: int
    kind: str


@dataclass(frozen=True)
class RoleSignature:
    token: str
    rank: int
    stage: int
    chain: int
    pred_ld_count: int
    pred_ld_fanout: tuple[int, ...]
    pred_ld_stage: tuple[int, ...]
    pred_tensor_count: int
    succ_tensor_count: int
    succ_tensor_stage: tuple[int, ...]
    succ_tensor_chain: tuple[int, ...]


@dataclass(frozen=True)
class GuideItem:
    token: str
    cls: str
    rank: int
    line: int
    text: str
    sig: RoleSignature


@dataclass(frozen=True)
class CostModel:
    height: float
    age: float
    switch: float
    recent_same: float
    bigram: float
    trigram: float
    fourgram: float
    target_align: float
    target_lookahead: int
    target_dist: float
    target_dist_window: int
    alt_repeat: float
    class_bias: dict[str, float]
    sass_bigram_log: dict[str, float]
    sass_trigram_log: dict[str, float]
    sass_fourgram_log: dict[str, float]
    target_sequence: str
    target_items: tuple[GuideItem, ...] = ()
    target_identity: float = 0.0

    def summary(self) -> str:
        class_bits = ",".join(f"{key}:{self.class_bias.get(key, 0.0):.2f}" for key in WINDOW_FIELDS)
        return (
            f"height={self.height:.3f} age={self.age:.3f} switch={self.switch:.3f} "
            f"recent_same={self.recent_same:.3f} bigram={self.bigram:.3f} "
            f"trigram={self.trigram:.3f} fourgram={self.fourgram:.3f} "
            f"target_align={self.target_align:.3f} lookahead={self.target_lookahead} "
            f"target_dist={self.target_dist:.3f} dist_window={self.target_dist_window} "
            f"identity={self.target_identity:.3f} alt_repeat={self.alt_repeat:.3f} "
            f"class_bias={{{class_bits}}}"
        )


def strip_predicate(text: str) -> tuple[str, str | None]:
    stripped = text.strip()
    pred = None
    if stripped.startswith("@"):
        pieces = stripped.split(None, 1)
        if len(pieces) == 2:
            pred = pieces[0].lstrip("@!")
            stripped = pieces[1].strip()
    return stripped, pred


def split_top_operands(operand_text: str) -> list[str]:
    out: list[str] = []
    start = 0
    depth = 0
    pairs = {"{": "}", "[": "]", "(": ")"}
    closers = set(pairs.values())
    for i, ch in enumerate(operand_text):
        if ch in pairs:
            depth += 1
        elif ch in closers and depth > 0:
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(operand_text[start:i].strip())
            start = i + 1
    tail = operand_text[start:].strip()
    if tail:
        out.append(tail)
    return out


REG_RE = re.compile(r"%[A-Za-z_][A-Za-z0-9_]*")


def regs_in(text: str) -> set[str]:
    return set(REG_RE.findall(text))


def ptx_op_from_line(line: str) -> tuple[str, str, str | None] | None:
    text = line.split("//", 1)[0].strip()
    if not text or text.startswith(".") or text.endswith(":"):
        return None
    if ";" not in text:
        return None
    text = text.split(";", 1)[0].strip()
    if text.startswith("{"):
        text = text[1:].strip()
    if text.endswith("}"):
        text = text[:-1].strip()
    text, pred = strip_predicate(text)
    match = re.match(r"([A-Za-z0-9_.]+)\b\s*(.*)$", text)
    if not match:
        return None
    op = match.group(1)
    if op in {"{", "}"}:
        return None
    return op, match.group(2).strip(), pred


def is_ptx_label(line: str) -> bool:
    text = line.split("//", 1)[0].strip()
    return bool(text.endswith(":") and not text.startswith(".") and re.match(r"[$A-Za-z_][\w$.-]*:$", text))


def is_control_or_fence(op: str) -> bool:
    return (
        op.startswith("bra")
        or op.startswith("ret")
        or op in {"exit", "trap"}
        or op.startswith("bar.")
        or op.startswith("membar")
        or op.startswith("cp.async.wait")
        or op.startswith("cp.async.commit")
        or op.startswith("wgmma.commit")
        or op.startswith("wgmma.wait")
        or op.startswith("wgmma.fence")
        or op.startswith("fence")
    )


def ptx_class(op: str) -> str:
    if op.startswith("mma.") or op.startswith("wgmma.") and ".mma" in op:
        return "tensor"
    if op.startswith("ldmatrix"):
        return "ldmatrix"
    if op.startswith("ex2.") or op.startswith("lg2.") or op.startswith("rcp.") or op.startswith("tanh."):
        return "sfu"
    if (
        op.startswith("fma.")
        or op.startswith("mad.")
        or op.startswith("mul.ftz.f32")
        or op.startswith("add.ftz.f32")
        or op.startswith("sub.ftz.f32")
        or op.startswith("max.ftz.f32")
        or op.startswith("min.ftz.f32")
        or op.startswith("cvt.")
        and ".f32" in op
    ):
        return "fp32"
    if op.startswith("shfl."):
        return "shfl"
    if op.startswith("cp.async") or op.startswith("stmatrix"):
        return "mem"
    if op.startswith(("ld.", "st.", "atom.", "red.", "prefetch.")):
        return "mem"
    if is_control_or_fence(op):
        return "control"
    if op.startswith(("add.", "sub.", "mul.", "mad.", "shl.", "shr.", "and.", "or.", "xor.", "setp.", "selp.", "mov.", "cvt.")):
        return "int"
    return "other"


def extract_ptx_defs_uses(op: str, operands_text: str, pred: str | None) -> tuple[set[str], set[str]]:
    operands = split_top_operands(operands_text)
    defs: set[str] = set()
    uses: set[str] = set()
    if pred:
        uses.add("%" + pred if not pred.startswith("%") else pred)

    no_def = (
        op.startswith("st.")
        or op.startswith("cp.async")
        or op.startswith("bar.")
        or op.startswith("membar")
        or op.startswith("bra")
        or op.startswith("ret")
        or op in {"exit", "trap"}
        or op.startswith("prefetch")
    )
    if not operands:
        return defs, uses

    if no_def:
        for operand in operands:
            uses.update(regs_in(operand))
        return defs, uses

    if op.startswith("setp.") and len(operands) >= 2:
        defs.update(regs_in(operands[0]))
        for operand in operands[1:]:
            uses.update(regs_in(operand))
        return defs, uses

    defs.update(regs_in(operands[0]))
    for operand in operands[1:]:
        uses.update(regs_in(operand))
    return defs, uses


def read_stats(path: Path | None) -> dict[int, int]:
    if path is None:
        return {}
    counts: dict[int, int] = {}
    for line in path.read_text(errors="replace").splitlines():
        match = re.match(r".*\.ptx\s+(\d+)\s*:\s*(\d+)\b", line)
        if match:
            counts[int(match.group(1))] = int(match.group(2))
    return counts


PTX_FUNCTION_RE = re.compile(r"\s*(?:\.visible\s+)?\.(?:entry|func)\s+(\S+)\s*\(")


def list_ptx_functions(path: Path) -> list[str]:
    funcs = []
    for line in path.read_text(errors="replace").splitlines():
        match = PTX_FUNCTION_RE.match(line)
        if match:
            funcs.append(match.group(1))
    return funcs


def parse_ptx(path: Path, stats_path: Path | None, function_regex: str | None = None) -> list[Inst]:
    stats = read_stats(stats_path)
    selected_lines = set(stats) if stats else None
    function_pattern = re.compile(function_regex) if function_regex else None
    capture_function = function_pattern is None
    insts: list[Inst] = []
    for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        function_match = PTX_FUNCTION_RE.match(line)
        if function_match:
            capture_function = function_pattern is None or bool(function_pattern.search(function_match.group(1)))
        if not capture_function:
            continue
        if is_ptx_label(line):
            insts.append(
                Inst(
                    idx=len(insts),
                    line=lineno,
                    text=line.strip(),
                    op="label",
                    cls="boundary",
                    defs=frozenset(),
                    uses=frozenset(),
                    count=0,
                )
            )
            continue
        if selected_lines is not None and lineno not in selected_lines:
            continue
        parsed = ptx_op_from_line(line)
        if parsed is None:
            continue
        op, operands, pred = parsed
        defs, uses = extract_ptx_defs_uses(op, operands, pred)
        insts.append(
            Inst(
                idx=len(insts),
                line=lineno,
                text=line.strip(),
                op=op,
                cls=ptx_class(op),
                defs=frozenset(defs),
                uses=frozenset(uses),
                count=stats.get(lineno, 1),
            )
        )
    return insts


def sass_class(op: str) -> str:
    base = op.upper()
    if base.startswith(("HMMA", "IMMA", "DMMA", "WGMMA", "MMA")):
        return "tensor"
    if base.startswith("LDSM"):
        return "ldmatrix"
    if base.startswith(("MUFU", "RRO")):
        return "sfu"
    if base.startswith(("FFMA", "FMUL", "FADD", "FMNMX", "FSET", "FSETP", "F2F")):
        return "fp32"
    if base.startswith("SHFL"):
        return "shfl"
    if base.startswith(("LDG", "LDC", "ULDC", "LDS", "STS", "STG", "LDGSTS", "LD ", "ST ")):
        return "mem"
    if base.startswith(("BRA", "EXIT", "RET", "BAR", "DEPBAR", "LDGDEPBAR", "NOP", "WARPSYNC", "BSYNC")):
        return "control"
    if base.startswith(
        (
            "I",
            "UI",
            "UIMAD",
            "IMAD",
            "LOP",
            "PLOP",
            "MOV",
            "S2R",
            "S2UR",
            "CS2R",
            "LEA",
            "SHF",
            "PRMT",
            "SEL",
            "P2R",
            "R2P",
            "VOTE",
        )
    ):
        return "int"
    return "other"


SASS_R_REG_RE = re.compile(r"\bRZ\b|\bR\d+\b", re.IGNORECASE)


def sass_r_regs_in(text: str) -> set[str]:
    regs = set()
    for match in SASS_R_REG_RE.finditer(text):
        reg = match.group(0).upper()
        if reg != "RZ":
            regs.add(reg)
    return regs


def expand_sass_r_base(reg: str, width: int) -> set[str]:
    reg = reg.upper()
    if reg == "RZ":
        return set()
    match = re.fullmatch(r"R(\d+)", reg)
    if not match:
        return {reg}
    base = int(match.group(1))
    return {f"R{base + offset}" for offset in range(max(1, width))}


def sass_ldmatrix_width(op: str) -> int:
    pieces = op.upper().split(".")
    for piece in reversed(pieces):
        if piece.isdigit():
            return int(piece)
    return 1


def sass_tensor_dest_width(op: str) -> int:
    base = op.upper()
    if base.startswith(("HMMA", "IMMA", "DMMA", "MMA")):
        return 4
    return 1


def extract_sass_defs_uses(op: str, operands_text: str, pred: str | None) -> tuple[set[str], set[str]]:
    operands = split_top_operands(operands_text)
    defs: set[str] = set()
    uses: set[str] = set()
    if pred:
        # Predicates are not part of the LDSM/HMMA identity model, but keeping
        # a lightweight token prevents predicated instructions from appearing
        # completely independent if we extend the parser later.
        uses.add(pred.lstrip("@!"))
    if not operands:
        return defs, uses

    cls = sass_class(op)
    if cls == "ldmatrix":
        dest_regs = sass_r_regs_in(operands[0])
        if len(dest_regs) == 1:
            defs.update(expand_sass_r_base(next(iter(dest_regs)), sass_ldmatrix_width(op)))
        else:
            defs.update(dest_regs)
        for operand in operands[1:]:
            uses.update(sass_r_regs_in(operand))
        return defs, uses

    if cls == "tensor":
        dest_regs = sass_r_regs_in(operands[0])
        if len(dest_regs) == 1:
            defs.update(expand_sass_r_base(next(iter(dest_regs)), sass_tensor_dest_width(op)))
        else:
            defs.update(dest_regs)
        for operand in operands[1:]:
            uses.update(sass_r_regs_in(operand))
        return defs, uses

    return defs, uses


def list_sass_functions(path: Path) -> list[str]:
    funcs = []
    for line in path.read_text(errors="replace").splitlines():
        match = re.search(r"Function\s*:\s*(\S+)", line)
        if match:
            funcs.append(match.group(1))
    return funcs


def parse_sass(path: Path, function_regex: str | None) -> list[Inst]:
    pattern = re.compile(function_regex) if function_regex else None
    capture = pattern is None
    seen_target = False
    insts: list[Inst] = []
    for raw in path.read_text(errors="replace").splitlines():
        func = re.search(r"Function\s*:\s*(\S+)", raw)
        if func:
            if pattern is None:
                capture = not seen_target
                seen_target = True
            else:
                capture = bool(pattern.search(func.group(1)))
                seen_target = seen_target or capture
            continue
        if not capture:
            continue
        if raw.startswith("Fatbin "):
            capture = False
            continue
        match = re.match(r"\s*/\*([0-9a-fA-F]+)\*/\s+(?:(@\S+)\s+)?([A-Z][A-Z0-9_.]*)\b(.*?);", raw)
        if not match:
            continue
        op = match.group(3)
        defs, uses = extract_sass_defs_uses(op, match.group(4).strip(), match.group(2))
        insts.append(
            Inst(
                idx=len(insts),
                line=int(match.group(1), 16),
                text=raw.strip(),
                op=op,
                cls=sass_class(op),
                defs=frozenset(defs),
                uses=frozenset(uses),
            )
        )
    return insts


LATENCY = {
    "tensor": 32,
    "sfu": 16,
    "ldmatrix": 8,
    "mem": 8,
    "fp32": 4,
    "shfl": 4,
    "int": 1,
    "control": 1,
    "other": 1,
}


PIPE_LATENCY = {
    "tensor": 32,
    "sfu": 28,
    "ldmatrix": 8,
    "mem": 8,
    "fp32": 4,
    "shfl": 4,
    "int": 4,
    "control": 1,
    "other": 1,
}

PIPE_INITIATION = {
    "tensor": 8,
    "sfu": 8,
    "ldmatrix": 1,
    "mem": 1,
    "fp32": 1,
    "shfl": 1,
    "int": 1,
    "control": 1,
    "other": 1,
}

PIPE_KIND = {
    "tensor": "tensor",
    "sfu": "sfu",
    "ldmatrix": "ldst",
    "mem": "ldst",
    "fp32": "fp32",
    "shfl": "xbar",
    "int": "int",
    "control": "control",
    "other": "other",
}

PIPE_SWITCH_BONUS = {
    # One-step class transition preferences observed in H100 SASS.  This is
    # intentionally much weaker than the n-gram cost model: it only nudges
    # choices among instructions that are already ready, or nearly ready when
    # --pipe-ready-slack is used.
    "T": {"L": 16.0, "F": 12.0, "I": 7.0, "S": 5.0, "T": 3.0},
    "L": {"T": 18.0, "F": 6.0, "I": 5.0, "S": 4.0, "L": 2.0},
    "F": {"T": 10.0, "I": 10.0, "S": 8.0, "M": 4.0, "L": 2.0, "F": 2.0},
    "S": {"F": 18.0, "I": 8.0, "T": 7.0, "L": 2.0},
    "M": {"I": 16.0, "F": 8.0, ".": 4.0, "M": 2.0},
    "I": {"M": 12.0, "F": 10.0, "T": 6.0, ".": 5.0, "S": 4.0, "L": 2.0, "I": 1.0},
    ".": {"I": 12.0, "F": 7.0, "M": 4.0, "T": 2.0},
    "C": {"I": 8.0, ".": 4.0, "T": 2.0},
    "H": {"F": 10.0, "I": 2.0, "H": 1.0},
}


def inst_latency(inst: Inst) -> int:
    if inst.op.startswith("cp.async"):
        return 1
    return PIPE_LATENCY.get(inst.cls, 1)


def inst_initiation(inst: Inst) -> int:
    if inst.op.startswith("cp.async"):
        return 1
    return PIPE_INITIATION.get(inst.cls, 1)


def inst_pipe(inst: Inst) -> str:
    return PIPE_KIND.get(inst.cls, "other")


def is_barrier_inst(inst: Inst) -> bool:
    return inst.op.startswith("bar.")


def is_barrier_sensitive(inst: Inst) -> bool:
    return inst.cls in {"ldmatrix", "mem", "control", "boundary"}


def pipe_model_summary() -> str:
    parts = []
    for cls in WINDOW_FIELDS:
        parts.append(
            f"{cls}:{PIPE_KIND.get(cls, 'other')}/lat{PIPE_LATENCY.get(cls, 1)}/ii{PIPE_INITIATION.get(cls, 1)}"
        )
    return " ".join(parts)


def pipe_policy_summary(policy: str, ready_slack: int) -> str:
    if policy == "pipe-latency-tl":
        return f"TL/LT adjacency bonus only; ready_slack={ready_slack}"
    if policy == "pipe-latency-switch":
        return (
            "one-step SASS-like transition bonus with same-class run penalty; "
            f"ready_slack={ready_slack}"
        )
    if policy == "pipe-latency-defuse":
        return (
            "pipe-latency-switch plus real ldmatrix->tensor RAW consumer/producers; "
            f"ready_slack={ready_slack}"
        )
    if policy == "pipe-latency-stage":
        return (
            "GEMM stage-preload policy from tensor accumulator chains and "
            f"ldmatrix consumer stages; ready_slack={ready_slack}"
        )
    if policy == "pipe-latency-antirun":
        return (
            "pipe-latency-stage plus hard same-class run penalty and strong "
            f"LDSM/HMMA alternation bonus; ready_slack={ready_slack}"
        )
    if policy == "pipe-latency-inventory":
        return (
            "pipe-latency-antirun plus remaining-class inventory balancing, "
            "so scarce F/L breakers are spread across long tensor regions; "
            f"ready_slack={ready_slack}"
        )
    return f"earliest pipe-ready issue; ready_slack={ready_slack}"


def build_dependency_graph(
    chunk: list[Inst],
    relax_barrier_reg: bool = False,
    ignore_anti_deps: bool = False,
    relax_ldmatrix_order: bool = False,
) -> tuple[list[set[int]], list[int], list[DepEdge]]:
    n = len(chunk)
    succ: list[set[int]] = [set() for _ in range(n)]
    indeg = [0 for _ in range(n)]
    edge_map: dict[tuple[int, int], DepEdge] = {}

    def add_edge(a: int, b: int, latency: int = 0, kind: str = "order") -> None:
        if a == b:
            return
        old = edge_map.get((a, b))
        if old is not None:
            if latency > old.latency or (latency == old.latency and old.kind != "raw" and kind == "raw"):
                edge_map[(a, b)] = DepEdge(a, b, latency, kind)
            return
        succ[a].add(b)
        indeg[b] += 1
        edge_map[(a, b)] = DepEdge(a, b, latency, kind)

    last_def: dict[str, int] = {}
    last_uses: dict[str, set[int]] = defaultdict(set)
    last_mem: int | None = None
    last_barrier: int | None = None
    barrier_sensitive_since_last: list[int] = []
    for i, inst in enumerate(chunk):
        if relax_barrier_reg and is_barrier_inst(inst):
            for prior in barrier_sensitive_since_last:
                add_edge(prior, i, 0, "barrier-before")
            last_barrier = i
            barrier_sensitive_since_last = []

        for reg in inst.uses:
            if reg in last_def:
                producer = last_def[reg]
                add_edge(producer, i, inst_latency(chunk[producer]), "raw")
            last_uses[reg].add(i)
        for reg in inst.defs:
            if not ignore_anti_deps and reg in last_def:
                add_edge(last_def[reg], i, 0, "waw")
            if not ignore_anti_deps:
                for prior_use in last_uses.get(reg, set()):
                    add_edge(prior_use, i, 0, "war")
            last_def[reg] = i
            last_uses[reg].clear()
        if inst.cls in {"mem", "ldmatrix"}:
            if last_mem is not None:
                if not relax_ldmatrix_order or inst.cls != "ldmatrix" or chunk[last_mem].cls != "ldmatrix":
                    add_edge(last_mem, i, 0, "mem-order")
            if not relax_ldmatrix_order or inst.cls != "ldmatrix":
                last_mem = i
        if relax_barrier_reg and not is_barrier_inst(inst) and is_barrier_sensitive(inst):
            if last_barrier is not None:
                add_edge(last_barrier, i, 0, "barrier-after")
            barrier_sensitive_since_last.append(i)

    return succ, indeg, list(edge_map.values())


def validate_dependency_order(order: list[int], edges: list[DepEdge]) -> None:
    pos = {node: i for i, node in enumerate(order)}
    for edge in edges:
        if pos[edge.src] >= pos[edge.dst]:
            raise RuntimeError(
                f"scheduler violated dependency edge {edge.src}->{edge.dst} ({edge.kind}): "
                f"positions {pos[edge.src]} >= {pos[edge.dst]}"
            )


def compute_role_signatures(
    insts: list[Inst],
    relax_barrier_reg: bool = False,
    ignore_anti_deps: bool = False,
    relax_ldmatrix_order: bool = False,
) -> dict[int, RoleSignature]:
    _, _, edges = build_dependency_graph(
        insts, relax_barrier_reg, ignore_anti_deps, relax_ldmatrix_order
    )

    class_rank: dict[int, int] = {}
    class_seen: Counter[str] = Counter()
    for i, inst in enumerate(insts):
        token = class_token(inst.cls)
        class_rank[i] = class_seen[token]
        class_seen[token] += 1

    ldmatrix_tensor_succ: dict[int, list[int]] = defaultdict(list)
    tensor_ldmatrix_pred: dict[int, list[int]] = defaultdict(list)
    tensor_tensor_pred: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        if edge.kind != "raw":
            continue
        if insts[edge.src].cls == "ldmatrix" and insts[edge.dst].cls == "tensor":
            ldmatrix_tensor_succ[edge.src].append(edge.dst)
            tensor_ldmatrix_pred[edge.dst].append(edge.src)
        if insts[edge.src].cls == "tensor" and insts[edge.dst].cls == "tensor":
            tensor_tensor_pred[edge.dst].append(edge.src)

    tensor_stage: dict[int, int] = {}
    tensor_chain: dict[int, int] = {}
    next_chain = 0
    for i, inst in enumerate(insts):
        if inst.cls != "tensor":
            continue
        pred_stages = [tensor_stage[src] for src in tensor_tensor_pred.get(i, []) if src in tensor_stage]
        tensor_stage[i] = max(pred_stages) + 1 if pred_stages else 0
        pred_chains = [tensor_chain[src] for src in tensor_tensor_pred.get(i, []) if src in tensor_chain]
        if pred_chains:
            tensor_chain[i] = min(pred_chains)
        else:
            tensor_chain[i] = next_chain
            next_chain += 1

    ldmatrix_stage: dict[int, int] = {}
    for src, consumers in ldmatrix_tensor_succ.items():
        stages = [tensor_stage[dst] for dst in consumers if dst in tensor_stage]
        if stages:
            ldmatrix_stage[src] = min(stages)

    out: dict[int, RoleSignature] = {}
    for i, inst in enumerate(insts):
        token = class_token(inst.cls)
        pred_ld = sorted(tensor_ldmatrix_pred.get(i, []))
        succ_tensor = sorted(ldmatrix_tensor_succ.get(i, []))
        pred_ld_fanout = tuple(sorted(len(ldmatrix_tensor_succ.get(src, [])) for src in pred_ld))
        pred_ld_stage = tuple(sorted(ldmatrix_stage.get(src, -1) for src in pred_ld))
        succ_tensor_stage = tuple(sorted(tensor_stage.get(dst, -1) for dst in succ_tensor))
        succ_tensor_chain = tuple(sorted(tensor_chain.get(dst, -1) for dst in succ_tensor))
        out[i] = RoleSignature(
            token=token,
            rank=class_rank[i],
            stage=tensor_stage.get(i, ldmatrix_stage.get(i, -1)),
            chain=tensor_chain.get(i, -1),
            pred_ld_count=len(pred_ld),
            pred_ld_fanout=pred_ld_fanout,
            pred_ld_stage=pred_ld_stage,
            pred_tensor_count=len(tensor_tensor_pred.get(i, [])),
            succ_tensor_count=len(succ_tensor),
            succ_tensor_stage=succ_tensor_stage,
            succ_tensor_chain=succ_tensor_chain,
        )
    return out


def tuple_overlap_score(a: tuple[int, ...], b: tuple[int, ...], weight: float) -> float:
    if not a and not b:
        return weight
    if a == b:
        return weight
    ca = Counter(a)
    cb = Counter(b)
    keys = set(ca) | set(cb)
    if not keys:
        return 0.0
    inter = sum(min(ca[key], cb[key]) for key in keys)
    union = sum(max(ca[key], cb[key]) for key in keys)
    return weight * inter / union if union else 0.0


def role_signature_similarity(candidate: RoleSignature, guide: RoleSignature) -> float:
    if candidate.token != guide.token:
        return -1000.0

    score = 4.0
    if candidate.stage >= 0 and guide.stage >= 0:
        if candidate.stage == guide.stage:
            score += 4.0
        else:
            score -= min(4.0, abs(candidate.stage - guide.stage))

    if candidate.token == "L":
        score += 4.0 if candidate.succ_tensor_count == guide.succ_tensor_count else -2.0 * abs(
            candidate.succ_tensor_count - guide.succ_tensor_count
        )
        score += tuple_overlap_score(candidate.succ_tensor_stage, guide.succ_tensor_stage, 4.0)
        score += tuple_overlap_score(candidate.succ_tensor_chain, guide.succ_tensor_chain, 10.0)
    elif candidate.token == "T":
        if candidate.chain >= 0 and guide.chain >= 0:
            if candidate.chain == guide.chain:
                score += 12.0
            else:
                score -= min(8.0, 0.5 * abs(candidate.chain - guide.chain))
        score += 4.0 if candidate.pred_ld_count == guide.pred_ld_count else -2.0 * abs(
            candidate.pred_ld_count - guide.pred_ld_count
        )
        score += 2.0 if candidate.pred_tensor_count == guide.pred_tensor_count else -1.0 * abs(
            candidate.pred_tensor_count - guide.pred_tensor_count
        )
        score += tuple_overlap_score(candidate.pred_ld_fanout, guide.pred_ld_fanout, 3.0)
        score += tuple_overlap_score(candidate.pred_ld_stage, guide.pred_ld_stage, 3.0)

    # Rank is only a weak prior: ptxas can reorder the logical tiles, but when
    # two candidates have the same dataflow role the occurrence rank often
    # breaks ties in the right direction.
    score -= 0.02 * abs(candidate.rank - guide.rank)
    return score


def target_item_alignment_score(
    cost_model: CostModel,
    target_cursor: int,
    token: str,
    sig: RoleSignature | None,
) -> tuple[float, int]:
    if cost_model.target_align == 0.0 or not cost_model.target_items:
        return target_alignment_score(cost_model, target_cursor, token)
    limit = min(len(cost_model.target_items), target_cursor + max(1, cost_model.target_lookahead))
    best_score: float | None = None
    best_cursor = target_cursor
    for idx in range(target_cursor, limit):
        item = cost_model.target_items[idx]
        if item.token != token:
            continue
        offset = idx - target_cursor
        identity = max(0.0, role_signature_similarity(sig, item.sig)) if sig is not None else 0.0
        # SASS guidance first fixes the class stream.  Def-use identity is only
        # a tie-break among candidates competing for the same nearby guide slot;
        # otherwise a high identity score can incorrectly jump over the next
        # required LDSM/HMMA token and reproduce the class-only failure mode in
        # a subtler form.
        score = -10.0 * float(offset) + cost_model.target_identity * identity
        if best_score is None or score > best_score:
            best_score = score
            best_cursor = idx + 1
    if best_score is None:
        return -float(cost_model.target_lookahead), target_cursor
    return best_score, best_cursor


def build_guide_items(
    insts: list[Inst],
    keep_classes: set[str],
    relax_barrier_reg: bool = False,
    ignore_anti_deps: bool = False,
    relax_ldmatrix_order: bool = False,
) -> tuple[GuideItem, ...]:
    sigs = compute_role_signatures(
        insts, relax_barrier_reg, ignore_anti_deps, relax_ldmatrix_order
    )
    items: list[GuideItem] = []
    for i, inst in enumerate(insts):
        if inst.cls not in keep_classes:
            continue
        sig = sigs[i]
        items.append(
            GuideItem(
                token=class_token(inst.cls),
                cls=inst.cls,
                rank=sig.rank,
                line=inst.line,
                text=inst.text,
                sig=sig,
            )
        )
    return tuple(items)


def role_signature_brief(sig: RoleSignature) -> str:
    return (
        f"{sig.token}{sig.rank}:stage={sig.stage} chain={sig.chain} "
        f"pld={sig.pred_ld_count}/{sig.pred_ld_fanout}/{sig.pred_ld_stage} "
        f"sT={sig.succ_tensor_count}/{sig.succ_tensor_stage}/{sig.succ_tensor_chain}"
    )


def role_alignment_metrics(candidate: list[Inst], sass: list[Inst], keep_classes: set[str]) -> dict[str, float]:
    cand_items = build_guide_items(candidate, keep_classes, relax_ldmatrix_order=True)
    sass_items = build_guide_items(sass, keep_classes, relax_ldmatrix_order=True)
    n = min(len(cand_items), len(sass_items))
    if n == 0:
        return {
            "aligned": 0,
            "token_match": 0.0,
            "avg_role_same_token": 0.0,
            "avg_role_all": 0.0,
            "first_token_mismatch": -1,
        }
    token_matches = 0
    same_token_scores: list[float] = []
    all_scores: list[float] = []
    first_mismatch = -1
    for idx in range(n):
        cand = cand_items[idx]
        guide = sass_items[idx]
        if cand.token == guide.token:
            token_matches += 1
            same_token_scores.append(role_signature_similarity(cand.sig, guide.sig))
        elif first_mismatch < 0:
            first_mismatch = idx
        all_scores.append(role_signature_similarity(cand.sig, guide.sig))
    return {
        "aligned": n,
        "token_match": token_matches / n,
        "avg_role_same_token": sum(same_token_scores) / len(same_token_scores) if same_token_scores else 0.0,
        "avg_role_all": sum(all_scores) / len(all_scores),
        "first_token_mismatch": first_mismatch,
    }


def print_role_alignment_report(name: str, candidate: list[Inst], sass: list[Inst], keep_classes: set[str]) -> None:
    metrics = role_alignment_metrics(candidate, sass, keep_classes)
    print(
        f"  {name:22s} aligned={metrics['aligned']:.0f} "
        f"token_match={metrics['token_match']:.4f} "
        f"avg_role_same_token={metrics['avg_role_same_token']:.2f} "
        f"avg_role_all={metrics['avg_role_all']:.2f} "
        f"first_token_mismatch={metrics['first_token_mismatch']:.0f}"
    )


def target_alignment_score(cost_model: CostModel, target_cursor: int, token: str) -> tuple[float, int]:
    if cost_model.target_align == 0.0 or not cost_model.target_sequence:
        return 0.0, target_cursor
    limit = min(len(cost_model.target_sequence), target_cursor + max(1, cost_model.target_lookahead))
    window = cost_model.target_sequence[target_cursor:limit]
    offset = window.find(token)
    if offset < 0:
        return -float(cost_model.target_lookahead), target_cursor
    return -float(offset), target_cursor + offset + 1


def schedule_chunk(
    chunk: list[Inst],
    policy: str,
    cost_model: CostModel | None,
    target_cursor: int = 0,
    relax_barrier_reg: bool = False,
    ignore_anti_deps: bool = False,
    relax_ldmatrix_order: bool = False,
) -> tuple[list[Inst], int, int]:
    n = len(chunk)
    succ, indeg, edges = build_dependency_graph(
        chunk, relax_barrier_reg, ignore_anti_deps, relax_ldmatrix_order
    )
    pred_by_dst: list[list[int]] = [[] for _ in range(n)]
    for edge in edges:
        pred_by_dst[edge.dst].append(edge.src)

    height = [LATENCY.get(inst.cls, 1) for inst in chunk]
    for i in range(n - 1, -1, -1):
        if succ[i]:
            height[i] = LATENCY.get(chunk[i].cls, 1) + max(height[j] for j in succ[i])

    ready = {i for i in range(n) if indeg[i] == 0}
    emitted: list[int] = []
    last_cls = ""
    recent: deque[str] = deque(maxlen=16)
    cursor = target_cursor
    target_tokens = set(cost_model.target_sequence) if cost_model is not None else set()
    role_sigs = (
        compute_role_signatures(chunk, relax_barrier_reg, ignore_anti_deps, relax_ldmatrix_order)
        if policy == "sass-guided-defuse"
        else {}
    )
    emitted_flag = [False for _ in range(n)]
    while ready:
        current_guide: GuideItem | None = None
        desired_node: int | None = None
        desired_ready_ancestors: set[int] = set()
        if (
            policy == "sass-guided-defuse"
            and cost_model is not None
            and cursor < len(cost_model.target_items)
        ):
            current_guide = cost_model.target_items[cursor]
            best_identity: float | None = None
            for candidate in range(n):
                if emitted_flag[candidate]:
                    continue
                if class_token(chunk[candidate].cls) != current_guide.token:
                    continue
                identity = role_signature_similarity(role_sigs[candidate], current_guide.sig)
                if best_identity is None or identity > best_identity:
                    best_identity = identity
                    desired_node = candidate
            if desired_node is not None and desired_node not in ready:
                stack = list(pred_by_dst[desired_node])
                seen: set[int] = set()
                while stack:
                    pred = stack.pop()
                    if pred in seen or emitted_flag[pred]:
                        continue
                    seen.add(pred)
                    if pred in ready:
                        desired_ready_ancestors.add(pred)
                    else:
                        stack.extend(pred_by_dst[pred])

        def key(i: int) -> tuple[float, float, float, int]:
            inst = chunk[i]
            pipe_bonus = {
                "tensor": 5,
                "sfu": 4,
                "ldmatrix": 3,
                "fp32": 2,
                "shfl": 1,
            }.get(inst.cls, 0)
            if policy == "critical":
                switch_bonus = 4 if inst.cls != last_cls else 0
                return (height[i], switch_bonus, pipe_bonus, -inst.idx)
            if policy == "balanced":
                recent_penalty = sum(1 for cls in recent if cls == inst.cls)
                switch_bonus = 12 if inst.cls != last_cls else -4
                score = height[i] + switch_bonus - 4 * recent_penalty + pipe_bonus
                return (score, height[i], pipe_bonus, -inst.idx)
            if policy == "aggressive-mix":
                recent_penalty = sum(1 for cls in recent if cls == inst.cls)
                switch_bonus = 100 if inst.cls != last_cls else -100
                score = switch_bonus - 15 * recent_penalty + 0.2 * height[i] + pipe_bonus
                return (score, height[i], pipe_bonus, -inst.idx)
            if policy == "cost-model":
                if cost_model is None:
                    raise ValueError("policy=cost-model requires a CostModel")
                recent_same = sum(1 for cls in recent if cls == inst.cls)
                switch = 1.0 if last_cls and inst.cls != last_cls else 0.0
                age = -float(inst.idx)
                bigram = 0.0
                trigram = 0.0
                fourgram = 0.0
                token = class_token(inst.cls)
                if last_cls:
                    bigram = cost_model.sass_bigram_log.get(class_token(last_cls) + token, 0.0)
                if len(recent) >= 2:
                    tri_key = class_token(recent[-2]) + class_token(recent[-1]) + token
                    trigram = cost_model.sass_trigram_log.get(tri_key, 0.0)
                if len(recent) >= 3:
                    four_key = class_token(recent[-3]) + class_token(recent[-2]) + class_token(recent[-1]) + token
                    fourgram = cost_model.sass_fourgram_log.get(four_key, 0.0)
                target_score, _ = target_alignment_score(cost_model, cursor, token)
                dist_score = 0.0
                if cost_model.target_dist and cost_model.target_sequence:
                    dist_end = min(
                        len(cost_model.target_sequence),
                        cursor + max(1, cost_model.target_dist_window),
                    )
                    future = cost_model.target_sequence[cursor:dist_end]
                    if future:
                        dist_score = math.log1p(future.count(token))
                alt_repeat = 0.0
                if len(recent) >= 3:
                    # Penalize continuing a strict A/B/A/B alternation. This
                    # keeps n-gram fitting from degenerating into mechanical
                    # T/F/T/F patterns when SASS has richer local structure.
                    if recent[-1] != recent[-2] and recent[-2] == inst.cls and recent[-3] == recent[-1]:
                        alt_repeat = 1.0
                score = (
                    cost_model.height * height[i]
                    + cost_model.age * age
                    + cost_model.switch * switch
                    + cost_model.recent_same * recent_same
                    + cost_model.bigram * bigram
                    + cost_model.trigram * trigram
                    + cost_model.fourgram * fourgram
                    + cost_model.target_align * target_score
                    + cost_model.target_dist * dist_score
                    + cost_model.alt_repeat * alt_repeat
                    + cost_model.class_bias.get(inst.cls, 0.0)
                )
                return (score, height[i], cost_model.class_bias.get(inst.cls, 0.0), -inst.idx)
            if policy == "sass-guided":
                if cost_model is None:
                    raise ValueError("policy=sass-guided requires a CostModel")
                token = class_token(inst.cls)
                target_score = 0.0
                if token in target_tokens:
                    target_score, _ = target_alignment_score(cost_model, cursor, token)
                recent_same = sum(1 for cls in recent if cls == inst.cls)
                switch = 1.0 if last_cls and inst.cls != last_cls else 0.0
                age = -float(inst.idx)
                score = (
                    cost_model.target_align * target_score
                    + cost_model.height * height[i]
                    + cost_model.age * age
                    + cost_model.switch * switch
                    + cost_model.recent_same * recent_same
                    + cost_model.class_bias.get(inst.cls, 0.0)
                )
                return (score, target_score, height[i], -inst.idx)
            if policy == "sass-guided-defuse":
                if cost_model is None:
                    raise ValueError("policy=sass-guided-defuse requires a CostModel")
                token = class_token(inst.cls)
                target_score = 0.0
                identity = 0.0
                if current_guide is not None and token == current_guide.token:
                    identity = max(0.0, role_signature_similarity(role_sigs[i], current_guide.sig))
                    if i == desired_node:
                        target_score = 100000.0 + cost_model.target_identity * identity
                    else:
                        target_score = 90000.0 + cost_model.target_identity * identity
                elif current_guide is not None and desired_node is not None and i in desired_ready_ancestors:
                    target_score = 50000.0 + 0.001 * height[i]
                elif current_guide is not None and token in target_tokens:
                    target_score = -100000.0
                elif token in target_tokens:
                    target_score, _ = target_alignment_score(cost_model, cursor, token)
                recent_same = sum(1 for cls in recent if cls == inst.cls)
                switch = 1.0 if last_cls and inst.cls != last_cls else 0.0
                age = -float(inst.idx)
                score = (
                    cost_model.target_align * target_score
                    + cost_model.height * height[i]
                    + cost_model.age * age
                    + cost_model.switch * switch
                    + cost_model.recent_same * recent_same
                    + cost_model.class_bias.get(inst.cls, 0.0)
                )
                return (score, target_score, height[i], -inst.idx)
            raise ValueError(f"unknown policy: {policy}")

        pick = max(ready, key=key)
        ready.remove(pick)
        emitted.append(pick)
        emitted_flag[pick] = True
        last_cls = chunk[pick].cls
        recent.append(last_cls)
        if cost_model is not None and policy in {"cost-model", "sass-guided", "sass-guided-defuse"}:
            token = class_token(last_cls)
            if policy == "sass-guided-defuse":
                if (
                    cursor < len(cost_model.target_items)
                    and token == cost_model.target_items[cursor].token
                ):
                    cursor += 1
            elif policy != "sass-guided" or token in target_tokens:
                _, cursor = target_alignment_score(cost_model, cursor, token)
        for dst in succ[pick]:
            indeg[dst] -= 1
            if indeg[dst] == 0:
                ready.add(dst)
    if len(emitted) != n:
        raise RuntimeError("dependency cycle in scheduler chunk")
    validate_dependency_order(emitted, edges)
    return [chunk[i] for i in emitted], len(edges), cursor


def schedule_chunk_pipe_latency(
    chunk: list[Inst],
    policy: str,
    ready_slack: int,
    relax_barrier_reg: bool = False,
    ignore_anti_deps: bool = False,
    relax_ldmatrix_order: bool = False,
) -> tuple[list[Inst], int, int]:
    n = len(chunk)
    succ, indeg, edges = build_dependency_graph(
        chunk, relax_barrier_reg, ignore_anti_deps, relax_ldmatrix_order
    )
    edge_by_src: list[list[DepEdge]] = [[] for _ in range(n)]
    ldmatrix_tensor_succ: dict[int, list[int]] = defaultdict(list)
    tensor_ldmatrix_pred: dict[int, list[int]] = defaultdict(list)
    tensor_tensor_pred: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        edge_by_src[edge.src].append(edge)
        if (
            edge.kind == "raw"
            and chunk[edge.src].cls == "ldmatrix"
            and chunk[edge.dst].cls == "tensor"
        ):
            ldmatrix_tensor_succ[edge.src].append(edge.dst)
            tensor_ldmatrix_pred[edge.dst].append(edge.src)
        if edge.kind == "raw" and chunk[edge.src].cls == "tensor" and chunk[edge.dst].cls == "tensor":
            tensor_tensor_pred[edge.dst].append(edge.src)

    tensor_stage: dict[int, int] = {}
    for i, inst in enumerate(chunk):
        if inst.cls != "tensor":
            continue
        pred_stages = [tensor_stage[src] for src in tensor_tensor_pred.get(i, []) if src in tensor_stage]
        tensor_stage[i] = max(pred_stages) + 1 if pred_stages else 0
    ldmatrix_stage: dict[int, int] = {}
    for src, consumers in ldmatrix_tensor_succ.items():
        stages = [tensor_stage[dst] for dst in consumers if dst in tensor_stage]
        if stages:
            ldmatrix_stage[src] = min(stages)

    height = [inst_latency(inst) for inst in chunk]
    for i in range(n - 1, -1, -1):
        if edge_by_src[i]:
            height[i] = max(edge.latency + height[edge.dst] for edge in edge_by_src[i])

    ready = {i for i in range(n) if indeg[i] == 0}
    dep_ready = [0 for _ in range(n)]
    pipe_ready: dict[str, int] = defaultdict(int)
    warp_issue_ready = 0
    issue_time = [-1 for _ in range(n)]
    emitted_step = [-1 for _ in range(n)]
    emitted: list[int] = []
    recent: deque[str] = deque(maxlen=12)
    remaining_tokens: Counter[str] = Counter(class_token(inst.cls) for inst in chunk)
    last_token = ""
    last_node: int | None = None
    run_token = ""
    run_len = 0
    current_tensor_stage = -1
    max_completion = 0

    while ready:
        issues = {
            i: max(dep_ready[i], pipe_ready[inst_pipe(chunk[i])], warp_issue_ready)
            for i in ready
        }
        min_issue = min(issues.values())

        def key(i: int) -> tuple[int, int, int]:
            inst = chunk[i]
            issue = issues[i]
            return (issue, -height[i], inst.idx)

        if policy == "pipe-latency":
            pick = min(ready, key=key)
        else:
            pool = [i for i in ready if issues[i] <= min_issue + max(0, ready_slack)]

            def local_key(i: int) -> tuple[float, float, int, int]:
                inst = chunk[i]
                token = class_token(inst.cls)
                score = -0.02 * issues[i] + 0.001 * height[i] - 0.0001 * inst.idx
                if policy == "pipe-latency-tl":
                    if (last_token, token) in {("T", "L"), ("L", "T")}:
                        score += 12.0
                elif policy == "pipe-latency-switch":
                    if last_token:
                        score += 0.6 * PIPE_SWITCH_BONUS.get(last_token, {}).get(token, 0.0)
                        if (last_token, token) in {("T", "L"), ("L", "T")}:
                            score += 4.0
                        same_recent = sum(1 for recent_token in recent if recent_token == token)
                        score -= 2.0 * same_recent
                elif policy == "pipe-latency-defuse":
                    if last_token:
                        score += 0.6 * PIPE_SWITCH_BONUS.get(last_token, {}).get(token, 0.0)
                        same_recent = sum(1 for recent_token in recent if recent_token == token)
                        score -= 2.0 * same_recent
                    if inst.cls == "ldmatrix" and i in ldmatrix_tensor_succ:
                        pending_consumers = [
                            dst for dst in ldmatrix_tensor_succ[i] if emitted_step[dst] < 0
                        ]
                        if pending_consumers:
                            fanout = min(len(pending_consumers), 4)
                            nearest = min(abs(chunk[dst].idx - inst.idx) for dst in pending_consumers)
                            score += 3.0 * fanout
                            score += 4.0 / (1.0 + nearest / 8.0)
                            if last_token == "T":
                                score += 1.5
                    if inst.cls == "tensor" and i in tensor_ldmatrix_pred:
                        emitted_producers = [
                            src for src in tensor_ldmatrix_pred[i] if issue_time[src] >= 0
                        ]
                        if emitted_producers:
                            best_age = min(issues[i] - issue_time[src] for src in emitted_producers)
                            score += 6.0 * max(0.0, (48.0 - best_age) / 48.0)
                            producer_step = max(emitted_step[src] for src in emitted_producers)
                            if producer_step >= 0:
                                score += 6.0 * max(0.0, (16.0 - (len(emitted) - producer_step)) / 16.0)
                        if last_node is not None and last_node in tensor_ldmatrix_pred[i]:
                            score += 6.0
                elif policy in {"pipe-latency-stage", "pipe-latency-antirun", "pipe-latency-inventory"}:
                    if last_token:
                        score += 0.6 * PIPE_SWITCH_BONUS.get(last_token, {}).get(token, 0.0)
                        same_recent = sum(1 for recent_token in recent if recent_token == token)
                        score -= 2.0 * same_recent
                    if policy in {"pipe-latency-antirun", "pipe-latency-inventory"} and last_token:
                        run_threshold = {
                            "T": 2,
                            "L": 1,
                            "F": 3,
                            "S": 1,
                            "M": 2,
                            "I": 4,
                            ".": 2,
                        }.get(last_token, 2)
                        if policy == "pipe-latency-inventory" and last_token == "T":
                            remaining_breakers = remaining_tokens["F"] + remaining_tokens["L"]
                            if remaining_breakers > 0:
                                target_gap = min(32, max(2, math.ceil(remaining_tokens["T"] / (remaining_breakers + 1))))
                                run_threshold = max(run_threshold, target_gap)
                        if token == last_token and run_len >= run_threshold:
                            score -= 32.0 * (run_len - run_threshold + 1)
                        elif token != last_token and run_len >= run_threshold:
                            score += min(48.0, 8.0 * (run_len - run_threshold + 1))
                        if (last_token, token) in {("T", "L"), ("L", "T")}:
                            score += 28.0 + 4.0 * min(run_len, 8)
                        if token == last_token:
                            score -= 4.0 * sum(1 for recent_token in recent if recent_token == token)
                    if policy == "pipe-latency-inventory":
                        breaker_tokens = {"F", "L"}
                        remaining_breakers = remaining_tokens["F"] + remaining_tokens["L"]
                        if remaining_tokens["T"] > 0 and remaining_breakers > 0 and last_token == "T":
                            target_gap = min(32, max(2, math.ceil(remaining_tokens["T"] / (remaining_breakers + 1))))
                            if token in breaker_tokens:
                                early = target_gap - run_len
                                if early > 0:
                                    score -= 14.0 * early
                                else:
                                    score += min(36.0, 6.0 * (run_len - target_gap + 1))
                                after_breakers = remaining_breakers - 1
                                after_t = remaining_tokens["T"]
                                if after_breakers == 0 and after_t > target_gap:
                                    score -= min(160.0, 4.0 * (after_t - target_gap))
                            elif token == "T":
                                if run_len < target_gap:
                                    score += min(48.0, 6.0 * (target_gap - run_len))
                                else:
                                    score -= min(32.0, 2.0 * (run_len - target_gap))
                    if inst.cls == "ldmatrix" and i in ldmatrix_stage:
                        pending_consumers = [
                            dst for dst in ldmatrix_tensor_succ[i] if emitted_step[dst] < 0
                        ]
                        if pending_consumers:
                            stage = ldmatrix_stage[i]
                            if current_tensor_stage < 0:
                                score += 8.0 * max(0.0, 1.5 - stage)
                            else:
                                distance = stage - current_tensor_stage
                                if distance <= 0:
                                    score += 6.0
                                elif distance <= 2:
                                    score += 8.0 * (3 - distance)
                                    if last_token == "T":
                                        score += 4.0
                                else:
                                    score -= 1.0 * (distance - 2)
                            nearest = min(abs(chunk[dst].idx - inst.idx) for dst in pending_consumers)
                            score += 2.0 / (1.0 + nearest / 16.0)
                    if inst.cls == "tensor" and i in tensor_stage:
                        stage = tensor_stage[i]
                        if current_tensor_stage >= 0:
                            if stage == current_tensor_stage:
                                score += 4.0
                            elif stage == current_tensor_stage + 1:
                                score += 1.0
                            elif stage > current_tensor_stage + 1:
                                score -= 8.0 * (stage - current_tensor_stage - 1)
                        emitted_producers = [
                            src for src in tensor_ldmatrix_pred.get(i, []) if issue_time[src] >= 0
                        ]
                        if emitted_producers:
                            best_age = min(issues[i] - issue_time[src] for src in emitted_producers)
                            score += 2.0 * max(0.0, (64.0 - best_age) / 64.0)
                else:
                    raise ValueError(f"unknown pipe policy: {policy}")
                return (score, -issues[i], height[i], -inst.idx)

            pick = max(pool, key=local_key)
        ready.remove(pick)
        inst = chunk[pick]
        issue = issues[pick]
        issue_time[pick] = issue
        emitted.append(pick)
        emitted_step[pick] = len(emitted) - 1
        token = class_token(inst.cls)
        remaining_tokens[token] -= 1
        recent.append(token)
        last_token = token
        if token == run_token:
            run_len += 1
        else:
            run_token = token
            run_len = 1
        last_node = pick
        if inst.cls == "tensor" and pick in tensor_stage:
            current_tensor_stage = max(current_tensor_stage, tensor_stage[pick])
        pipe_ready[inst_pipe(inst)] = issue + inst_initiation(inst)
        warp_issue_ready = issue + 1
        max_completion = max(max_completion, issue + inst_latency(inst))

        for edge in edge_by_src[pick]:
            dep_ready[edge.dst] = max(dep_ready[edge.dst], issue + edge.latency)
            indeg[edge.dst] -= 1
            if indeg[edge.dst] == 0:
                ready.add(edge.dst)

    if len(emitted) != n:
        raise RuntimeError("dependency cycle in pipe-latency scheduler chunk")
    validate_dependency_order(emitted, edges)
    return [chunk[i] for i in emitted], len(edges), max_completion


def schedule_windowed(
    insts: list[Inst],
    window: int,
    policy: str,
    cost_model: CostModel | None = None,
    pipe_ready_slack: int = 0,
    relax_barrier_reg: bool = False,
    ignore_anti_deps: bool = False,
    relax_ldmatrix_order: bool = False,
) -> ScheduleResult:
    out: list[Inst] = []
    segment: list[Inst] = []
    chunks = 0
    edges_checked = 0
    max_chunk_size = 0
    target_cursor = 0
    estimated_issue_cycles = 0

    def flush() -> None:
        nonlocal segment, chunks, edges_checked, max_chunk_size, target_cursor, estimated_issue_cycles
        chunk_window = len(segment) if window <= 0 else window
        if chunk_window <= 0:
            segment = []
            return
        for start in range(0, len(segment), chunk_window):
            chunk = segment[start : start + chunk_window]
            if not chunk:
                continue
            if policy.startswith("pipe-latency"):
                scheduled_chunk, edge_count, chunk_cycles = schedule_chunk_pipe_latency(
                    chunk,
                    policy,
                    pipe_ready_slack,
                    relax_barrier_reg,
                    ignore_anti_deps,
                    relax_ldmatrix_order,
                )
                estimated_issue_cycles += chunk_cycles
            else:
                scheduled_chunk, edge_count, target_cursor = schedule_chunk(
                    chunk,
                    policy,
                    cost_model,
                    target_cursor,
                    relax_barrier_reg,
                    ignore_anti_deps,
                    relax_ldmatrix_order,
                )
            out.extend(scheduled_chunk)
            chunks += 1
            edges_checked += edge_count
            max_chunk_size = max(max_chunk_size, len(chunk))
        segment = []

    for inst in insts:
        if relax_barrier_reg and is_barrier_inst(inst):
            segment.append(inst)
        elif inst.cls in {"control", "boundary"}:
            flush()
            if inst.cls == "control":
                out.append(inst)
        else:
            segment.append(inst)
    flush()
    renumbered = [Inst(i, inst.line, inst.text, inst.op, inst.cls, inst.defs, inst.uses, inst.count) for i, inst in enumerate(out)]
    return ScheduleResult(renumbered, chunks, edges_checked, max_chunk_size, estimated_issue_cycles)


def estimate_chunk_issue_cycles_in_order(chunk: list[Inst]) -> tuple[int, int]:
    _, _, edges = build_dependency_graph(chunk)
    edge_by_src: list[list[DepEdge]] = [[] for _ in range(len(chunk))]
    for edge in edges:
        edge_by_src[edge.src].append(edge)

    dep_ready = [0 for _ in range(len(chunk))]
    pipe_ready: dict[str, int] = defaultdict(int)
    warp_issue_ready = 0
    max_completion = 0

    for i, inst in enumerate(chunk):
        issue = max(dep_ready[i], pipe_ready[inst_pipe(inst)], warp_issue_ready)
        pipe_ready[inst_pipe(inst)] = issue + inst_initiation(inst)
        warp_issue_ready = issue + 1
        max_completion = max(max_completion, issue + inst_latency(inst))
        for edge in edge_by_src[i]:
            dep_ready[edge.dst] = max(dep_ready[edge.dst], issue + edge.latency)

    return max_completion, len(edges)


def estimate_ordered_issue_cycles(insts: list[Inst]) -> IssueEstimate:
    segment: list[Inst] = []
    cycles = 0
    chunks = 0
    edges_checked = 0
    max_chunk_size = 0

    def flush() -> None:
        nonlocal segment, cycles, chunks, edges_checked, max_chunk_size
        if not segment:
            return
        chunk_cycles, edge_count = estimate_chunk_issue_cycles_in_order(segment)
        cycles += chunk_cycles
        chunks += 1
        edges_checked += edge_count
        max_chunk_size = max(max_chunk_size, len(segment))
        segment = []

    for inst in insts:
        if inst.cls in {"control", "boundary"}:
            flush()
        else:
            segment.append(inst)
    flush()

    return IssueEstimate(cycles, chunks, edges_checked, max_chunk_size)


def class_counts(insts: Iterable[Inst]) -> Counter[str]:
    return Counter(inst.cls for inst in insts if inst.cls in WINDOW_FIELDS)


def weighted_counts(insts: Iterable[Inst]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for inst in insts:
        if inst.cls in WINDOW_FIELDS:
            counts[inst.cls] += inst.count
    return counts


def windows(insts: list[Inst], size: int, stride: int) -> list[tuple[int, Counter[str]]]:
    if not insts:
        return []
    if len(insts) <= size:
        return [(0, class_counts(insts))]
    return [(start, class_counts(insts[start : start + size])) for start in range(0, len(insts) - size + 1, stride)]


def vector(counts: Counter[str]) -> list[float]:
    return [float(counts.get(field, 0)) for field in WINDOW_FIELDS]


def cosine(a: Counter[str], b: Counter[str]) -> float:
    va = vector(a)
    vb = vector(b)
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(x * x for x in vb))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def mixed_score(counts: Counter[str]) -> int:
    softmax = counts.get("sfu", 0) + counts.get("fp32", 0) + counts.get("shfl", 0)
    pv = counts.get("tensor", 0) + counts.get("ldmatrix", 0)
    return min(softmax, pv)


def summarize_windows(name: str, insts: list[Inst], size: int, stride: int) -> dict[str, float]:
    ws = windows(insts, size, stride)
    if not ws:
        return {}
    mixed = [mixed_score(c) for _, c in ws]
    has_mix = [score > 0 for score in mixed]
    return {
        "windows": len(ws),
        "mixed_windows": sum(has_mix),
        "mixed_pct": 100.0 * sum(has_mix) / len(ws),
        "mean_mixed_score": sum(mixed) / len(ws),
        "max_mixed_score": max(mixed),
    }


def best_cosine_to_sass(candidate: list[Inst], sass: list[Inst], size: int, stride: int) -> tuple[float, float]:
    cand_w = windows(candidate, size, stride)
    sass_w = windows(sass, size, stride)
    if not cand_w or not sass_w:
        return 0.0, 0.0
    best_scores = []
    for _, s_counts in sass_w:
        best_scores.append(max(cosine(s_counts, c_counts) for _, c_counts in cand_w))
    return sum(best_scores) / len(best_scores), min(best_scores)


def run_lengths(insts: list[Inst]) -> tuple[float, tuple[str, int]]:
    if not insts:
        return 0.0, ("", 0)
    lengths: list[tuple[str, int]] = []
    current = insts[0].cls
    count = 1
    for inst in insts[1:]:
        if inst.cls == current:
            count += 1
        else:
            lengths.append((current, count))
            current = inst.cls
            count = 1
    lengths.append((current, count))
    avg = sum(length for _, length in lengths) / len(lengths)
    return avg, max(lengths, key=lambda item: item[1])


def max_run_by_class(insts: list[Inst]) -> dict[str, int]:
    out = {cls: 0 for cls in WINDOW_FIELDS}
    if not insts:
        return out
    current = insts[0].cls
    count = 1
    for inst in insts[1:]:
        if inst.cls == current:
            count += 1
            continue
        if current in out:
            out[current] = max(out[current], count)
        current = inst.cls
        count = 1
    if current in out:
        out[current] = max(out[current], count)
    return out


def alphabet(insts: list[Inst], start: int, length: int = 128) -> str:
    return "".join(class_token(inst.cls) for inst in insts[start : start + length])


def sequence_string(insts: list[Inst]) -> str:
    return "".join(class_token(inst.cls) for inst in insts if inst.cls in CLASS_TOKENS)


def filtered_sequence_string(insts: list[Inst], keep_classes: set[str]) -> str:
    return "".join(class_token(inst.cls) for inst in insts if inst.cls in keep_classes)


def lcs_length(a: str, b: str) -> int:
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a
    prev = [0] * (len(short) + 1)
    for ch_long in long:
        cur = [0]
        diag = 0
        for j, ch_short in enumerate(short, start=1):
            up = prev[j]
            left = cur[-1]
            cur.append(diag + 1 if ch_long == ch_short else max(up, left))
            diag = up
        prev = cur
    return prev[-1]


def ngram_counter(seq: str, n: int) -> Counter[str]:
    if len(seq) < n:
        return Counter()
    return Counter(seq[i : i + n] for i in range(len(seq) - n + 1))


def ngram_log_scores(seq: str, n: int) -> dict[str, float]:
    counts = ngram_counter(seq, n)
    return {key: math.log1p(value) for key, value in counts.items()}


def weighted_jaccard(a: Counter[str], b: Counter[str]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return inter / union if union else 0.0


def order_metrics(insts: list[Inst], sass: list[Inst]) -> dict[str, float]:
    seq = sequence_string(insts)
    sass_seq = sequence_string(sass)
    matcher_ratio = SequenceMatcher(None, seq, sass_seq, autojunk=False).ratio()
    return {
        "matcher": matcher_ratio,
        "bigram_j": weighted_jaccard(ngram_counter(seq, 2), ngram_counter(sass_seq, 2)),
        "trigram_j": weighted_jaccard(ngram_counter(seq, 3), ngram_counter(sass_seq, 3)),
        "fourgram_j": weighted_jaccard(ngram_counter(seq, 4), ngram_counter(sass_seq, 4)),
    }


def fit_objective(metrics: dict[str, float]) -> float:
    return (
        0.15 * metrics["matcher"]
        + 0.25 * metrics["bigram_j"]
        + 0.30 * metrics["trigram_j"]
        + 0.30 * metrics["fourgram_j"]
    )


def sequence_order_report(name: str, insts: list[Inst], sass: list[Inst]) -> None:
    seq = sequence_string(insts)
    sass_seq = sequence_string(sass)
    sequence_order_report_for_strings(name, seq, sass_seq)


def sequence_order_report_for_strings(name: str, seq: str, sass_seq: str) -> None:
    matcher_ratio = SequenceMatcher(None, seq, sass_seq, autojunk=False).ratio()
    lcs = lcs_length(seq, sass_seq)
    print(
        f"  {name:22s} len={len(seq):4d} matcher={matcher_ratio:.4f} "
        f"lcs/min={lcs / min(len(seq), len(sass_seq)) if seq and sass_seq else 0.0:.4f} "
        f"bigram_j={weighted_jaccard(ngram_counter(seq, 2), ngram_counter(sass_seq, 2)):.4f} "
        f"trigram_j={weighted_jaccard(ngram_counter(seq, 3), ngram_counter(sass_seq, 3)):.4f} "
        f"fourgram_j={weighted_jaccard(ngram_counter(seq, 4), ngram_counter(sass_seq, 4)):.4f}"
    )


def class_region(insts: list[Inst], keep_classes: set[str]) -> list[Inst]:
    kept = [i for i, inst in enumerate(insts) if inst.cls in keep_classes]
    if not kept:
        return []
    return insts[kept[0] : kept[-1] + 1]


def token_run_summary(seq: str) -> tuple[float, tuple[str, int], str]:
    if not seq:
        return 0.0, ("", 0), ""
    runs: list[tuple[str, int]] = []
    token = seq[0]
    count = 1
    for ch in seq[1:]:
        if ch == token:
            count += 1
        else:
            runs.append((token, count))
            token = ch
            count = 1
    runs.append((token, count))
    avg = sum(length for _, length in runs) / len(runs)
    compact = " ".join(f"{tok}{length}" for tok, length in runs[:32])
    if len(runs) > 32:
        compact += " ..."
    return avg, max(runs, key=lambda item: item[1]), compact


def print_focus_region_report(
    name: str, insts: list[Inst], sass: list[Inst], keep_classes: set[str], width: int = 160
) -> None:
    region = class_region(insts, keep_classes)
    sass_region = class_region(sass, keep_classes)
    seq = filtered_sequence_string(region, keep_classes)
    sass_seq = filtered_sequence_string(sass_region, keep_classes)
    avg_run, max_run, runs = token_run_summary(seq)
    print(
        f"  {name:22s} region_insts={len(region):4d} filtered_len={len(seq):4d} "
        f"avg_run={avg_run:.2f} max_run={max_run[0]}:{max_run[1]}"
    )
    print(f"    head={seq[:width]}")
    print(f"    runs={runs}")
    if sass_seq:
        sequence_order_report_for_strings(f"{name}_focus", seq, sass_seq)


def make_cost_model(
    sass: list[Inst],
    *,
    height: float,
    age: float,
    switch: float,
    recent_same: float,
    bigram: float,
    trigram: float,
    fourgram: float = 0.0,
    target_align: float = 0.0,
    target_lookahead: int = 64,
    target_dist: float = 0.0,
    target_dist_window: int = 96,
    alt_repeat: float = 0.0,
    class_bias: dict[str, float],
    target_sequence_override: str | None = None,
    target_items: tuple[GuideItem, ...] = (),
    target_identity: float = 0.0,
) -> CostModel:
    sass_seq = sequence_string(sass)
    target_sequence = target_sequence_override if target_sequence_override is not None else sass_seq
    return CostModel(
        height=height,
        age=age,
        switch=switch,
        recent_same=recent_same,
        bigram=bigram,
        trigram=trigram,
        fourgram=fourgram,
        target_align=target_align,
        target_lookahead=target_lookahead,
        target_dist=target_dist,
        target_dist_window=target_dist_window,
        alt_repeat=alt_repeat,
        class_bias=class_bias,
        sass_bigram_log=ngram_log_scores(sass_seq, 2),
        sass_trigram_log=ngram_log_scores(sass_seq, 3),
        sass_fourgram_log=ngram_log_scores(sass_seq, 4),
        target_sequence=target_sequence,
        target_items=target_items,
        target_identity=target_identity,
    )


def default_cost_model(sass: list[Inst]) -> CostModel:
    return make_cost_model(
        sass,
        height=1.0,
        age=0.002,
        switch=12.0,
        recent_same=-4.0,
        bigram=0.0,
        trigram=0.0,
        fourgram=0.0,
        target_align=0.0,
        target_lookahead=64,
        target_dist=0.0,
        target_dist_window=96,
        alt_repeat=0.0,
        class_bias={
            "tensor": 5.0,
            "sfu": 4.0,
            "ldmatrix": 3.0,
            "fp32": 2.0,
            "shfl": 1.0,
            "mem": 0.0,
            "int": 0.0,
            "control": 0.0,
            "other": 0.0,
        },
    )


def sass_guided_cost_model(
    sass: list[Inst],
    target_sequence: str,
    *,
    guide_weight: float,
    guide_lookahead: int,
    height_weight: float,
    age_weight: float,
    switch_weight: float,
    recent_same_weight: float,
    guide_items: tuple[GuideItem, ...] = (),
    identity_weight: float = 0.0,
) -> CostModel:
    return make_cost_model(
        sass,
        height=height_weight,
        age=age_weight,
        switch=switch_weight,
        recent_same=recent_same_weight,
        bigram=0.0,
        trigram=0.0,
        fourgram=0.0,
        target_align=guide_weight,
        target_lookahead=guide_lookahead,
        target_dist=0.0,
        target_dist_window=96,
        alt_repeat=0.0,
        class_bias={
            "tensor": 0.0,
            "ldmatrix": 0.0,
            "sfu": 0.0,
            "fp32": 0.0,
            "shfl": 0.0,
            "mem": 0.0,
            "int": 0.0,
            "control": 0.0,
            "other": 0.0,
        },
        target_sequence_override=target_sequence,
        target_items=guide_items,
        target_identity=identity_weight,
    )


def random_cost_model(rng: random.Random, sass: list[Inst]) -> CostModel:
    pipe_bias = {
        "tensor": 5.0,
        "sfu": 4.0,
        "ldmatrix": 3.0,
        "fp32": 2.0,
        "shfl": 1.0,
        "mem": 0.0,
        "int": 0.0,
        "control": 0.0,
        "other": 0.0,
    }
    class_bias = {
        cls: pipe_bias.get(cls, 0.0) * rng.uniform(0.0, 2.0) + rng.uniform(-4.0, 4.0)
        for cls in WINDOW_FIELDS
    }
    return make_cost_model(
        sass,
        height=rng.choice([0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]),
        age=rng.uniform(-0.01, 0.02),
        switch=rng.uniform(-8.0, 24.0),
        recent_same=rng.uniform(-12.0, 3.0),
        bigram=rng.choice([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0]),
        trigram=rng.choice([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0]),
        fourgram=rng.choice([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0]),
        target_align=rng.choice([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 24.0]),
        target_lookahead=rng.choice([16, 32, 64, 128, 256]),
        target_dist=rng.choice([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0]),
        target_dist_window=rng.choice([32, 64, 96, 128, 192, 256]),
        alt_repeat=rng.choice([0.0, -1.0, -2.0, -4.0, -8.0, -12.0]),
        class_bias=class_bias,
    )


def fit_cost_models(
    ptx_with_boundaries: list[Inst],
    sass: list[Inst],
    window: int,
    trials: int,
    seed: int,
    top_k: int,
    relax_barrier_reg: bool = False,
    ignore_anti_deps: bool = False,
    relax_ldmatrix_order: bool = False,
) -> list[tuple[float, dict[str, float], CostModel, ScheduleResult]]:
    rng = random.Random(seed)
    candidates = [
        default_cost_model(sass),
        make_cost_model(
            sass,
            height=0.2,
            age=0.004,
            switch=16.0,
            recent_same=-6.0,
            bigram=4.0,
            trigram=4.0,
            fourgram=2.0,
            target_align=0.0,
            target_lookahead=64,
            target_dist=0.0,
            target_dist_window=96,
            alt_repeat=-4.0,
            class_bias={"tensor": 5.0, "ldmatrix": 3.0, "sfu": 4.0, "fp32": 2.0, "shfl": 1.0},
        ),
        make_cost_model(
            sass,
            height=0.0,
            age=0.006,
            switch=8.0,
            recent_same=-4.0,
            bigram=8.0,
            trigram=12.0,
            fourgram=8.0,
            target_align=0.0,
            target_lookahead=64,
            target_dist=0.0,
            target_dist_window=96,
            alt_repeat=-4.0,
            class_bias={"tensor": 2.0, "ldmatrix": 2.0, "sfu": 2.0, "fp32": 1.0, "int": 0.5},
        ),
        make_cost_model(
            sass,
            height=0.2,
            age=0.01,
            switch=8.0,
            recent_same=-4.0,
            bigram=2.0,
            trigram=4.0,
            fourgram=2.0,
            target_align=8.0,
            target_lookahead=128,
            target_dist=2.0,
            target_dist_window=96,
            alt_repeat=-4.0,
            class_bias={"tensor": 3.0, "ldmatrix": 3.0, "sfu": 2.0, "fp32": 1.0, "int": 0.5},
        ),
    ]
    candidates.extend(random_cost_model(rng, sass) for _ in range(max(0, trials)))

    results: list[tuple[float, dict[str, float], CostModel, ScheduleResult]] = []
    for model in candidates:
        result = schedule_windowed(
            ptx_with_boundaries,
            window,
            "cost-model",
            model,
            relax_barrier_reg=relax_barrier_reg,
            ignore_anti_deps=ignore_anti_deps,
            relax_ldmatrix_order=relax_ldmatrix_order,
        )
        metrics = order_metrics(result.insts, sass)
        results.append((fit_objective(metrics), metrics, model, result))

    results.sort(key=lambda row: row[0], reverse=True)
    return results[:top_k]


def window_order_similarity(candidate: list[Inst], sass: list[Inst], size: int, stride: int) -> tuple[float, float, float]:
    cand_windows = [(start, sequence_string(candidate[start : start + size])) for start, _ in windows(candidate, size, stride)]
    sass_windows = [(start, sequence_string(sass[start : start + size])) for start, _ in windows(sass, size, stride)]
    if not cand_windows or not sass_windows:
        return 0.0, 0.0, 0.0
    best_scores: list[float] = []
    for _, sass_seq in sass_windows:
        best = 0.0
        for _, cand_seq in cand_windows:
            if not sass_seq or not cand_seq:
                continue
            best = max(best, SequenceMatcher(None, sass_seq, cand_seq, autojunk=False).ratio())
        best_scores.append(best)
    return sum(best_scores) / len(best_scores), min(best_scores), max(best_scores)


def print_sequence_report(name: str, insts: list[Inst], size: int, stride: int, dynamic: bool = False) -> None:
    counts = weighted_counts(insts) if dynamic else class_counts(insts)
    total = sum(counts.values())
    avg_run, max_run = run_lengths(insts)
    max_by_class = max_run_by_class(insts)
    print(f"\n== {name} ==")
    print(f"instructions={len(insts)} counted_total={total} avg_run={avg_run:.2f} max_run={max_run[0]}:{max_run[1]}")
    print(
        "  max_run_by_class "
        + " ".join(f"{class_token(cls)}:{max_by_class.get(cls, 0)}" for cls in WINDOW_FIELDS)
    )
    for field in WINDOW_FIELDS:
        value = counts.get(field, 0)
        print(f"  {field:9s} {value:8d} {100.0 * value / total if total else 0.0:6.2f}%")
    summary = summarize_windows(name, insts, size, stride)
    print(
        "  windows={windows:.0f} mixed={mixed_windows:.0f} ({mixed_pct:.1f}%) "
        "mean_mix={mean_mixed_score:.2f} max_mix={max_mixed_score:.0f}".format(**summary)
    )
    top = sorted(windows(insts, size, stride), key=lambda item: mixed_score(item[1]), reverse=True)[:3]
    for start, c in top:
        softmax = c.get("sfu", 0) + c.get("fp32", 0) + c.get("shfl", 0)
        pv = c.get("tensor", 0) + c.get("ldmatrix", 0)
        print(
            f"  top_mix start={start:5d} score={mixed_score(c):2d} "
            f"softmax={softmax:2d} tensor+ldm={pv:2d} "
            f"counts={{T:{c.get('tensor',0)},L:{c.get('ldmatrix',0)},S:{c.get('sfu',0)},F:{c.get('fp32',0)},H:{c.get('shfl',0)}}}"
        )
        print(f"    {alphabet(insts, start, min(size, 96))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptx", type=Path, default=Path(DEFAULT_PTX))
    parser.add_argument("--stats", type=Path, default=Path(DEFAULT_STATS))
    parser.add_argument("--sass", type=Path, default=Path(DEFAULT_SASS))
    parser.add_argument(
        "--ptx-function-regex",
        default=None,
        help=(
            "Optional regex selecting one PTX .entry/.func. This is useful when "
            "comparing a PTX/SASS pair from the same dump without line stats."
        ),
    )
    parser.add_argument(
        "--sass-function-regex",
        default=r"ELb0ELb0ELb0ELb0ELb1ELb1ELb0ELb0",
        help="Regex selecting the SASS function to compare. Use --list-sass-functions to inspect names.",
    )
    parser.add_argument("--list-ptx-functions", action="store_true")
    parser.add_argument("--list-sass-functions", action="store_true")
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument(
        "--metric-window",
        type=int,
        default=96,
        help="Sliding-window size for reporting metrics. Scheduling uses --window; use --window 0 for whole basic-block chunks.",
    )
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument(
        "--policy",
        choices=(
            "critical",
            "balanced",
            "aggressive-mix",
            "cost-model",
            "sass-guided",
            "sass-guided-defuse",
            "pipe-latency",
            "pipe-latency-tl",
            "pipe-latency-switch",
            "pipe-latency-defuse",
            "pipe-latency-stage",
            "pipe-latency-antirun",
            "pipe-latency-inventory",
        ),
        default="critical",
        help="Ready-list priority for the offline window scheduler.",
    )
    parser.add_argument(
        "--sass-guide-filter",
        choices=tuple(SASS_GUIDE_FILTERS),
        default="major",
        help="SASS instruction classes used as the direct guide stream for policy=sass-guided.",
    )
    parser.add_argument(
        "--sass-guide-lookahead",
        type=int,
        default=256,
        help="Maximum guide-stream distance searched for each ready PTX candidate.",
    )
    parser.add_argument(
        "--sass-guide-weight",
        type=float,
        default=32.0,
        help="Penalty weight for distance from the current SASS guide cursor.",
    )
    parser.add_argument(
        "--sass-guide-height-weight",
        type=float,
        default=0.05,
        help="Tie-break weight for dependency critical height under policy=sass-guided.",
    )
    parser.add_argument(
        "--sass-guide-age-weight",
        type=float,
        default=0.002,
        help="Tie-break weight favoring original PTX source order under policy=sass-guided.",
    )
    parser.add_argument(
        "--sass-guide-switch-weight",
        type=float,
        default=0.0,
        help="Tie-break weight for switching instruction classes under policy=sass-guided.",
    )
    parser.add_argument(
        "--sass-guide-recent-same-weight",
        type=float,
        default=-0.25,
        help="Tie-break weight for repeating recently emitted classes under policy=sass-guided.",
    )
    parser.add_argument(
        "--sass-guide-identity-weight",
        type=float,
        default=0.25,
        help=(
            "Extra guide score for matching LDSM/HMMA def-use role signatures under "
            "policy=sass-guided-defuse."
        ),
    )
    parser.add_argument(
        "--pipe-ready-slack",
        type=int,
        default=0,
        help="Extra cycles a pipe-latency policy may wait when choosing among ready-list candidates.",
    )
    parser.add_argument(
        "--relax-barrier-reg",
        action="store_true",
        help=(
            "Allow register-only instructions to be scheduled across bar.sync "
            "in the offline model; ldmatrix/memory/control stay barrier-ordered."
        ),
    )
    parser.add_argument(
        "--ignore-anti-deps",
        action="store_true",
        help=(
            "Offline upper bound: drop PTX WAR/WAW edges. This is not semantics-preserving "
            "unless a later register-renaming pass recreates equivalent PTX values."
        ),
    )
    parser.add_argument(
        "--relax-ldmatrix-order",
        action="store_true",
        help=(
            "Offline experiment: do not force independent ldmatrix instructions to remain in "
            "PTX source order. Other memory instructions still keep memory order."
        ),
    )
    parser.add_argument(
        "--fit-cost-model",
        type=int,
        default=0,
        metavar="TRIALS",
        help="Random-search TRIALS cost models and schedule with policy=cost-model.",
    )
    parser.add_argument("--fit-seed", type=int, default=1)
    parser.add_argument("--fit-top-k", type=int, default=8)
    parser.add_argument(
        "--skip-window-order-similarity",
        action="store_true",
        help="Skip the expensive all-window SequenceMatcher comparison.",
    )
    args = parser.parse_args()

    if args.list_ptx_functions:
        for fn in list_ptx_functions(args.ptx):
            print(fn)
        return 0

    if args.list_sass_functions:
        for fn in list_sass_functions(args.sass):
            print(fn)
        return 0

    ptx_with_boundaries = parse_ptx(
        args.ptx, args.stats if args.stats.exists() else None, args.ptx_function_regex
    )
    ptx = [inst for inst in ptx_with_boundaries if inst.cls != "boundary"]
    sass = parse_sass(args.sass, args.sass_function_regex)
    sass_guide_classes = SASS_GUIDE_FILTERS[args.sass_guide_filter]
    sass_guide_sequence = filtered_sequence_string(sass, sass_guide_classes)
    sass_guide_items = build_guide_items(
        sass,
        sass_guide_classes,
        args.relax_barrier_reg,
        args.ignore_anti_deps,
        True,
    )
    metric_window = args.metric_window if args.metric_window > 0 else 96

    best_model: CostModel | None = None
    if args.fit_cost_model > 0:
        fit_results = fit_cost_models(
            ptx_with_boundaries,
            sass,
            args.window,
            args.fit_cost_model,
            args.fit_seed,
            args.fit_top_k,
            args.relax_barrier_reg,
            args.ignore_anti_deps,
            args.relax_ldmatrix_order,
        )
        print("== Cost Model Fit ==")
        print(
            f"trials={args.fit_cost_model} seed={args.fit_seed} "
            f"schedule_window={'whole-boundary-region' if args.window <= 0 else args.window} "
            f"top_k={args.fit_top_k}"
        )
        for rank, (score, metrics, model, result) in enumerate(fit_results, start=1):
            print(
                f"rank={rank:02d} score={score:.4f} matcher={metrics['matcher']:.4f} "
                f"bigram_j={metrics['bigram_j']:.4f} trigram_j={metrics['trigram_j']:.4f} "
                f"fourgram_j={metrics['fourgram_j']:.4f} "
                f"chunks={result.chunks} max_chunk={result.max_chunk_size} edges={result.edges_checked}"
            )
            print(f"  {model.summary()}")
        _, _, best_model, schedule_result = fit_results[0]
        scheduled = schedule_result.insts
    else:
        if args.policy == "cost-model":
            best_model = default_cost_model(sass)
        elif args.policy in {"sass-guided", "sass-guided-defuse"}:
            best_model = sass_guided_cost_model(
                sass,
                sass_guide_sequence,
                guide_weight=args.sass_guide_weight,
                guide_lookahead=args.sass_guide_lookahead,
                height_weight=args.sass_guide_height_weight,
                age_weight=args.sass_guide_age_weight,
                switch_weight=args.sass_guide_switch_weight,
                recent_same_weight=args.sass_guide_recent_same_weight,
                guide_items=sass_guide_items if args.policy == "sass-guided-defuse" else (),
                identity_weight=args.sass_guide_identity_weight if args.policy == "sass-guided-defuse" else 0.0,
            )
        else:
            best_model = None
        schedule_result = schedule_windowed(
            ptx_with_boundaries,
            args.window,
            args.policy,
            best_model,
            args.pipe_ready_slack,
            args.relax_barrier_reg,
            args.ignore_anti_deps,
            args.relax_ldmatrix_order,
        )
        scheduled = schedule_result.insts

    print("PTX window scheduler probe")
    print(f"ptx={args.ptx}")
    print(f"ptx_function_regex={args.ptx_function_regex if args.ptx_function_regex else '<all>'}")
    print(f"stats={args.stats if args.stats and args.stats.exists() else '<none>'}")
    print(f"sass={args.sass}")
    print(f"sass_function_regex={args.sass_function_regex}")
    print(f"schedule_window={'whole-boundary-region' if args.window <= 0 else args.window} metric_window={metric_window} stride={args.stride}")
    active_policy = "cost-model-fit" if args.fit_cost_model > 0 else args.policy
    print(f"policy={active_policy}")
    print(f"relax_barrier_reg={int(args.relax_barrier_reg)}")
    print(f"ignore_anti_deps={int(args.ignore_anti_deps)}")
    print(f"relax_ldmatrix_order={int(args.relax_ldmatrix_order)}")
    if args.policy in {"sass-guided", "sass-guided-defuse"}:
        print(
            f"sass_guide_filter={args.sass_guide_filter} "
            f"sass_guide_classes={','.join(sorted(sass_guide_classes))} "
            f"sass_guide_len={len(sass_guide_sequence)} "
            f"sass_guide_lookahead={args.sass_guide_lookahead} "
            f"sass_guide_weight={args.sass_guide_weight} "
            f"sass_guide_items={len(sass_guide_items)} "
            f"sass_guide_identity_weight={args.sass_guide_identity_weight if args.policy == 'sass-guided-defuse' else 0.0}"
        )
        if args.policy == "sass-guided-defuse" and sass_guide_items:
            print("sass_guide_identity_head:")
            for item in sass_guide_items[:12]:
                print(f"  {role_signature_brief(item.sig)} line=0x{item.line:x}")
    if best_model is not None:
        print(f"cost_model={best_model.summary()}")
    if args.policy.startswith("pipe-latency"):
        print(f"pipe_model={pipe_model_summary()}")
        print(f"pipe_policy={pipe_policy_summary(args.policy, args.pipe_ready_slack)}")
        original_estimate = estimate_ordered_issue_cycles(ptx_with_boundaries)
        print(
            f"original_pipe_issue_cycles={original_estimate.cycles} "
            f"chunks={original_estimate.chunks} "
            f"max_chunk_size={original_estimate.max_chunk_size} "
            f"dependency_edges_checked={original_estimate.edges_checked}"
        )
    print(
        f"scheduler_chunks={schedule_result.chunks} "
        f"max_chunk_size={schedule_result.max_chunk_size} "
        f"dependency_edges_checked={schedule_result.edges_checked}"
    )
    if schedule_result.estimated_issue_cycles:
        print(f"estimated_pipe_issue_cycles={schedule_result.estimated_issue_cycles}")
        if args.policy.startswith("pipe-latency"):
            print(
                "estimated_reorder_delta_cycles="
                f"{original_estimate.cycles - schedule_result.estimated_issue_cycles} "
                f"speedup={original_estimate.cycles / schedule_result.estimated_issue_cycles:.3f}x"
            )
    print("note=PTX sequence is static source order filtered by executed line stats, not a dynamic trace.")
    print("note=PTX labels and control/fence instructions are hard scheduling boundaries; labels are not counted as instructions.")

    print_sequence_report("ptx_original_static", ptx, metric_window, args.stride)
    print_sequence_report("ptx_scheduled_static", scheduled, metric_window, args.stride)
    print_sequence_report("sass_static", sass, metric_window, args.stride)

    avg_orig, min_orig = best_cosine_to_sass(ptx, sass, metric_window, args.stride)
    avg_sched, min_sched = best_cosine_to_sass(scheduled, sass, metric_window, args.stride)
    print("\n== SASS window similarity ==")
    print(f"best-match cosine avg: original={avg_orig:.4f} scheduled={avg_sched:.4f}")
    print(f"best-match cosine min: original={min_orig:.4f} scheduled={min_sched:.4f}")

    print("\n== SASS order similarity ==")
    print("  metrics use class-token order, not only class counts.")
    sequence_order_report("ptx_original", ptx, sass)
    sequence_order_report("ptx_scheduled", scheduled, sass)
    if args.policy in {"sass-guided", "sass-guided-defuse"}:
        print("\n== SASS guide order similarity ==")
        print("  metrics use only classes selected by --sass-guide-filter.")
        sequence_order_report_for_strings(
            "ptx_original_guide",
            filtered_sequence_string(ptx, sass_guide_classes),
            sass_guide_sequence,
        )
        sequence_order_report_for_strings(
            "ptx_scheduled_guide",
            filtered_sequence_string(scheduled, sass_guide_classes),
            sass_guide_sequence,
        )
        print("\n== Focus class region ==")
        print("  region spans first-to-last selected class; head/runs are filtered to selected classes.")
        print_focus_region_report("ptx_original_region", ptx, sass, sass_guide_classes)
        print_focus_region_report("ptx_scheduled_region", scheduled, sass, sass_guide_classes)
        print_focus_region_report("sass_region", sass, sass, sass_guide_classes)
        print("\n== SASS guide role alignment ==")
        print("  metrics compare per-position LDSM/HMMA def-use role signatures in the selected guide stream.")
        print_role_alignment_report("ptx_original_role", ptx, sass, sass_guide_classes)
        print_role_alignment_report("ptx_scheduled_role", scheduled, sass, sass_guide_classes)
    if args.skip_window_order_similarity:
        print("  best-window SequenceMatcher avg/min/max: <skipped>")
    else:
        win_orig = window_order_similarity(ptx, sass, metric_window, args.stride)
        win_sched = window_order_similarity(scheduled, sass, metric_window, args.stride)
        print(
            f"  best-window SequenceMatcher avg/min/max: "
            f"original={win_orig[0]:.4f}/{win_orig[1]:.4f}/{win_orig[2]:.4f} "
            f"scheduled={win_sched[0]:.4f}/{win_sched[1]:.4f}/{win_sched[2]:.4f}"
        )

    if args.stats and args.stats.exists():
        print_sequence_report("ptx_original_dynamic_weighted_counts", ptx, metric_window, args.stride, dynamic=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
