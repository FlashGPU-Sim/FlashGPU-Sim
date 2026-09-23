#!/usr/bin/env python3
"""Fair NCU vs GPGPU-Sim comparison for TMA workloads (cycles + L1/L2 cache).

==============================================================================
CRITICAL: NCU ``--cache-control all`` (cold-start vs warm-cache)
==============================================================================

GPGPU-Sim runs each kernel from a **cold** L1/L2 (empty tags at kernel start
unless you explicitly model warm-up). Real-GPU NCU reports are **not** cold by
default, and that single choice can dominate L2 hit-rate comparison.

NCU option ``--cache-control``:

  all   (REQUIRED for fair cache comparison with FlashGPU-Sim)
        Flush / invalidate GPU caches between profiling passes so each
        measured kernel launch sees a cold (or controlled) cache hierarchy.
        Without this, later NCU replay passes hit a warm L2 and hit rates
        climb toward ~100% even when the algorithm is streaming.

  none  (UNFAIR for L2 hit rate vs sim)
        Leave caches as-is across passes. Application or multi-pass kernel
        replay then measures a **warm** L2. On the 128^3 TMA GEMM this alone
        moved NCU L2 hit rate from ~25% (cold, matches sim) to ~99% (warm),
        creating a false multi-x "sim is wrong" signal.

Why this matters for the 10% cache goal:

  * L2 hit rate is highly sensitive to cold vs warm start. A warm NCU run is
    not comparable to a cold sim run; any |diff| is mostly methodology.
  * Cycle counts also shift slightly with warm L2 (fewer DRAM fills), so
    cold-start NCU is preferred for the cycle band as well when validating
    TMA memory behavior.
  * ``--cache-control all`` is independent of metric choice: even with fair
    TMA-scoped sector metrics, warm L2 still invalidates hit-rate comparison.

Recommended cold-start NCU collection (save as ``*_cold.ncu-rep``)::

    ncu -f -o <shape>_cold --set full \\
        --replay-mode kernel \\
        --cache-control all \\
        --clock-control none \\
        --target-processes all \\
        ./kernel_tma_gemm_launch1

Notes:

  * Prefer ``--replay-mode kernel`` over ``application`` for single-kernel
    harnesses; still use ``--cache-control all`` so multi-pass metric
    collection does not warm the cache across passes.
  * This script prefers ``results/ncu-rep/<shape>_cold.ncu-rep`` when present.
  * Do **not** use reports collected with ``--cache-control none`` as the
    oracle for L2 hit rate against GPGPU-Sim.

==============================================================================
WHY THE OLD SCRIPTS ARE UNFAIR (do not use them as the sole cache oracle)
==============================================================================

1) extract_metrics.py uses NCU metric ``lts__t_sectors.sum`` (ALL L2 clients)
   and compares it to GPGPU-Sim ``L2_total_cache_accesses``, which for TMA
   GEMM is dominated by TMA sector traffic only.

   For example, on a 128x128x128 TMA GEMM case this caused a fake volume gap:
     - Sim L2 accesses ≈ 4096 sectors  (TMA only: 3072R + 1024W)
     - NCU lts__t_sectors.sum ≈ 13187  (TMA/tex + many non-TMA clients)
   Fair TMA-scoped NCU counts match sim almost exactly:
     - lts__t_sectors_srcunit_tex_op_read.sum  = 3072
     - lts__t_sectors_srcunit_tex_op_write.sum = 1024

2) compare_cycles.py / extract_metrics.py often consume NCU reports collected
   with ``--replay-mode application`` and ``--cache-control none``. Application
   replay re-runs the process many times with a **warm** L2 (see
   ``--cache-control`` section above), inflating
   ``lts__t_sector_hit_rate.pct`` toward ~99% while GPGPU-Sim is a cold-start
   single kernel run (~25% hit rate for a streaming 128^3 TMA GEMM). That is a
   methodology mismatch, not necessarily a TMA modeling bug.

3) L1 hit rate from extract_metrics.py is misleading for TMA: TMA bypasses L1D
   (sim L1D Access = 0). A "N/A vs some L1tex rate" pair is not a failure of
   the TMA path; fair comparison must treat L1 as expected-bypass for TMA.

4) DRAM volume is a separate fidelity axis (write-back L2 without real flush
   writeback). This script deliberately de-prioritizes DRAM and focuses on
   cycles + L1/L2 cache for TMA validation.

==============================================================================
FAIR COMPARISON METHOD (this script)
==============================================================================

Cycles
  - Sim:  gpu_tot_sim_cycle from GPGPU-Sim log (or sim-log/summary.csv)
  - NCU:  sm__cycles_elapsed.avg from the .ncu-rep
  - Target band: |diff%| <= --cycle-tol (default 10%)

L2 traffic (TMA-scoped)
  - Sim:  L2_total_cache_accesses  (for pure-TMA kernels ≈ TMA sectors)
          also reports traffic_breakdown TMA_ACC_R / TMA_ACC_W if present
  - NCU:  tex/TMA path sector counts:
            lts__t_sectors_srcunit_tex_op_read.sum
          + lts__t_sectors_srcunit_tex_op_write.sum
          (+ optional tex atom sectors if present)
  - NOT used: lts__t_sectors.sum (unfair whole-L2 aggregate)

L2 hit rate (TMA-scoped when possible)
  - Sim:  100 * (1 - L2_total_cache_miss_rate)
          and a second view that counts pending hits as non-misses:
            hit_or_pending% = 100 * (accesses - misses) / accesses
          Note: GPGPU-Sim "Pending_hits" are concurrent MSHR merges, not pure
          tag hits after fill; both numbers are printed.
  - NCU:  Prefer tex-path hit rate when sector hit/miss available:
            hits   = lts__t_sectors_srcunit_tex_lookup_hit.sum
            misses = lts__t_sectors_srcunit_tex_lookup_miss.sum
            hit%   = 100 * hits / (hits + misses)
          Fallback: lts__t_sector_hit_rate.pct (warn: whole-L2, often warm)
  - NCU must be collected with ``--cache-control all`` (see top section).

L1
  - Expected bypass for TMA GEMM. Script reports sim L1D accesses and NCU
    l1tex global LD sectors; does not fail the 10% band on L1 for TMA.

Usage
-----
  # 1) Cold-start NCU (REQUIRED for fair L2 hit rate):
  ncu -f -o results/ncu-rep/m128_n128_k128_cold --set full \\
      --replay-mode kernel --cache-control all --clock-control none \\
      --target-processes all ./kernel_tma_gemm_launch1

  # 2) Cold sim (from launchers/ with setup_environment):
  ./kernel_tma_gemm_launch1   # log under results/sim-log/

  # 3) Fair compare:
  python3 compare_tma_fair.py test_tma_gemm --shape m128_n128_k128

  # All shapes that have both sim-log and ncu-rep:
  python3 compare_tma_fair.py test_tma_gemm

  # Custom tolerances (default 10% cycles, 10% L2 traffic, hit-rate abs 10 pp):
  python3 compare_tma_fair.py test_tma_gemm --cycle-tol 10 --l2-traffic-tol 10 \\
      --l2-hit-tol 10

Outputs a per-shape table, pass/fail against the 10% goals, and writes
  triton_kernel_tracking/<test>/results/fair_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRITON_TRACE_DIR = os.path.dirname(SCRIPT_DIR)
TRACKING_DIR = os.path.join(TRITON_TRACE_DIR, "triton_kernel_tracking")

SECTOR_BYTES = 32

# Metrics pulled from NCU for fair TMA comparison
NCU_METRICS = [
    "sm__cycles_elapsed.avg",
    "sm__cycles_elapsed.avg.per_second",
    "gpu__time_duration.avg",
    # TMA/tex-scoped L2 sectors (fair traffic)
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_write.sum",
    "lts__t_sectors_srcunit_tex_op_atom.sum",
    "lts__t_sectors_srcunit_tex_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex.sum",
    # Unfair aggregates (reported only for diagnosis)
    "lts__t_sectors.sum",
    "lts__t_sector_hit_rate.pct",
    # L1 (expect little/no TMA data path)
    "l1tex__t_sector_hit_rate.pct",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _f(x: Optional[float], nd: int = 2) -> str:
    if x is None:
        return "N/A"
    return f"{x:.{nd}f}"


def _pct_diff(sim: Optional[float], ncu: Optional[float]) -> Optional[float]:
    """(sim - ncu) / ncu * 100."""
    if sim is None or ncu is None or ncu == 0:
        return None
    return (sim - ncu) / ncu * 100.0


def _abs_diff(sim: Optional[float], ncu: Optional[float]) -> Optional[float]:
    if sim is None or ncu is None:
        return None
    return sim - ncu


def parse_sim_log(path: str) -> Dict[str, Optional[float]]:
    with open(path) as f:
        content = f.read()

    out: Dict[str, Optional[float]] = {}

    def grepf(pat: str) -> Optional[float]:
        m = re.search(pat, content, re.MULTILINE)
        return float(m.group(1)) if m else None

    out["sim_cycles"] = grepf(r"^gpu_tot_sim_cycle\s*=\s*([\d.]+)")
    out["sim_insn"] = grepf(r"^gpu_tot_sim_insn\s*=\s*([\d.]+)")
    out["sim_ipc"] = grepf(r"^gpu_tot_ipc\s*=\s*([\d.]+)")

    out["l2_accesses"] = grepf(r"L2_total_cache_accesses\s*=\s*(\d+)")
    out["l2_misses"] = grepf(r"L2_total_cache_misses\s*=\s*(\d+)")
    out["l2_pending_hits"] = grepf(r"L2_total_cache_pending_hits\s*=\s*(\d+)")
    out["l2_miss_rate"] = grepf(r"L2_total_cache_miss_rate\s*=\s*([\d.]+)")

    if out["l2_accesses"] is not None:
        out["l2_traffic_sectors"] = out["l2_accesses"]
        out["l2_traffic_mb"] = out["l2_accesses"] * SECTOR_BYTES / (1024 * 1024)
    else:
        out["l2_traffic_sectors"] = None
        out["l2_traffic_mb"] = None

    # Pure tag-hit style: non-miss fraction (includes pending hits as "not miss")
    if out["l2_accesses"] and out["l2_misses"] is not None and out["l2_accesses"] > 0:
        out["l2_hit_or_pending_pct"] = (
            100.0 * (out["l2_accesses"] - out["l2_misses"]) / out["l2_accesses"]
        )
    else:
        out["l2_hit_or_pending_pct"] = None

    # GPGPU-Sim reported hit rate (same as 1 - miss_rate)
    if out["l2_miss_rate"] is not None:
        out["l2_hit_rate_pct"] = (1.0 - out["l2_miss_rate"]) * 100.0
    else:
        out["l2_hit_rate_pct"] = out["l2_hit_or_pending_pct"]

    # Pure hits excluding pending: accesses - misses - pending
    if (
        out["l2_accesses"]
        and out["l2_misses"] is not None
        and out["l2_pending_hits"] is not None
        and out["l2_accesses"] > 0
    ):
        pure_hits = out["l2_accesses"] - out["l2_misses"] - out["l2_pending_hits"]
        out["l2_pure_hit_pct"] = 100.0 * max(pure_hits, 0) / out["l2_accesses"]
    else:
        out["l2_pure_hit_pct"] = None

    # TMA traffic breakdown (bytes on interconnect, convert to sector-ish counts)
    # Format: traffic_breakdown_coretomem[TMA_ACC_R] = 24576 {8:3072,}
    # The {packet_size:count} map's count is the number of requests.
    def tma_req_count(access: str, direction: str) -> Optional[float]:
        # direction: coretomem or memtocore
        pat = rf"traffic_breakdown_{direction}\[{access}\]\s*=\s*\d+\s*\{{([^}}]+)\}}"
        m = re.search(pat, content)
        if not m:
            return None
        body = m.group(1)
        total = 0
        for part in body.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            _, cnt = part.split(":")
            total += int(cnt)
        return float(total) if total else None

    out["tma_acc_r_reqs"] = tma_req_count("TMA_ACC_R", "coretomem")
    out["tma_acc_w_reqs"] = tma_req_count("TMA_ACC_W", "coretomem")
    if out["tma_acc_r_reqs"] is not None or out["tma_acc_w_reqs"] is not None:
        r = out["tma_acc_r_reqs"] or 0.0
        w = out["tma_acc_w_reqs"] or 0.0
        out["tma_traffic_sectors"] = r + w
        out["tma_traffic_mb"] = (r + w) * SECTOR_BYTES / (1024 * 1024)
    else:
        out["tma_traffic_sectors"] = None
        out["tma_traffic_mb"] = None

    # Prefer TMA breakdown for fair traffic if present, else L2 total
    out["fair_l2_traffic_sectors"] = (
        out["tma_traffic_sectors"]
        if out["tma_traffic_sectors"] is not None
        else out["l2_traffic_sectors"]
    )
    out["fair_l2_traffic_mb"] = (
        out["tma_traffic_mb"]
        if out["tma_traffic_mb"] is not None
        else out["l2_traffic_mb"]
    )

    # L1D
    l1_acc = re.findall(r"L1D_cache_core\[\d+\]:\s*Access\s*=\s*(\d+)", content)
    l1_miss = re.findall(
        r"L1D_cache_core\[\d+\]:\s*Access\s*=\s*\d+,\s*Miss\s*=\s*(\d+)", content
    )
    total_l1 = sum(int(v) for v in l1_acc)
    total_l1_miss = sum(int(v) for v in l1_miss)
    out["l1_accesses"] = float(total_l1)
    if total_l1 > 0:
        out["l1_hit_rate_pct"] = 100.0 * (1.0 - total_l1_miss / total_l1)
    else:
        out["l1_hit_rate_pct"] = None  # expected for TMA bypass

    out["validation_passed"] = 1.0 if "Validation PASSED" in content else 0.0
    return out


def parse_ncu_rep(path: str) -> Dict[str, Optional[float]]:
    """Import selected metrics from an .ncu-rep via ncu --csv."""
    try:
        result = subprocess.run(
            [
                "ncu",
                "--import",
                path,
                "--csv",
                "--page",
                "raw",
                "--metrics",
                ",".join(NCU_METRICS),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"[WARN] ncu import failed for {path}: {e}", file=sys.stderr)
        return {}

    if result.returncode != 0:
        # Fallback: full raw page (slower, larger)
        try:
            result = subprocess.run(
                ["ncu", "--import", path, "--csv", "--page", "raw"],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return {}
        if result.returncode != 0:
            print(f"[WARN] ncu import error: {result.stderr[:200]}", file=sys.stderr)
            return {}

    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) < 2:
        return {}

    header = rows[0]
    # units row if present
    data = rows[-1]
    d = dict(zip(header, data))

    def get(name: str) -> Optional[float]:
        v = d.get(name)
        if v is None:
            return None
        v = v.strip().strip('"').replace(",", "")
        if v in ("", "n/a", "N/A"):
            return None
        try:
            return float(v)
        except ValueError:
            return None

    out: Dict[str, Optional[float]] = {}
    out["ncu_cycles"] = get("sm__cycles_elapsed.avg")
    out["ncu_duration_us"] = get("gpu__time_duration.avg")
    out["ncu_sm_freq_ghz"] = get("sm__cycles_elapsed.avg.per_second")

    out["tex_op_read"] = get("lts__t_sectors_srcunit_tex_op_read.sum")
    out["tex_op_write"] = get("lts__t_sectors_srcunit_tex_op_write.sum")
    out["tex_op_atom"] = get("lts__t_sectors_srcunit_tex_op_atom.sum")
    out["tex_lookup_hit"] = get("lts__t_sectors_srcunit_tex_lookup_hit.sum")
    out["tex_lookup_miss"] = get("lts__t_sectors_srcunit_tex_lookup_miss.sum")
    out["tex_sectors_sum"] = get("lts__t_sectors_srcunit_tex.sum")

    # Fair TMA-scoped traffic
    parts = [x for x in (out["tex_op_read"], out["tex_op_write"]) if x is not None]
    if parts:
        out["fair_l2_traffic_sectors"] = sum(parts)
        # atoms are small; include if present for volume completeness
        if out["tex_op_atom"] is not None:
            out["fair_l2_traffic_sectors"] += out["tex_op_atom"]
        out["fair_l2_traffic_mb"] = (
            out["fair_l2_traffic_sectors"] * SECTOR_BYTES / (1024 * 1024)
        )
    else:
        out["fair_l2_traffic_sectors"] = None
        out["fair_l2_traffic_mb"] = None

    # Fair TMA-scoped hit rate
    hits = out["tex_lookup_hit"]
    misses = out["tex_lookup_miss"]
    if hits is not None and misses is not None and (hits + misses) > 0:
        out["fair_l2_hit_rate_pct"] = 100.0 * hits / (hits + misses)
    else:
        out["fair_l2_hit_rate_pct"] = None

    # Unfair whole-L2 (diagnosis only)
    out["unfair_l2_sectors"] = get("lts__t_sectors.sum")
    out["unfair_l2_hit_rate_pct"] = get("lts__t_sector_hit_rate.pct")
    if out["unfair_l2_sectors"] is not None:
        out["unfair_l2_traffic_mb"] = (
            out["unfair_l2_sectors"] * SECTOR_BYTES / (1024 * 1024)
        )
    else:
        out["unfair_l2_traffic_mb"] = None

    out["l1_hit_rate_pct"] = get("l1tex__t_sector_hit_rate.pct")
    out["l1_global_ld_sectors"] = get(
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
    )
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class ShapeResult:
    shape: str
    sim: Dict[str, Optional[float]] = field(default_factory=dict)
    ncu: Dict[str, Optional[float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def cycle_diff_pct(self) -> Optional[float]:
        return _pct_diff(self.sim.get("sim_cycles"), self.ncu.get("ncu_cycles"))

    def l2_traffic_diff_pct(self) -> Optional[float]:
        return _pct_diff(
            self.sim.get("fair_l2_traffic_sectors"),
            self.ncu.get("fair_l2_traffic_sectors"),
        )

    def l2_hit_diff_pp(self) -> Optional[float]:
        """Absolute difference in percentage points (sim - ncu)."""
        return _abs_diff(
            self.sim.get("l2_hit_or_pending_pct"),
            self.ncu.get("fair_l2_hit_rate_pct"),
        )


def discover_shapes(test_name: str) -> List[str]:
    """Shapes that have at least a sim log or ncu rep under results/."""
    results = os.path.join(TRACKING_DIR, test_name, "results")
    shapes = set()
    sim_dir = os.path.join(results, "sim-log")
    ncu_dir = os.path.join(results, "ncu-rep")
    if os.path.isdir(sim_dir):
        for f in os.listdir(sim_dir):
            if f.endswith("_gpgpusim.log"):
                shapes.add(f[: -len("_gpgpusim.log")])
    if os.path.isdir(ncu_dir):
        for f in os.listdir(ncu_dir):
            if f.endswith(".ncu-rep"):
                shapes.add(f[: -len(".ncu-rep")])
    return sorted(shapes)


def load_shape(test_name: str, shape: str) -> ShapeResult:
    results = os.path.join(TRACKING_DIR, test_name, "results")
    sim_path = os.path.join(results, "sim-log", f"{shape}_gpgpusim.log")
    ncu_path = os.path.join(results, "ncu-rep", f"{shape}.ncu-rep")
    # Also accept cold-named reports
    ncu_cold = os.path.join(results, "ncu-rep", f"{shape}_cold.ncu-rep")

    sr = ShapeResult(shape=shape)
    if os.path.isfile(sim_path):
        sr.sim = parse_sim_log(sim_path)
    else:
        sr.notes.append("missing sim log")

    ncu_use = ncu_cold if os.path.isfile(ncu_cold) else ncu_path
    if os.path.isfile(ncu_cold):
        sr.notes.append("using cold NCU report (*_cold.ncu-rep)")
    if os.path.isfile(ncu_use):
        sr.ncu = parse_ncu_rep(ncu_use)
        if not sr.ncu:
            sr.notes.append("ncu parse empty")
    else:
        sr.notes.append("missing ncu-rep")

    # Methodology hints
    if sr.ncu.get("fair_l2_hit_rate_pct") is not None and sr.ncu.get(
        "unfair_l2_hit_rate_pct"
    ) is not None:
        if (
            sr.ncu["fair_l2_hit_rate_pct"] is not None
            and sr.ncu["unfair_l2_hit_rate_pct"] is not None
            and abs(sr.ncu["fair_l2_hit_rate_pct"] - sr.ncu["unfair_l2_hit_rate_pct"])
            > 5
        ):
            sr.notes.append(
                "tex-scoped hit% differs from whole-L2 hit% (warm/unfair risk)"
            )
    return sr


def evaluate(
    sr: ShapeResult,
    cycle_tol: float,
    l2_traffic_tol: float,
    l2_hit_tol: float,
) -> Dict[str, str]:
    """Return pass/fail/skip per metric."""
    status = {}

    c = sr.cycle_diff_pct()
    if c is None:
        status["cycles"] = "SKIP"
    elif abs(c) <= cycle_tol:
        status["cycles"] = "PASS"
    else:
        status["cycles"] = "FAIL"

    t = sr.l2_traffic_diff_pct()
    if t is None:
        status["l2_traffic"] = "SKIP"
    elif abs(t) <= l2_traffic_tol:
        status["l2_traffic"] = "PASS"
    else:
        status["l2_traffic"] = "FAIL"

    h = sr.l2_hit_diff_pp()
    if h is None:
        status["l2_hit"] = "SKIP"
    elif abs(h) <= l2_hit_tol:
        status["l2_hit"] = "PASS"
    else:
        status["l2_hit"] = "FAIL"

    # L1: informational only for TMA
    l1_acc = sr.sim.get("l1_accesses")
    if l1_acc is not None and l1_acc == 0:
        status["l1"] = "OK_BYPASS"
    elif l1_acc is not None and l1_acc > 0:
        status["l1"] = "INFO_ACTIVE"
    else:
        status["l1"] = "SKIP"

    if sr.sim.get("validation_passed") == 1.0:
        status["functional"] = "PASS"
    elif sr.sim:
        status["functional"] = "FAIL"
    else:
        status["functional"] = "SKIP"

    return status


def print_header_rationale():
    print("=" * 78)
    print("FAIR TMA comparison (cycles + L1/L2) — see module docstring for why")
    print("the old extract_metrics.py / whole-L2 NCU path is unfair.")
    print("=" * 78)
    print("CRITICAL NCU flags for fair cache comparison with cold GPGPU-Sim:")
    print("  --cache-control all   # flush GPU caches between passes (REQUIRED)")
    print("  --replay-mode kernel  # preferred for single-kernel harnesses")
    print("  WITHOUT --cache-control all, NCU L2 is warm across multi-pass")
    print("  profiling (~99% hit) while sim is cold (~25% on 128^3 TMA GEMM).")
    print("  Prefer results/ncu-rep/<shape>_cold.ncu-rep when present.")
    print("-" * 78)
    print(
        "Fair L2 traffic : sim TMA_ACC_* (or L2 total) vs "
        "NCU tex_op_read+write[+atom]"
    )
    print(
        "Fair L2 hit rate: sim (access-miss)/access vs "
        "NCU tex_lookup hit/(hit+miss)"
    )
    print("L1              : TMA expected bypass (not in 10% fail band)")
    print("DRAM            : not scored here")
    print("=" * 78)


def print_shape_report(
    sr: ShapeResult,
    status: Dict[str, str],
    cycle_tol: float,
    l2_traffic_tol: float,
    l2_hit_tol: float,
):
    print(f"\n--- {sr.shape} ---")
    if sr.notes:
        print("  notes:", "; ".join(sr.notes))

    sim_c = sr.sim.get("sim_cycles")
    ncu_c = sr.ncu.get("ncu_cycles")
    print(
        f"  Cycles     sim={_f(sim_c, 0):>12}  ncu={_f(ncu_c, 1):>12}  "
        f"diff%={_f(sr.cycle_diff_pct(), 2):>8}  [{status['cycles']}]  "
        f"(tol ±{cycle_tol:g}%)"
    )

    sim_s = sr.sim.get("fair_l2_traffic_sectors")
    ncu_s = sr.ncu.get("fair_l2_traffic_sectors")
    print(
        f"  L2 traffic sim={_f(sim_s, 0):>12}  ncu={_f(ncu_s, 0):>12}  "
        f"sectors  diff%={_f(sr.l2_traffic_diff_pct(), 2):>8}  "
        f"[{status['l2_traffic']}]  (tol ±{l2_traffic_tol:g}%)"
    )
    print(
        f"             sim_mb={_f(sr.sim.get('fair_l2_traffic_mb'), 4)}  "
        f"ncu_mb={_f(sr.ncu.get('fair_l2_traffic_mb'), 4)}  "
        f"(TMA-scoped; NOT lts__t_sectors.sum)"
    )
    if sr.sim.get("tma_acc_r_reqs") is not None:
        print(
            f"             sim TMA_ACC_R={_f(sr.sim.get('tma_acc_r_reqs'), 0)}  "
            f"TMA_ACC_W={_f(sr.sim.get('tma_acc_w_reqs'), 0)}  |  "
            f"ncu tex_R={_f(sr.ncu.get('tex_op_read'), 0)}  "
            f"tex_W={_f(sr.ncu.get('tex_op_write'), 0)}  "
            f"tex_atom={_f(sr.ncu.get('tex_op_atom'), 0)}"
        )

    sim_h = sr.sim.get("l2_hit_or_pending_pct")
    ncu_h = sr.ncu.get("fair_l2_hit_rate_pct")
    print(
        f"  L2 hit%    sim={_f(sim_h, 2):>12}  ncu={_f(ncu_h, 2):>12}  "
        f"diff_pp={_f(sr.l2_hit_diff_pp(), 2):>8}  [{status['l2_hit']}]  "
        f"(tol ±{l2_hit_tol:g} pp)"
    )
    print(
        f"             sim pure_hit%={_f(sr.sim.get('l2_pure_hit_pct'), 2)}  "
        f"pending_hits={_f(sr.sim.get('l2_pending_hits'), 0)}  "
        f"misses={_f(sr.sim.get('l2_misses'), 0)}"
    )
    if sr.ncu.get("unfair_l2_hit_rate_pct") is not None:
        print(
            f"             [unfair old path] whole-L2 hit%="
            f"{_f(sr.ncu.get('unfair_l2_hit_rate_pct'), 2)}  "
            f"whole-L2 sectors={_f(sr.ncu.get('unfair_l2_sectors'), 0)}  "
            f"mb={_f(sr.ncu.get('unfair_l2_traffic_mb'), 4)}"
        )

    print(
        f"  L1         sim_accesses={_f(sr.sim.get('l1_accesses'), 0)}  "
        f"sim_hit%={_f(sr.sim.get('l1_hit_rate_pct'), 2)}  "
        f"ncu_l1tex_hit%={_f(sr.ncu.get('l1_hit_rate_pct'), 2)}  "
        f"ncu_global_ld_sectors={_f(sr.ncu.get('l1_global_ld_sectors'), 0)}  "
        f"[{status['l1']}]"
    )
    print(f"  Functional [{status['functional']}]")


def write_csv(path: str, rows: List[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV written: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fair TMA NCU vs Sim comparison (cycles + L1/L2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "test_name",
        nargs="?",
        default="test_tma_gemm",
        help="Workload under triton_kernel_tracking/ (default: test_tma_gemm)",
    )
    parser.add_argument(
        "--shape",
        action="append",
        default=None,
        help="Shape dir name (e.g. m128_n128_k128). Repeatable. Default: all.",
    )
    parser.add_argument(
        "--cycle-tol",
        type=float,
        default=10.0,
        help="Pass if |cycle diff%%| <= this (default 10)",
    )
    parser.add_argument(
        "--l2-traffic-tol",
        type=float,
        default=10.0,
        help="Pass if |TMA-scoped L2 sector diff%%| <= this (default 10)",
    )
    parser.add_argument(
        "--l2-hit-tol",
        type=float,
        default=10.0,
        help="Pass if |L2 hit rate diff| in percentage points <= this (default 10)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: <results>/fair_comparison.csv)",
    )
    args = parser.parse_args()

    test_name = args.test_name
    results_dir = os.path.join(TRACKING_DIR, test_name, "results")
    if not os.path.isdir(results_dir):
        print(f"ERROR: results dir not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    shapes = args.shape or discover_shapes(test_name)
    if not shapes:
        print("No shapes found with sim-log or ncu-rep.", file=sys.stderr)
        sys.exit(1)

    print_header_rationale()
    print(f"Test: {test_name}")
    print(f"Shapes: {', '.join(shapes)}")
    print(
        f"Tolerances: cycles ±{args.cycle_tol:g}%, "
        f"L2 traffic ±{args.l2_traffic_tol:g}%, "
        f"L2 hit ±{args.l2_hit_tol:g} pp"
    )

    csv_rows = []
    any_fail = False

    for shape in shapes:
        sr = load_shape(test_name, shape)
        status = evaluate(
            sr, args.cycle_tol, args.l2_traffic_tol, args.l2_hit_tol
        )
        print_shape_report(
            sr, status, args.cycle_tol, args.l2_traffic_tol, args.l2_hit_tol
        )
        for k, v in status.items():
            if v == "FAIL" and k in ("cycles", "l2_traffic", "l2_hit", "functional"):
                any_fail = True

        csv_rows.append(
            {
                "shape": shape,
                "sim_cycles": sr.sim.get("sim_cycles"),
                "ncu_cycles": sr.ncu.get("ncu_cycles"),
                "cycle_diff_pct": sr.cycle_diff_pct(),
                "cycles_status": status["cycles"],
                "sim_l2_sectors": sr.sim.get("fair_l2_traffic_sectors"),
                "ncu_l2_sectors_tma": sr.ncu.get("fair_l2_traffic_sectors"),
                "l2_traffic_diff_pct": sr.l2_traffic_diff_pct(),
                "l2_traffic_status": status["l2_traffic"],
                "sim_l2_hit_or_pending_pct": sr.sim.get("l2_hit_or_pending_pct"),
                "sim_l2_pure_hit_pct": sr.sim.get("l2_pure_hit_pct"),
                "ncu_l2_hit_tex_pct": sr.ncu.get("fair_l2_hit_rate_pct"),
                "l2_hit_diff_pp": sr.l2_hit_diff_pp(),
                "l2_hit_status": status["l2_hit"],
                "sim_l1_accesses": sr.sim.get("l1_accesses"),
                "l1_status": status["l1"],
                "ncu_unfair_l2_sectors": sr.ncu.get("unfair_l2_sectors"),
                "ncu_unfair_l2_hit_pct": sr.ncu.get("unfair_l2_hit_rate_pct"),
                "functional": status["functional"],
                "notes": "; ".join(sr.notes),
            }
        )

    out_path = args.output or os.path.join(results_dir, "fair_comparison.csv")
    write_csv(out_path, csv_rows)

    print("\n" + "=" * 78)
    if any_fail:
        print("OVERALL: FAIL (one or more scored metrics outside tolerance)")
        sys.exit(2)
    print("OVERALL: PASS (all scored metrics within tolerance or SKIP)")
    sys.exit(0)


if __name__ == "__main__":
    main()
