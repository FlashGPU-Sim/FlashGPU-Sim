#!/usr/bin/env python3
"""H200 CLUSTER132 calibration suite (optional; not default CI).

Historical script name. This is not the deleted hop-delay DSM path.
Kernels live in the git-ignored calibration/ tree. Populate with:

    bash scripts/sync_calibration_kernels.sh

Functional cluster / DSM / TMA tests: ./test/run_tests.sh on
SM120_RTX5090_REDUCED_CLUSTER4x4 (or 2x1 / 2x2).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from compare_h200_calibration import dsm_fit_metrics, tma_latency_metrics


ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "calibration/kernels/h200_probes"
MICRO = PROBES / "vendor/microbench/bin/flashgpu"
TMA = PROBES / "vendor/tma_bw"
DSM = PROBES / "vendor/dsm_bw"
RUNNER = ROOT / "scripts/run_calibration_sim.sh"
FAIL_RE = re.compile(
    r"^\[\s*(?:FAILED|SKIPPED)\s*\]|^(?:FAIL|SKIP|TIMEOUT):|^\s*gemm_\S+\s+FAIL\b|validation failed|CUDA error|result mismatch|deadlock detected|aborted|timed launch timeout",
    re.IGNORECASE | re.MULTILINE,
)
LIMIT_RE = re.compile(r"break due to reaching the maximum cycles", re.IGNORECASE)
CSV_HEADER = ("suite", "metric", "sim_knob", "median", "p90", "mean", "unit", "notes")


@dataclass(frozen=True)
class Case:
    name: str
    group: str
    origin: str
    source: str
    binary: Path
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    timeout: int = 900
    note: str = ""
    max_cycles: int = 1_000_000


def hcase(name: str, suite: str, *extra: str, timeout: int = 900, env=(), note="", max_cycles=1_000_000) -> Case:
    source = {
        "hbm": "src/old/probe_l2_dram_OLD.cu",
        "dsm_calibration": "src/probe_dsm.cu (includes probe_dsm_l23.cu)",
        "gemm_compare": "src/probe_gemm_triton.cu",
    }.get(suite, f"src/probe_{suite}.cu")
    # Job 2119329 GEMMs use one immediate same-variant warmup; smoke is not equivalent.
    warmup = "1" if suite == "gemm_compare" else "0"
    return Case(
        name, "h200_probes", "H200_profiling", source, PROBES / "nvprof",
        ("--suite", suite, "--samples", "1", "--warmup", warmup, "--csv", "yes", *extra),
        (("FLASHGPU_DEVICE_ITERS", "1000"), *env), timeout, note, max_cycles,
    )


def mcase(name: str, rel: str, args: tuple[str, ...], env=(), note="") -> Case:
    suffix = ".cc" if any(x in rel for x in ("inst_latency", "mma_issue", "tma_latency", "mbarrier", "wgmma")) else ".cu"
    return Case(name, "flashgpu_microbench", "FlashGPU-Sim exact mirror", f"vendor/microbench/flashgpu_sim/{rel}{suffix}", MICRO / rel, args, env, note=note)


def cases(profile: str) -> list[Case]:
    dsm_args = ("--suite", "all" if profile == "exhaustive" else "bandwidth",
                "--warmups", "0", "--iterations", "1", "--output", "sim_dsm.jsonl")
    out = [
        hcase("device", "device"),
        Case("vendor_dsm", "vendor_dsm", "vendor", "vendor/dsm_bw/benchmark.cu", DSM / "dsm_h200.out", dsm_args, timeout=7200, max_cycles=10_000_000,
             note="Vendor DSM size slopes, mixed BW7/BW8 traffic, and TMA scaling; one sample, no redundant warmup."),
        Case("vendor_tma_normal", "vendor_tma", "vendor", "vendor/tma_bw/simple_normal_load_test.cu", TMA / "simple_normal_load_test.out", timeout=3600, note="Normal-load HBM winner: 132 blocks, 1024 threads, stages=16, chunk=8192 B; hardware-matched 1 GiB, one sample."),
        Case("vendor_tma_cp_async", "vendor_tma", "vendor", "vendor/tma_bw/simple_cp_async_test.cu", TMA / "simple_cp_async_test.out", timeout=3600, note="cp.async HBM winner: 132 blocks, 1024 threads, stages=16, chunk=8192 B; hardware-matched 1 GiB, one sample."),
        Case("vendor_tma_tma", "vendor_tma", "vendor", "vendor/tma_bw/simple_tma_test.cu", TMA / "simple_tma_test.out", timeout=3600, note="TMA HBM winner: 132 blocks, 1024 threads, stages=16, chunk=8192 B; hardware-matched 1 GiB, one sample."),
    ]
    out += [
        mcase("cp_async_latency", "cp_async/cp_async_latency_bench", ("--blocks=1", "--threads=128", "--iters=1", "--warmup=0", "--tiles=4", "--mode=fa2-issue", "--fa2-stages=2", "--active-warps=1", "--active-lanes=32", "--smem-slots=64", "--global-slots=64", "--block-hot")),
        mcase("cp_async_ptx", "cp_async/cp_async_ptx_bench", ("--blocks=1", "--threads=32", "--active-warps=1", "--iters=1", "--warmup=0", "--groups=1", "--copies=1", "--bytes=16", "--mode=issue")),
        mcase("cp_async_issue_scope", "cp_async/cp_async_issue_scope_bench", ("--blocks=1", "--threads=32", "--active-warps=1", "--iters=1", "--warmup=0", "--chains=1")),
        mcase("global_load_bw", "memory/global_load_bw_bench", ("--blocks=1", "--threads=256", "--iters=1", "--warmup=0", "--tiles=1", "--block-hot")),
        mcase("l2_latency", "memory/l2_latency_bench", ("--case=l2-hit", "--bytes=4096", "--steps=4", "--samples=1", "--warmup-steps=0", "--cache-op=cg")),
        mcase("l2_hbm_interleave", "memory/l2_hbm_interleave_bench", ("--op=ldg", "--pattern=stream", "--blocks=1", "--threads=32", "--iters=1", "--warmup=0", "--repeat=1", "--data-bytes=64K", "--tile-bytes=4K", "--smem-bytes=4K", "--event")),
        mcase("l2_partition", "memory/l2_partition_latency_probe", ("--blocks=1", "--threads=32", "--repeats=1", "--warmup=0", "--bit-first=7", "--bit-last=7", "--dense-offsets=1", "--data-bytes=1M", "--powers-only")),
        mcase("mma_accept_queue", "mma/mma_accept_queue_bench", ("--samples", "1", "--warmup", "0", "--bench", "accept", "--n", "1", "--modes", "mma", "--warps", "1")),
        mcase("mma_saturation", "mma/mma_saturation_bench", ("--warps", "1", "--unroll", "4", "--repeat", "1", "--blocks", "1", "--warmup", "0", "--samples", "1")),
        mcase("tma_completion", "tma/tma_completion_latency_bench", ("--blocks=1", "--iters=1", "--warmup=0", "--tiles=64", "--case=pair48", "--issue=llama", "--stride")),
        mcase("tma_descriptor", "tma/tma_descriptor_setup_bench", ("--blocks=1", "--iters=1", "--warmup=0", "--tiles=64", "--case=full", "--stride")),
        mcase("tma_fa3_m3", "tma/tma_fa3_m3_bench", ("--blocks=1", "--iters=1", "--warmup=0", "--batch=1", "--heads=1", "--seqlen=512", "--m-block=3", "--sweep-hb=0")),
        mcase("mma_inst_latency", "mma/inst_latency_bench", ("--gtest_color=no", "--gtest_filter=InstLatencyTest.FullCalibrationSuite"), note="Full internal instruction sweep; covered functionally by the single-case MMA probes."),
        mcase("mma_issue", "mma/mma_issue_bench", ("--gtest_color=no", "--gtest_filter=MMAIssueTest/0.ILPMinimal")),
        mcase("tma_latency_256b", "tma/tma_latency_bench", ("--gtest_color=no", "--gtest_filter=TMALatencyTest.TMALoad256B"), note="Vendor TMA load, 256 B end-to-end latency."),
        mcase("tma_latency_1k", "tma/tma_latency_bench", ("--gtest_color=no", "--gtest_filter=TMALatencyTest.TMALoad1KB"), note="Vendor TMA load, 1 KiB end-to-end latency."),
        mcase("tma_latency_4k", "tma/tma_latency_bench", ("--gtest_color=no", "--gtest_filter=TMALatencyTest.TMALoad4KB"), note="Vendor TMA load, 4 KiB end-to-end latency."),
        mcase("tma_latency_16k", "tma/tma_latency_bench", ("--gtest_color=no", "--gtest_filter=TMALatencyTest.TMALoad16KB"), note="Vendor TMA load, 16 KiB end-to-end latency."),
        mcase("mbarrier_trywait", "mbarrier/mbarrier_trywait_latency_bench", ("--gtest_color=no", "--gtest_filter=MBarrierLatencyTest.WaitFalse")),
        mcase("wgmma_async", "wgmma/wgmma_async_latency_bench", ("--gtest_color=no", "--gtest_filter=WgmmaAsyncLatencyBench.Fa3LikeSelectedKernel"), (("WGMMA_FA3_PROBE_BLOCKS", "1"), ("WGMMA_FA3_PROBE_ROUNDS", "1"), ("WGMMA_FA3_PROBE_WARMUP", "0"), ("WGMMA_FA3_PROBE_SELECTED", "kt1_sm0"))),
    ]
    for kind in ("ss", "rs"):
        for group in (1, 2, 4):
            out.append(mcase(f"wgmma_{kind}_g{group}", f"wgmma/wgmma_fp16_{kind}_g{group}_bench", ("--gtest_color=no", "--gtest_filter=WgmmaFp16CoreSweep.Selected"), (("WGMMA_FP16_SWEEP_BLOCKS", "1"), ("WGMMA_FP16_SWEEP_ROUNDS", "1"), ("WGMMA_FP16_SWEEP_WARMUP_ROUNDS", "0"), ("WGMMA_FP16_SWEEP_FILTER", f"n64_g{group}_o1_same"))))
    out += [
        mcase("wgmma_n16", "wgmma/wgmma_n16_chain_bench", ("--gtest_color=no", "--gtest_filter=WgmmaN16ChainBench.Selected"), (("WGMMA_N16_CHAIN_BLOCKS", "1"), ("WGMMA_N16_CHAIN_ROUNDS", "1"), ("WGMMA_N16_CHAIN_SELECTED", "chain11_k1"))),
        mcase("wgmma_rf", "wgmma/wgmma_rf_bandwidth_bench", ("--gtest_color=no", "--gtest_filter=WgmmaRfBandwidth.Sweep"), (("WGMMA_RF_BW_BLOCKS", "1"), ("WGMMA_RF_BW_ROUNDS", "1"), ("WGMMA_RF_BW_WARMUP_ROUNDS", "0"), ("WGMMA_RF_BW_WGMMA_OPS", "1"), ("WGMMA_RF_BW_OPERANDS", "ss"), ("WGMMA_RF_BW_ACC_MODES", "accumulate"), ("WGMMA_RF_BW_SHAPES", "256"), ("WGMMA_RF_BW_MODES", "wgmma_only"), ("WGMMA_RF_BW_MATH_KINDS", "fma_indep8"), ("WGMMA_RF_BW_MATH_OPS", "0"))),
        mcase("wgmma_softmax", "wgmma/wgmma_softmax_mix_bench", ("--gtest_color=no", "--gtest_filter=WgmmaSoftmaxMixBench.Selected"), (("WGMMA_SOFTMAX_MIX_BLOCKS", "1"), ("WGMMA_SOFTMAX_MIX_ROUNDS", "1"), ("WGMMA_SOFTMAX_MIX_WARMUP", "0"), ("WGMMA_SOFTMAX_MIX_QK_OPS", "1"), ("WGMMA_SOFTMAX_MIX_PV_OPS", "1"))),
        hcase("hbm", "hbm", timeout=1800, env=(("FLASHGPU_HBM_KIB", "4"), ("FLASHGPU_HBM_REPS", "1"), ("FLASHGPU_HBM_BLOCKS", "1")), note="Functional STREAM path only: 4 KiB, one block, one repetition; not an HBM bandwidth estimate."),
        hcase("dsm_calibration", "dsm_calibration", "--dsm-cluster", "16", timeout=7200, max_cycles=8_000_000,
              note="DSM latency matrix, dependent round trip, store visibility and contention; cluster=16 matches job 2119329; no matrix warmup or bandwidth sweep."),
        hcase("tma_multicast", "tma_multicast", "--dsm-cluster", "8", env=(("FLASHGPU_TMA_MC_REPRESENTATIVE", "1"),), note="H200_profiling TMA unicast/multicast latency and correctness; cluster=8 matches job 2119329. Multicast must not use dsm_fabric."),
        hcase("cycle_gate", "cycle_gate", timeout=1800, note="Fixed-loop calibration gate; run separately when fitting cycle-scale knobs."),
        hcase("mbarrier", "mbarrier"),
        hcase("gemm_cyc_1e5", "gemm_compare", "--gemm-smoke", timeout=7200, max_cycles=3_000_000,
              env=(("FLASHGPU_GEMM_CASE", "cyc_1e5"),),
              note="G1/G2/G3 at M=256 N=8192 K=2048; job 2119329: 108-289K cycles."),
        # At OMP=4 the measured host cost is roughly 3-6 ms/sim cycle.
        # Larger shapes include smoke + timed launches of all three variants;
        # retain cycle caps but allow enough host time for useful completion.
        hcase("gemm_k2k", "gemm_compare", "--gemm-smoke", timeout=14400, max_cycles=5_000_000,
              env=(("FLASHGPU_GEMM_CASE", "k2k"),),
              note="G1/G2/G3 at M=512 N=16384 K=2048; 256 clusters; job 2119329: 207-892K cycles."),
        hcase("gemm_sq_2k", "gemm_compare", "--gemm-smoke", timeout=21600, max_cycles=8_000_000,
              env=(("FLASHGPU_GEMM_CASE", "sq_2k"),),
              note="G1/G2/G3 at M=N=2048 K=8192; required square control."),
    ]
    return out


def shell(command: str) -> None:
    print(f"[build] {command.split('&&')[-1].strip()}", flush=True)
    if subprocess.run(["bash", "-lc", f"source setup.sh && source setup_environment && {command}"], cwd=ROOT).returncode:
        raise SystemExit("build failed")


def build() -> None:
    if not PROBES.is_dir():
        raise SystemExit(
            "missing git-ignored calibration/kernels/h200_probes.\n"
            "Populate with: bash scripts/sync_calibration_kernels.sh\n"
            "(optional sibling trees: H200_PROFILING_DIR, HOPPER_BENCH_DIR)"
        )
    # Link the versioned toolkit runtime; setup_environment substitutes the
    # simulator runtime at execution and uses the version tag for PTX setup.
    common = "-std=c++17 -O2 -lineinfo -arch=sm_90a -cudart shared"
    shell('make FLASH=1 -j"$(nproc)"')
    shell(f"make -C {shlex.quote(str(PROBES))} NVCCFLAGS={shlex.quote(common + ' -Iinclude')}")
    shell(f"make -C {shlex.quote(str(PROBES / 'vendor/dsm_bw'))} NVCC_FLAGS={shlex.quote(common)}")
    shell(f"make -C {shlex.quote(str(PROBES / 'vendor/microbench'))} flashgpu-all FLASHGPU_FLAGS={shlex.quote(common + ' -DFLASHGPU_SIM_REPRESENTATIVE --compiler-options -pthread -Igtest/googletest/include -Igtest/googletest -Iflashgpu_sim')}")
    shell(f"make -C {shlex.quote(str(TMA))} simple_normal_load_test.out simple_cp_async_test.out simple_tma_test.out NVCC_FLAGS={shlex.quote(common + ' -DFLASHGPU_SIM_REPRESENTATIVE')}")


def parse_metrics(text: str, case: str) -> list[dict[str, str]]:
    found = []
    rows = iter(csv.reader(text.splitlines()))
    for row in rows:
        if tuple(row) != CSV_HEADER:
            continue
        for row in rows:
            if tuple(row) == CSV_HEADER:
                continue
            if len(row) < len(CSV_HEADER):
                break
            try:
                for value in row[3:6]:
                    if value:
                        float(value)
            except ValueError:
                break
            row = row[:7] + [",".join(row[7:])]
            found.append(dict(zip(CSV_HEADER, row), case=case))
    cycles = re.findall(r"gpu_sim_cycle\s*=\s*(\d+)", text)
    if cycles:
        found.append({"case": case, "suite": "simulator", "metric": "gpu_sim_cycle", "sim_knob": "", "median": cycles[-1], "p90": cycles[-1], "mean": cycles[-1], "unit": "cycles", "notes": "simulator total"})
    return found


def vendor_hbm_metric(text: str, case: str) -> dict[str, str] | None:
    cycles = re.findall(r"gpu_sim_cycle\s*=\s*(\d+)", text)
    size = re.search(r"Passed .*?Data=([0-9.]+) MiB", text)
    if not cycles or not size:
        return None
    bpc = float(size.group(1)) * (1 << 20) / int(cycles[-1])
    metric = {"vendor_tma_normal": "normal_load", "vendor_tma_cp_async": "cp_async",
              "vendor_tma_tma": "tma"}[case]
    value = f"{bpc:.6f}"
    return {"case": case, "suite": "hbm", "metric": metric,
            "sim_knob": "HBM aggregate", "median": value, "p90": value,
            "mean": value, "unit": "bytes/cycle",
            "notes": f"{size.group(1)} MiB stream; gpu_sim_cycle, not CUDA-event wall time"}


def sha256(path: Path) -> str:
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def simulator_fingerprints() -> dict[str, str]:
    paths = set(ROOT.glob("lib/**/libcudart.so"))
    paths.update(Path(directory) / "libcudart.so"
                 for directory in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
                 if directory)
    return {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): sha256(path)
            for path in sorted({p.resolve() for p in paths if p.is_file()})}


def case_fingerprint(case: Case, config: str, max_cycles: int) -> str:
    material = {
        "case": asdict(case), "binary_sha256": sha256(case.binary),
        "config": config, "config_sha256": sha256(ROOT / "configs" / config / "gpgpusim.config"),
        "runner_sha256": sha256(RUNNER), "max_cycles": max_cycles, "omp_threads": 4,
        "simulator_sha256": simulator_fingerprints(),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "harness_sha256": sha256(Path(__file__)),
        "comparison_sha256": sha256(ROOT / "scripts/compare_h200_calibration.py"),
    }
    material["case"]["binary"] = str(case.binary)
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


def compress_log(path: Path) -> Path:
    target = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as source, gzip.open(target, "wb") as destination:
        shutil.copyfileobj(source, destination)
    path.unlink()
    return target


def classify_result(rc: int, timed_out: bool, text: str, metrics: list[dict]) -> str:
    if timed_out:
        return "TIMEOUT"
    if LIMIT_RE.search(text):
        return "LIMIT"
    if rc or FAIL_RE.search(text):
        return "FAIL"
    for row in metrics:
        if row.get("sim_knob") in {
            "gemm_correctness", "gemm_tma_correctness", "gemm_notma_correctness"
        }:
            try:
                if float(row["median"]) == 1.0:
                    continue
            except (KeyError, TypeError, ValueError):
                pass
            return "FAIL"
    return "PASS"


def run_case(case: Case, config: str, result_dir: Path, max_cycles: int, fingerprint: str) -> tuple[dict, list[dict]]:
    simulator_libraries = simulator_fingerprints()
    log = result_dir / "logs" / f"{case.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(case.env)
    env["OMP_NUM_THREADS"] = "4"
    env["CALIB_TIMEOUT_S"] = str(case.timeout)
    env["CALIB_MAX_CYCLES"] = str(max_cycles)
    started = time.monotonic()
    artifacts, csv_metrics = [], []
    with tempfile.TemporaryDirectory(prefix=f"flashgpu-{case.name}-") as run_dir:
        command = [str(RUNNER), "--config", config, "--run-dir", run_dir, "--", str(case.binary), *case.args]
        with log.open("w") as handle:
            handle.write(f"case: {case.name}\nsource: {case.source}\nfingerprint: {fingerprint}\ncommand: {shlex.join(command)}\n\n")
            handle.write(f"simulator libraries SHA-256: {json.dumps(simulator_libraries, sort_keys=True)}\n")
            handle.write(f"LD_LIBRARY_PATH: {env.get('LD_LIBRARY_PATH', '')}\n")
            handle.flush()
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = process.wait(timeout=case.timeout + 30)
                timed_out = rc in (124, 137)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                rc, timed_out = 124, True
        dsm_json = Path(run_dir) / "sim_dsm.jsonl"
        if case.group == "vendor_dsm" and rc == 0 and dsm_json.is_file():
            processed = subprocess.run(
                [sys.executable, str(DSM / "process.py"), "--input", str(dsm_json),
                 "--output-dir", str(Path(run_dir) / "dsm_analysis")]
            )
            if processed.returncode:
                rc = processed.returncode
        for source in (*Path(run_dir).rglob("*.csv"), *Path(run_dir).rglob("*.jsonl")):
            if source.name.startswith("dsm_hop_csv_"):
                continue
            destination = result_dir / ("csv" if source.suffix == ".csv" else "raw") / f"{case.name}__{'__'.join(source.relative_to(run_dir).parts)}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            artifacts.append(str(destination.relative_to(result_dir)))
            if source.suffix == ".csv":
                csv_metrics.extend(parse_metrics(source.read_text(errors="replace"), case.name))
                if case.group == "vendor_dsm" and source.name == "fits.csv":
                    csv_metrics.extend(dict(row, case=case.name) for row in
                                       dsm_fit_metrics(source.read_text()))
                if source.name.startswith("TMALatencyTest."):
                    csv_metrics.extend(dict(row, case=case.name) for row in
                                       tma_latency_metrics(source.read_text()))
    duration = round(time.monotonic() - started, 3)
    text = log.read_text(errors="replace")
    metrics = csv_metrics + parse_metrics(text, case.name)
    status = classify_result(rc, timed_out, text, metrics)
    if status == "PASS" and case.group == "vendor_tma":
        metric = vendor_hbm_metric(text, case.name)
        if metric:
            metrics.append(metric)
    log = compress_log(log)
    record = {**asdict(case), "max_cycles": max_cycles, "binary": str(case.binary.relative_to(ROOT)), "fingerprint": fingerprint, "simulator_sha256": simulator_libraries, "status": status, "returncode": rc, "duration_s": duration, "log": str(log.relative_to(result_dir)), "artifacts": artifacts}
    return record, metrics


def run_metadata(config: str, max_cycles: int) -> dict:
    manifest = PROBES / "SOURCE_MANIFEST.json"
    return {
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip(),
        "config_sha256": sha256(ROOT / "configs" / config / "gpgpusim.config"),
        "harness_sha256": sha256(Path(__file__)), "launcher_sha256": sha256(RUNNER),
        "simulator_sha256": simulator_fingerprints(),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
        "max_cycles": max_cycles, "omp_threads": 4,
        "kernel_sources": json.loads(manifest.read_text()) if manifest.is_file() else {},
    }


def write_reports(result_dir: Path, config: str, profile: str, records: list[dict], metrics: list[dict], metadata: dict) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    summary = {s: sum(r["status"] == s for r in records) for s in ("PASS", "FAIL", "TIMEOUT", "LIMIT", "SKIP")}
    payload = {"generated": datetime.now().astimezone().isoformat(), "config": config, "profile": profile, "run": metadata, "summary": summary, "cases": records, "metrics": metrics}
    (result_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (result_dir / "results.csv").open("w", newline="") as handle:
        fields = ("name", "group", "origin", "source", "fingerprint", "status", "returncode", "duration_s", "max_cycles", "log", "note")
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)
    with (result_dir / "metrics.csv").open("w", newline="") as handle:
        fields = ("case", *CSV_HEADER)
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(metrics)
    lines = [
        "# FlashGPU-Sim H200 Calibration Suite", "", f"Generated: {payload['generated']}",
        f"Configuration: `{config}`", f"Workload profile: `{profile}`",
        f"Git commit: `{metadata['git_commit']}`; config SHA-256: `{metadata['config_sha256']}`; cycle ceiling: {metadata['max_cycles'] or 'case defaults'}; OMP threads: {metadata['omp_threads']}",
        f"Outcome: **{summary['PASS']} passed, {summary['FAIL']} failed, {summary['TIMEOUT']} timed out, {summary['LIMIT']} reached the cycle limit, {summary['SKIP']} skipped**", "",
        "Hardware targets are external inputs; this simulator report does not claim hardware calibration acceptance by itself.", "",
        "The suite mirrors the H200 profiling kernel families. Representative mode reduces samples and warmups, and uses the vendor single-point drivers instead of repeating the L2/HBM capacity sweep.", "",
        "| Case | Family | Status | Seconds | Cycle ceiling | Source |", "|---|---|---:|---:|---:|---|",
    ]
    lines += [f"| {r['name']} | {r['group']} | {r['status']} | {r['duration_s']} | {r['max_cycles']} | `{r['source']}` |" for r in records]
    failures = [r for r in records if r["status"] in ("FAIL", "TIMEOUT", "LIMIT")]
    if failures:
        lines += ["", "## Items requiring attention", ""] + [f"- `{r['name']}`: {r['status']} (see `{r['log']}`)." for r in failures]
    skipped = [r for r in records if r["status"] == "SKIP"]
    if skipped:
        lines += ["", "## Intentionally skipped", ""] + [f"- `{r['name']}`: {r['note']}" for r in skipped]
    lines += ["", "## Artifacts", "", "- `results.csv`: case status and runtime", "- `results.json`: complete structured result", "- `metrics.csv`: extracted canonical calibration metrics", "- `csv/`: raw benchmark CSV outputs", "- `raw/`: non-CSV benchmark output such as vendor DSM JSONL", "- `logs/*.log.gz`: compressed full case logs", ""]
    report = "\n".join(lines)
    (result_dir / "report.md").write_text(report)
    with (result_dir / "suite.log").open("w") as handle:
        handle.write(report + "\nRaw case output is compressed under logs/*.log.gz.\n")


def self_test() -> None:
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "repo"
        library = root / "lib/test/libcudart.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"before rebuild")
        with patch.dict(globals(), ROOT=root):
            before = simulator_fingerprints()
            library.write_bytes(b"after rebuild")
            assert simulator_fingerprints() != before, "library rebuild must invalidate fingerprints"
            external = Path(directory) / "candidate/libcudart.so"
            external.parent.mkdir()
            external.write_bytes(b"external candidate")
            with patch.dict(os.environ, LD_LIBRARY_PATH=str(external.parent)):
                before = simulator_fingerprints()
                assert before[str(external)] == sha256(external)
                external.write_bytes(b"rebuilt candidate")
                assert simulator_fingerprints() != before, "external rebuild must invalidate fingerprints"
                case = cases("representative")[0]
                os.environ["LD_LIBRARY_PATH"] = f"{external.parent}:{library.parent}"
                before = case_fingerprint(case, "test", 1)
                os.environ["LD_LIBRARY_PATH"] = f"{library.parent}:{external.parent}"
                assert case_fingerprint(case, "test", 1) != before, "loader search order must invalidate fingerprints"
    sample = "x,y,z,1,2,1.5,cycles,note\n" + ",".join(CSV_HEADER) + "\ndemo,latency,-knob,1,2,1.5,cycles,note\ngpu_sim_cycle = 99\n"
    selected = cases("representative")
    assert len(selected) == 42
    assert [c.name for c in selected if c.name.startswith("gemm_")] == ["gemm_cyc_1e5", "gemm_k2k", "gemm_sq_2k"]
    assert [c.max_cycles for c in selected if c.name.startswith("gemm_")] == [3_000_000, 5_000_000, 8_000_000]
    assert [c.timeout for c in selected if c.name.startswith("gemm_")] == [7200, 14400, 21600]
    for case in selected:
        if case.group == "h200_probes":
            expected = "1" if case.args[1] == "gemm_compare" else "0"
            assert case.args[case.args.index("--warmup") + 1] == expected
    assert next(c for c in selected if c.name == "vendor_dsm").args[:2] == ("--suite", "bandwidth")
    assert len(parse_metrics(sample, "demo")) == 2
    hbm = vendor_hbm_metric("gpu_sim_cycle = 1024\nPassed | Data=1.00 MiB", "vendor_tma_tma")
    assert hbm and hbm["median"] == "1024.000000"
    assert hbm["notes"].startswith("1.00 MiB")
    assert FAIL_RE.search("Validation FAILED") and FAIL_RE.search("SKIP: unsupported") and FAIL_RE.search("TIMEOUT: launch")
    for metric, knob in (("gemm_reference_small", "gemm_correctness"),
                         ("gemm_c_match_small", "gemm_tma_correctness"),
                         ("gemm_notma_c_match_small", "gemm_notma_correctness")):
        if classify_result(0, False, f"  {metric:44s} FAIL  max_abs_err=inf", []) != "FAIL":
            raise AssertionError(f"missed GEMM failure log: {metric}")
        for value in ("0", "nan", "inf", "", "bad", None):
            if classify_result(0, False, "", [{"sim_knob": knob, "median": value}]) != "FAIL":
                raise AssertionError(f"accepted invalid correctness metric: {knob}={value}")
        if classify_result(0, False, "", [{"sim_knob": knob, "median": "1"}]) != "PASS":
            raise AssertionError(f"rejected valid correctness metric: {knob}")
    if classify_result(0, False, "FAIL: GEMM validation small", []) != "FAIL":
        raise AssertionError("missed explicit GEMM failure")
    print("self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="SM90_H200_CLUSTER132")
    parser.add_argument("--profile", choices=("representative", "exhaustive"), default="representative")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "calibration/results/full_scale_calibration_final")
    parser.add_argument("--only", help="Comma-separated case names or groups")
    parser.add_argument("--exclude", default="cycle_gate,dsm_calibration,mma_inst_latency,mbarrier_trywait,gemm_sq_2k", help="Comma-separated cases/groups to report as SKIP (default: long sweeps, deferred sq_2k, and the native-GPU-only vendor mbarrier test; use 'none' to run all)")
    parser.add_argument("--max-cycles", type=int, help="Override cumulative per-process ceiling (default: 1M; DSM 10M; GEMM 3M/5M/8M)")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Keep prior passing cases and rerun the rest")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.max_cycles is not None and args.max_cycles <= 0:
        parser.error("--max-cycles must be positive")
    if args.self_test:
        self_test(); return 0
    selected = cases(args.profile)
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        selected = [c for c in selected if c.name in wanted or c.group in wanted]
        missing = wanted - {x for c in selected for x in (c.name, c.group)}
        if missing: parser.error(f"unknown case or group: {', '.join(sorted(missing))}")
    excluded = set() if args.exclude == "none" else {x.strip() for x in args.exclude.split(",") if x.strip()}
    if args.list:
        for case in selected: print(f"{case.name:24} {case.group:22} {case.source}")
        return 0
    if not args.skip_build: build()
    metadata = run_metadata(args.config, args.max_cycles)
    prior, prior_metrics = {}, []
    prior_file = args.results_dir / "results.json"
    if args.resume and prior_file.exists():
        payload = json.loads(prior_file.read_text())
        prior = {r["name"]: r for r in payload.get("cases", []) if r["status"] == "PASS"}
    records, metrics = [], []
    for index, case in enumerate(selected, 1):
        max_cycles = args.max_cycles if args.max_cycles is not None else case.max_cycles
        fingerprint = case_fingerprint(case, args.config, max_cycles)
        if case.name in excluded or case.group in excluded:
            (args.results_dir / "logs" / f"{case.name}.log").unlink(missing_ok=True)
            (args.results_dir / "logs" / f"{case.name}.log.gz").unlink(missing_ok=True)
            record = {**asdict(case), "binary": str(case.binary.relative_to(ROOT)), "fingerprint": fingerprint, "status": "SKIP", "returncode": "", "duration_s": 0, "log": "", "artifacts": [], "note": case.note or "Excluded as simulator-expensive; run with --exclude none or a narrower list."}
            records.append(record); print(f"[{index}/{len(selected)}] {case.name}: SKIP", flush=True); continue
        if case.name in prior and prior[case.name].get("fingerprint") == fingerprint:
            print(f"[{index}/{len(selected)}] {case.name}: PASS (resumed)", flush=True)
            saved = prior[case.name]
            saved_log = args.results_dir / saved["log"]
            if saved_log.is_file() and saved_log.suffix != ".gz":
                saved["log"] = str(compress_log(saved_log).relative_to(args.results_dir))
            records.append({**asdict(case), "binary": str(case.binary.relative_to(ROOT)),
                            **{key: saved[key] for key in ("fingerprint", "status", "returncode", "duration_s", "log", "max_cycles")},
                            "simulator_sha256": saved.get("simulator_sha256", {}),
                            "artifacts": saved.get("artifacts", [])})
            metrics.extend(m for m in payload.get("metrics", []) if m["case"] == case.name)
            continue
        if not case.binary.is_file():
            record = {**asdict(case), "binary": str(case.binary), "fingerprint": fingerprint, "status": "FAIL", "returncode": 127, "duration_s": 0, "log": "", "artifacts": [], "note": "binary missing; build the suite"}
            records.append(record); print(f"[{index}/{len(selected)}] {case.name}: FAIL (missing binary)", flush=True); continue
        print(f"[{index}/{len(selected)}] {case.name}: running", flush=True)
        record, found = run_case(case, args.config, args.results_dir, max_cycles, fingerprint)
        records.append(record); metrics.extend(found)
        print(f"[{index}/{len(selected)}] {case.name}: {record['status']} ({record['duration_s']}s)", flush=True)
        write_reports(args.results_dir, args.config, args.profile, records, metrics, metadata)
    write_reports(args.results_dir, args.config, args.profile, records, metrics, metadata)
    return int(any(r["status"] in ("FAIL", "TIMEOUT", "LIMIT") for r in records))


if __name__ == "__main__":
    sys.exit(main())
