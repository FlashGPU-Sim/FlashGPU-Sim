#!/usr/bin/env python3
"""Detached large-forward regression, four pinned slots and shortest-S-first queue.

Run after sourcing setup_environment release with CUDA_INSTALL_PATH=cuda-13.3.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time


def skip_completed_case(path, resume):
    """Resume only completed cases; never adopt, signal or overwrite a process."""
    if not resume or not path.exists():
        return False
    previous = json.loads(path.read_text())
    if previous.get("state") == "done" and previous.get("rc") == 0:
        return True
    raise RuntimeError(f"Refusing to resume unfinished/failed case: {path}; use a new output directory")


def freeze_library(source, out, resume):
    directory = out / "simulator"
    snapshot = directory / "libcudart.so.13"
    identity = directory / "sha256.txt"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if resume:
        if not identity.is_file() or not snapshot.is_file():
            raise RuntimeError("Resume has no frozen simulator library; use a new output directory")
        expected = identity.read_text().strip()
        if digest != expected or hashlib.sha256(snapshot.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Resume simulator library mismatch; use a new output directory")
    else:
        directory.mkdir()
        shutil.copyfile(source, snapshot)
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Simulator library changed while being copied")
        snapshot.chmod(0o444)
        for name in ("libcudart.so", "libcuda.so", "libcuda.so.1"):
            (directory / name).symlink_to(snapshot.name)
        identity.write_text(digest + "\n")
    return directory, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True,
                        help="case_cycles_summary.csv collected with the same binary")
    parser.add_argument("--profile", choices=("medium", "large"), default="large")
    parser.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda-13.3"))
    parser.add_argument("--library", type=Path,
                        help="simulator libcudart.so (default: sourced GPGPUSIM_CONFIG build)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if os.environ.get("LD_PRELOAD"):
        parser.error("unset LD_PRELOAD so it cannot override the frozen simulator")
    root = Path(__file__).resolve().parents[2]
    library = args.library
    if library is None:
        if not os.environ.get("GPGPUSIM_CONFIG"):
            parser.error("source setup_environment release or provide --library")
        library = root / "lib" / os.environ["GPGPUSIM_CONFIG"] / "libcudart.so"
    library = library.resolve()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    config = args.config.resolve()
    binary = args.binary.resolve()
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    # Choose sixteen distinct physical cores from this process's allowed set.
    physical = set()
    cores = []
    for cpu in sorted(os.sched_getaffinity(0)):
        topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((topo / "physical_package_id").read_text(), (topo / "core_id").read_text())
        if key not in physical:
            physical.add(key)
            cores.append(cpu)
    if len(cores) < 16:
        parser.error("four four-thread jobs require sixteen distinct physical cores")
    cpus = [cores[i:i+4] for i in range(0, 16, 4)]
    hw = args.hardware.resolve()
    with hw.open() as stream:
        cases = [r for r in csv.DictReader(stream) if r["profile"] == args.profile and r["direction"] == "forward"]
    expected_count = 20 if args.profile == "large" else 4
    if len(cases) != expected_count or len({r['gtest'] for r in cases}) != expected_count:
        parser.error(f"expected {expected_count} distinct {args.profile} forward cases")
    for row in cases:
        if row.get("native_status") != "0" or row.get("ncu_status") != "0":
            parser.error(f"hardware collection failed: {row['gtest']}")
        if not re.fullmatch(r"Fa3PrefillFp16(?:Integration|Medium)Test\.H(?:16D128|32D64)(?:Full|Causal)B\d+S\d+", row["gtest"]) or float(row["end_to_end_duration_us"]) <= 0:
            parser.error(f"invalid hardware row: {row}")
    identity = dict(binary_sha256=digest, hardware_sha256=hashlib.sha256(hw.read_bytes()).hexdigest(), profile=args.profile)
    identity_path = out / "inputs.json"
    if args.resume:
        if json.loads(identity_path.read_text()) != identity:
            parser.error("resume binary/hardware/profile mismatch")
    else:
        identity_path.write_text(json.dumps(identity, indent=2))
        shutil.copyfile(binary, out / "workload")
        (out / "workload").chmod(0o555)
    binary = out / "workload"
    if hashlib.sha256(binary.read_bytes()).hexdigest() != digest:
        parser.error("frozen workload binary mismatch")
    # Freeze the simulator configuration for the entire multi-hour queue.
    if not args.resume:
        shutil.copytree(config, out / "config")
    else:
        for name in ("gpgpusim.config", "config_ampere_islip.icnt"):
            if (config / name).read_bytes() != (out / "config" / name).read_bytes():
                raise RuntimeError(f"Resume config mismatch: {name}")
    clock = re.search(r"^-gpgpu_clock_domains\s+([\d.]+):",
                      (out / "config/gpgpusim.config").read_text(), re.M)
    if clock is None or float(clock[1]) <= 0:
        raise RuntimeError("Missing or invalid core clock in frozen config")
    core_clock_mhz = float(clock[1])
    library_dir, library_digest = freeze_library(library, out, args.resume)
    # Later jobs must not depend on a mutable checkout's decoder entrypoints.
    decoder_dir = out / "decoder"
    decoder_dir.mkdir(exist_ok=args.resume)
    for name in ("dump_kernel_sassir.py", "nvdisasm_to_sassir.py"):
        source = root / "src/gpgpu-sim/flash/sass/tools" / name
        target = decoder_dir / name
        if args.resume:
            if not target.is_file() or target.read_bytes() != source.read_bytes():
                raise RuntimeError("Resume decoder mismatch; use a new output directory")
        else:
            shutil.copyfile(source, target)
    base = {k: v for k, v in os.environ.items()
            if not k.startswith("FLASHGPU_SASS_") and k != "PTX_SIM_MODE_FUNC"}
    base.update(FLASHGPU_SASS_AUTO="1", FLASHGPU_SASS_TIMING="1", FLASHGPU_SASS_ARCH="sm90",
        LD_LIBRARY_PATH=str(library_dir) + ":" + base.get("LD_LIBRARY_PATH", ""),
        FLASHGPU_SASS_BINARY=str(binary),
        FLASHGPU_SASS_DECODE_TOOL=str(decoder_dir / "dump_kernel_sassir.py"),
        FLASHGPU_SASS_PYTHON=shutil.which("python3"),
        FLASHGPU_SASS_NVDISASM=str(args.cuda / "bin/nvdisasm"),
        FLASHGPU_SASS_CUOBJDUMP=str(args.cuda / "bin/cuobjdump"),
        OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4",
        OMP_PROC_BIND="true", OMP_DYNAMIC="false")

    pending = []
    for row in sorted(cases, key=lambda r: (int(re.search(r'S(\d+)$', r['gtest'])[1]), r['gtest'])):
        path = out / (row['gtest'].split('.')[1] + '.json')
        if skip_completed_case(path, args.resume):
            continue
        pending.append(row)
    (out / ("resume_provenance.json" if args.resume else "provenance.json")).write_text(json.dumps(dict(binary=str(binary), sha256=digest,
        config=str(config), core_clock_mhz=core_clock_mhz, cpu_groups=cpus,
        hardware=str(hw), library=str(library), library_sha256=library_digest,
        pid=os.getpid()), indent=2))
    lock = threading.Lock()
    failed = threading.Event()
    def run(row, group):
        name = row["gtest"].split(".")[1]
        work = out / "work" / name
        work.parent.mkdir(exist_ok=True)
        shutil.copytree(out / "config", work)
        env = base.copy()
        env["OMP_PLACES"] = ",".join("{" + str(cpu) + "}" for cpu in group)
        command = ["taskset", "-c", ",".join(map(str, group)), str(binary), "--gtest_filter=" + row["gtest"]]
        start = time.time()
        status = dict(case=name, state="running", cpus=group, started=start, command=command)
        status_path = out / (name + ".json")
        def save():
            tmp = status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(status, indent=2))
            tmp.replace(status_path)
        with (work / "run.log").open("w") as log:
            proc = subprocess.Popen(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
            status["pid"] = proc.pid
            save()
            print(f"[start] {name} pid={proc.pid} cpus={group}", flush=True)
            rc = proc.wait()
        text = (work / "run.log").read_text()
        cycles = re.findall(r"^gpu_kernel_sim_cycle\[\d+\] = (\d+)", text, re.M)
        ok = rc == 0 and "[  PASSED  ] 1 test." in text and bool(cycles) and "mode=execution-driven SASS timing simulation" in text
        status.update(state="done" if ok else "failed", rc=rc, elapsed_sec=time.time()-start,
                      sim_cycles=sum(map(int, cycles)) if cycles else None,
                      ncu_us=float(row["end_to_end_duration_us"]))
        if ok:
            status["sim_us"] = status["sim_cycles"] / core_clock_mhz
            status["error_pct"] = (status["sim_us"] / status["ncu_us"] - 1) * 100
        save()
        print(f"[{status['state']}] {name} {status}", flush=True)
        return ok

    def worker(slot):
        try:
            while True:
                with lock:
                    if failed.is_set() or not pending:
                        return
                    row = pending.pop(0)
                if not run(row, cpus[slot]):
                    failed.set()
        except BaseException:
            failed.set()
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, slot) for slot in range(4)]
        for future in futures:
            future.result()
    if failed.is_set():
        print('[stopped] failure; remaining queued cases not launched', flush=True)
        return 1
    print(f"[complete] all {len(cases)} {args.profile} forward cases", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
