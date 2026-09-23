#!/usr/bin/env python3
"""Track, compile and replay FA4 cases with per-run configs and logs."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
from queue import Queue, Empty

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'tests/scripts'))
from cpu_affinity import parse_cpu_list, select_cpu_sets, validate_cpu_sets

def used_memory():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return (int(fields['MemTotal'].split()[0]) - int(fields['MemAvailable'].split()[0])) * 1024


def stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


class Processes:
    """Own child processes and serialize memory admission/eviction decisions."""
    def __init__(self, memory_limit_gib):
        self.memory_limit = memory_limit_gib * 1024**3
        self.memory_limit_label = f'{memory_limit_gib} GiB'
        self.lock = threading.Lock()
        self.active = []
        self.cancelled = threading.Event()
        self.evicted = set()

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            for process in reversed(self.active):
                stop(process)

    def execute(self, command, cwd, log, monitor=False):
        with log.open('w') as output:
            waiting = False
            while True:
                with self.lock:
                    if self.cancelled.is_set():
                        raise RuntimeError('Run cancelled')
                    if not monitor or used_memory() < self.memory_limit:
                        process = subprocess.Popen(command, cwd=cwd, stdout=output,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                        self.active.append(process)
                        break
                if not waiting:
                    print(f'Waiting for memory below {self.memory_limit_label}: {cwd.name}', flush=True)
                    waiting = True
                self.cancelled.wait(1)
            try:
                while True:
                    try:
                        code = process.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        if monitor:
                            with self.lock:
                                while used_memory() > self.memory_limit:
                                    victim = next((p for p in reversed(self.active) if p.poll() is None), None)
                                    if victim is None:
                                        break
                                    self.evicted.add(victim.pid)
                                    print(f'Memory exceeded {self.memory_limit_label}; stopping newest replay PID {victim.pid}', flush=True)
                                    stop(victim)
                if process.pid in self.evicted:
                    raise RuntimeError(f'Replay stopped because machine memory exceeded {self.memory_limit_label}')
                if code:
                    raise RuntimeError(f'Command exited with {code}; see {log}')
            finally:
                with self.lock:
                    stop(process)
                    self.active.remove(process)


def select_cases(args, parser):
    with (HERE / 'configs/fa4_cases.csv').open() as source:
        presets = list(csv.DictReader(source))
    if args.shape:
        if args.suite or args.case:
            parser.error('--shape cannot be combined with --suite or --case')
        result = []
        for value in args.shape:
            try:
                b, h, s, d, causal = value.split(',')
                b, h, s, d = map(int, (b, h, s, d))
                causal = causal.lower()
                if min(b, h, s) <= 0 or d not in (64, 128) or causal not in ('true', 'false'):
                    raise ValueError()
            except ValueError:
                parser.error('--shape expects positive B,H,S,D,true|false, with D=64 or 128')
            name = f'H{h}D{d}{"Causal" if causal == "true" else "Full"}B{b}S{s}'
            row = next((r for r in presets if r['name'] == name), None)
            result.append(row or dict(suite='custom', name=name, batch=str(b), heads=str(h),
                                     seqlen_q=str(s), head_dim=str(d), causal=causal))
    else:
        if args.case and set(args.case) - {r['name'] for r in presets}:
            parser.error('unknown case name; use --list to see presets')
        result = [r for r in presets if (not args.suite or 'all' in args.suite or r['suite'] in args.suite)
                  and (not args.case or r['name'] in args.case)]
    if not result:
        parser.error('no matching cases')
    return list({r['name']: r for r in result}.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('track', 'compile', 'replay', 'all'))
    parser.add_argument('--suite', action='append', choices=('smoke', 'small', 'medium', 'large', 'all'))
    parser.add_argument('--case', action='append', help='Preset case name; repeatable')
    parser.add_argument('--shape', action='append', help='B,H,S,D,true|false; repeatable')
    parser.add_argument('--list', action='store_true', help='Show selection without running')
    parser.add_argument('--export-root', type=Path, default=HERE.parent / 'exports/fa4')
    parser.add_argument('--run-root', type=Path, default=HERE.parent / 'runs/fa4')
    parser.add_argument('--config-dir', type=Path, default=ROOT / 'configs/SM100_B200')
    parser.add_argument('--native', action='store_true', help='Replay on GPU using the current environment')
    parser.add_argument('--sm-count', type=int, default=148)
    parser.add_argument('--memory-limit-gib', type=int, default=50,
                        help='Whole-machine used-memory limit for simulation in GiB (default: 50)')
    parser.add_argument('--jobs', type=int, default=4, help='Maximum concurrent simulations (default: 4)')
    parser.add_argument('--cpus-per-job', type=int, default=4, help='Physical cores per simulated replay (default: 4)')
    parser.add_argument('--cpus', help='Explicit replay CPU list, e.g. 0,2,4-5; overrides automatic selection')
    args = parser.parse_args()
    if args.memory_limit_gib <= 0:
        parser.error('--memory-limit-gib must be positive')
    if args.jobs <= 0:
        parser.error('--jobs must be positive')
    if args.sm_count <= 0:
        parser.error('--sm-count must be positive')
    if args.cpus_per_job <= 0:
        parser.error('--cpus-per-job must be positive')
    if args.cpus:
        try:
            args.cpus = validate_cpu_sets([args.cpus])[0]
        except ValueError as error:
            parser.error(str(error))
    cases = select_cases(args, parser)
    if args.list:
        for row in cases:
            print(f"{row['suite']}/{row['name']}")
        return
    stages = ('track', 'compile', 'replay') if args.stage == 'all' else (args.stage,)
    config = args.config_dir.resolve()
    if 'replay' in stages and not args.native and not (config / 'gpgpusim.config').is_file():
        parser.error(f'No gpgpusim.config in {config}')
    cpu_sets = [None]
    if 'replay' in stages and not args.native:
        try:
            cpu_sets = ([args.cpus] if args.cpus else
                        select_cpu_sets(min(args.jobs, len(cases)), args.cpus_per_job))
        except ValueError as error:
            parser.error(str(error))
        print(f'Simulation workers: {len(cpu_sets)}; CPU sets: {cpu_sets}', flush=True)
    run = args.run_root.resolve() / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    run.mkdir(parents=True)
    print(f'Run directory: {run}', flush=True)
    results = []
    summary_lock = threading.Lock()
    processes = Processes(args.memory_limit_gib)

    def run_case(row, stage, cpus=None):
        relative = Path(row['suite']) / row['name']
        export = args.export_root.resolve() / relative
        work = run / relative
        work.mkdir(parents=True, exist_ok=True)
        record = dict(case=str(relative), stage=stage, status='running', export=str(export))
        with summary_lock:
            results.append(record)
        print(f'{stage}: {relative}', flush=True)
        try:
            if stage == 'track':
                command = [sys.executable, str(HERE / 'fa4_forward.py'), '--output-dir', str(export),
                           '--batch', row['batch'], '--heads', row['heads'], '--seqlen', row['seqlen_q'],
                           '--head-dim', row['head_dim'], '--sm-count', str(args.sm_count)]
                if row['causal'] == 'false':
                    command.append('--non-causal')
            elif stage == 'compile':
                command = ['make', '-C', str(export)]
            else:
                if not (export / 'replay').is_file():
                    raise RuntimeError(f'Missing {export / "replay"}; run track and compile first')
                shutil.copy2(export / 'replay', work / 'replay')
                (work / 'modules').symlink_to(export / 'modules', target_is_directory=True)
                (work / 'data').mkdir()
                for path in (export / 'data').iterdir():
                    if path.is_file() and not path.name.endswith('.result.bin'):
                        (work / 'data' / path.name).symlink_to(path)
                if args.native:
                    command = [str(work / 'replay')]
                else:
                    for path in config.iterdir():
                        if path.is_file():
                            shutil.copy2(path, work / path.name)
                    command = ['bash', '-c', 'source "$1/setup_environment" release && exec "$2"',
                               'fa4-replay', str(ROOT), str(work / 'replay')]
            if stage == 'replay' and not args.native:
                threads = len(parse_cpu_list(cpus))
                with summary_lock:
                    record.update(cpus=cpus, omp_num_threads=threads)
                print(f'  CPUs: {cpus}; OpenMP threads: {threads}', flush=True)
                command = ['taskset', '--cpu-list', cpus, 'env',
                           f'OMP_NUM_THREADS={threads}', *command]
            processes.execute(command, work, work / f'{stage}.log', monitor=stage == 'replay' and not args.native)
            with summary_lock:
                record['status'] = 'passed'
        except (Exception, KeyboardInterrupt) as error:
            with summary_lock:
                record.update(status='failed', error=str(error) or 'Interrupted')
            raise
        finally:
            with summary_lock:
                (run / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')

    def worker(queue, cpus):
        while not processes.cancelled.is_set():
            try:
                row = queue.get_nowait()
            except Empty:
                return
            try:
                run_case(row, 'replay', cpus)
            except Exception:
                processes.cancelled.set()
                raise

    try:
        for stage in stages:
            if stage != 'replay' or args.native:
                for row in cases:
                    run_case(row, stage)
            else:
                queue = Queue()
                for row in cases:
                    queue.put(row)
                with ThreadPoolExecutor(max_workers=len(cpu_sets)) as pool:
                    futures = [pool.submit(worker, queue, cpus) for cpus in cpu_sets]
                    try:
                        for future in as_completed(futures):
                            future.result()
                    except BaseException:
                        processes.cancel()
                        raise
    except (Exception, KeyboardInterrupt) as error:
        processes.cancel()
        print(f'Failed: {error or "Interrupted"}', file=sys.stderr)
        return 1
    print(f'Passed {len(cases)} cases; logs: {run}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
