#!/usr/bin/env python3
"""Collect native numerics before unprofiled timing and a bounded NCU-full matrix."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


def matrix():
    cases = []
    for ts in (0, 1):
        for mode in range(4):
            for maximum, shift in ((0, 0), (8, 8), (16, 16), (32, 32), (32, 0)):
                for burst in (0, 1, 32, 128):
                    cases.append((maximum, shift, mode, 0, ts, burst))
        for mode in (1, 2):
            for mask in (1, 2):
                for burst in (32, 128):
                    cases.append((32, 32, mode, mask, ts, burst))
    assert len(cases) == len(set(cases)) == 176
    return [dict(name='w%d-s%d-op%d-mask%d-ts%d-b%d' % c,
                 args=list(map(str, c)), profile=c[-1] == 128, status='QUEUED')
            for c in cases]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path)
    ap.add_argument('--ncu', default='ncu')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--skip-ncu', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    plan = matrix()
    if a.dry_run:
        print(json.dumps(dict(numerical_tests=12, timing_cases=len(plan),
                              ncu_timing_cases=sum(c['profile'] for c in plan),
                              plan=plan), indent=2))
        return
    if not a.out or a.repeats < 1:
        ap.error('--out and positive --repeats are required')
    package = Path(__file__).resolve().parent
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('FLASHGPU_', 'PTX_SIM_', 'GPGPUSIM_'))
           and k not in ('LD_PRELOAD', 'LD_LIBRARY_PATH')}
    env['LD_LIBRARY_PATH'] = str(package / 'lib')
    state = dict(status='RUNNING', started=time.time(), clock_control_native='unchanged',
                 clock_control_ncu='base', binary_sha256={
                     n: hashlib.sha256((package / 'bin' / n).read_bytes()).hexdigest()
                     for n in ('ws_numerics', 'ws_timing')}, plan=plan)

    def save():
        tmp = out / 'status.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(out / 'status.json')

    def run(cmd, log, timeout=120):
        log.with_suffix('.command.json').write_text(json.dumps(cmd, indent=2) + '\n')
        with log.open('w') as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                           timeout=timeout, check=True)

    def profile(binary, args, dest):
        run([a.ncu, '--set', 'full', '--clock-control', 'base', '--target-processes',
             'all', '--export', str(dest / 'full'), str(package / 'bin' / binary), *args],
            dest / 'ncu.log', 900)
        if not (dest / 'full.ncu-rep').is_file():
            raise RuntimeError('missing NCU report')
        run([a.ncu, '--import', str(dest / 'full.ncu-rep'), '--page', 'raw', '--csv'],
            dest / 'full.csv', 120)

    save()
    try:
        run(['nvidia-smi', '-q'], out / 'gpu-before.log')
        if not a.skip_ncu:
            run([a.ncu, '--version'], out / 'ncu-version.log')
        run([str(package / 'bin/ws_numerics')], out / 'numerics.log', 180)
        if '[  PASSED  ] 12 tests.' not in (out / 'numerics.log').read_text():
            raise RuntimeError('all twelve numerical cases must pass before timing')
        state['numerical_pass'] = True
        if not a.skip_ncu:
            nd = out / 'numerics-ncu'
            nd.mkdir()
            profile('ws_numerics', [], nd)
        for case in plan:
            dest = out / case['name']
            dest.mkdir()
            case['status'] = 'RUNNING'
            save()
            try:
                for rep in range(a.repeats):
                    log = dest / f'native-{rep}.log'
                    run([str(package / 'bin/ws_timing'), *case['args'], '32'], log)
                    if 'verification=PASS errors=0' not in log.read_text():
                        raise RuntimeError('timing output failed numerical verification')
                if case['profile'] and not a.skip_ncu:
                    profile('ws_timing', [*case['args'], '8'], dest)
                    if 'verification=PASS errors=0' not in (dest / 'ncu.log').read_text():
                        raise RuntimeError('NCU replay failed numerical verification')
                case['status'] = 'PASS'
            except Exception as e:
                case.update(status='FAIL', error=repr(e))
            save()
            print(case['name'], case['status'], flush=True)
        rows = []
        for case in plan:
            samples = []
            for log in (out / case['name']).glob('native-*.log'):
                text = log.read_text()
                if 'verification=PASS errors=0' not in text:
                    continue
                samples += [(int(i), int(c)) for i, c in re.findall(r'^SAMPLE,\d+,(\d+),(\d+)$', text, re.M)]
            if samples:
                rows.append(dict(name=case['name'], samples=len(samples),
                                 issue=statistics.median(x[0] for x in samples),
                                 complete=statistics.median(x[1] for x in samples)))
        with (out / 'timing.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=['name', 'samples', 'issue', 'complete'])
            w.writeheader()
            w.writerows(rows)
        by_name = {r['name']: r for r in rows}
        slopes = []
        for r in rows:
            if not r['name'].endswith('-b128'):
                continue
            other = by_name.get(r['name'].removesuffix('128') + '32')
            if other:
                slopes.append(dict(case=r['name'].rsplit('-b', 1)[0],
                    issue_cycles_per_mma=(r['issue'] - other['issue']) / 96,
                    complete_cycles_per_mma=(r['complete'] - other['complete']) / 96))
        (out / 'slopes.json').write_text(json.dumps(slopes, indent=2) + '\n')
        run(['nvidia-smi', '-q'], out / 'gpu-after.log')
        state['status'] = 'PASS' if all(c['status'] == 'PASS' for c in plan) else 'FAIL'
    except Exception as e:
        state.update(status='FAIL', error=repr(e))
    state['finished'] = time.time()
    save()
    raise SystemExit(0 if state['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
