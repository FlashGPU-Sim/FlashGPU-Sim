#!/usr/bin/env python3
"""Native-only collection. Preserve raw samples and full NCU, with safe resume."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from plan import cases, smoke_cases


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def valid(text):
    return ('verification=PASS' in text or 'CHECK,PASS,bad=0' in text) and 'verification=FAIL' not in text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--out', type=Path)
    p.add_argument('--ncu', default='ncu')
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--family', choices=('all', 'mma', 'shared', 'solo', 'transfer'), default='all')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    matrix = [r for r in cases() if a.family == 'all' or r['family'] == a.family]
    if a.dry_run:
        print(json.dumps(dict(native_cases=len(matrix), repetitions=a.repetitions,
                             smoke_cases=len(list(smoke_cases(matrix))),
                             ncu_full=sum(r['profile'] for r in matrix), cases=matrix), indent=2))
        return
    if not a.out or a.repetitions < 1:
        p.error('--out and positive --repetitions are required')
    package, out = a.package.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    binaries = {r['binary']: digest(package / 'bin' / r['binary']) for r in matrix}
    spec = dict(binaries=binaries, matrix=matrix, repetitions=a.repetitions,
                ncu_clock_control='base', native_clock_control='unchanged',
                full_set=True, solo_ncu_skip_matching_warmup_launch=True,
                bundled_cuda=digest(package / 'lib/libcudart.so.13'))
    path = out / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise RuntimeError('Resume manifest mismatch; use a new results directory')
    path.write_text(json.dumps(spec, indent=2) + '\n')
    # Use only the packaged real cudart and the system GPU driver. In particular,
    # inherited simulator paths and preload settings must not reach native runs.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('FLASHGPU_', 'PTX_SIM_', 'GPGPUSIM_')) and k not in ('LD_PRELOAD', 'LD_LIBRARY_PATH')}
    env['LD_LIBRARY_PATH'] = str(package / 'lib')
    ncu = shutil.which(a.ncu) or a.ncu
    attempt = str(time.time_ns())
    for command, name in ((['nvidia-smi', '-q'], f'gpu-before-{attempt}.txt'), ([ncu, '--version'], f'ncu-version-{attempt}.txt')):
        with (out / name).open('w') as log:
            subprocess.run(command, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    progress = dict(status='RUNNING', started_unix=time.time(), completed=[])

    def save():
        temp = out / 'progress.tmp'
        temp.write_text(json.dumps(progress, indent=2) + '\n')
        temp.replace(out / 'progress.json')

    def run(row, suffix, profile=False):
        work = out / row['name']
        work.mkdir(exist_ok=True)
        log = work / (suffix + '.log')
        receipt = work / (suffix + '.json')
        report = work / (suffix + '.ncu-rep')
        csv = work / (suffix + '.csv')
        binary = package / 'bin' / row['binary']
        cmd = [str(binary), *row['args']]
        if profile:
            filters = (['--kernel-name-base', 'function', '--kernel-name', 'regex:tmem_probe',
                        '--launch-skip', '1', '--launch-count', '1'] if row['family'] == 'solo' else [])
            cmd = [ncu, '--clock-control', 'base', '--set', 'full', '--force-overwrite',
                   '--export', str(work / suffix), *filters, *cmd]
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old['command'] != cmd or old['binary_sha256'] != binaries[row['binary']]:
                raise RuntimeError('Attempt provenance mismatch: ' + str(work))
            if old['returncode'] == 0 and log.exists() and digest(log) == old['log_sha256'] and valid(log.read_text()):
                if not profile or (report.exists() and csv.exists() and digest(report) == old.get('ncu_sha256') and digest(csv) == old.get('csv_sha256')):
                    progress['completed'].append(str(receipt.relative_to(out)))
                    save()
                    return
            raise RuntimeError('Incomplete previous attempt: preserve it and use a new output directory')
        if log.exists() or report.exists() or csv.exists():
            raise RuntimeError('Unreceipted attempt exists; preserve it and use a new output directory: ' + str(work))
        if digest(binary) != binaries[row['binary']] or digest(package / 'lib/libcudart.so.13') != spec['bundled_cuda']:
            raise RuntimeError('Binary changed during collection')
        start = time.time()
        (work / (suffix + '.command.json')).write_text(json.dumps(cmd, indent=2))
        with log.open('w') as stream:
            result = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
        record = dict(command=cmd, binary_sha256=binaries[row['binary']], returncode=result.returncode,
                      log_sha256=digest(log), seconds=time.time() - start)
        if result.returncode or not valid(log.read_text(errors='replace')):
            receipt.write_text(json.dumps(record, indent=2))
            raise RuntimeError('Numerical/launch failure: ' + str(log))
        if profile:
            if not report.exists():
                raise RuntimeError('Missing NCU full report: ' + str(report))
            with csv.open('w') as stream:
                subprocess.run([ncu, '--import', str(report), '--page', 'raw', '--csv'], env=env,
                               stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=300)
            record.update(ncu_sha256=digest(report), csv_sha256=digest(csv))
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        progress['completed'].append(str(receipt.relative_to(out)))
        save()
        print(row['name'], suffix, 'PASS', flush=True)

    save()
    try:
        # Complete all numerical smoke gates before interpreting performance.
        for row in smoke_cases(matrix):
            run(row, 'smoke')
        for row in matrix:
            for repeat in range(a.repetitions):
                run(row, f'native-{repeat}')
        for row in matrix:
            if row['profile']:
                run(row, 'full', profile=True)
        progress['status'] = 'PASS'
    except Exception as error:
        progress.update(status='FAILED', error=repr(error))
        raise
    finally:
        progress['updated_unix'] = time.time()
        save()
        with (out / f'gpu-after-{attempt}.txt').open('w') as log:
            subprocess.run(['nvidia-smi', '-q'], env=env, stdout=log, stderr=subprocess.STDOUT)
        hashes = {str(f.relative_to(out)): digest(f) for f in out.rglob('*') if f.is_file() and f.name != 'SHA256.json'}
        (out / 'SHA256.json').write_text(json.dumps(hashes, indent=2) + '\n')


if __name__ == '__main__':
    main()
