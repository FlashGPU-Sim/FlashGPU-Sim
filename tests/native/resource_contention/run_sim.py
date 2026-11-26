#!/usr/bin/env python3
"""Replay the exact packaged cubin/PTX locally; freeze runtime and log provenance."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time
from plan import cases, smoke_cases


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--family', choices=('all', 'mma', 'shared', 'solo', 'transfer'), default='all')
    p.add_argument('--frontends', nargs='+', choices=('ptx', 'sass-timing'), default=['sass-timing'])
    p.add_argument('--case-regex', help='Select named cases before smoke deduplication')
    p.add_argument('--config-override', action='append', default=[], help='Append one explicit simulator configuration line; recorded in each frozen config')
    p.add_argument('--timing-matrix', action='store_true', help='Replay exact native timing arguments instead of reduced numerical smoke loops')
    p.add_argument('--full-topology', action='store_true', help='Retain all B200 SMs; default numerical smoke uses one SM for these one-CTA probes')
    a = p.parse_args()
    root, package, runtime, out = (x.resolve() for x in (a.root, a.package, a.runtime, a.out))
    out.mkdir(parents=True, exist_ok=False)
    cuda = Path('/usr/local/cuda-13.3')
    rows = [r for r in cases() if a.family == 'all' or r['family'] == a.family]
    if a.case_regex:
        rows = [r for r in rows if re.search(a.case_regex, r['name'])]
    if not rows:
        p.error('case selection is empty')
    if not a.timing_matrix:
        rows = list(smoke_cases(rows))
    jobs = [(f, r) for f in a.frontends for r in rows]
    # This can run concurrently with long acceptance work. Never rebuild or
    # replace either library or fixture underneath these launches.
    library = runtime / 'libcudart.so'
    report = dict(status='RUNNING', runtime=str(library), runtime_sha256=sha(library),
                  workers=a.workers, threads_per_job=1, total=len(jobs), runs=[],
                  numerical_smoke=not a.timing_matrix, full_topology=a.full_topology,
                  timing_matrix=a.timing_matrix)
    binary_hashes = {r['binary']: sha(package / 'bin' / r['binary']) for r in rows}

    def run(item):
        frontend, row = item
        work = out / frontend / row['name']
        work.mkdir(parents=True)
        for f in (root / 'configs/SM100_B200').iterdir():
            if f.is_file() and f.suffix in ('.config', '.icnt', '.rules'):
                shutil.copy2(f, work / f.name)
        cfg = work / 'gpgpusim.config'
        cfg.write_text(cfg.read_text() + '\n-gpgpu_ptx_reorder 0\n-gpgpu_ptx_reorder_sass_guided 0\n-gpgpu_execution_frontend ' + frontend + '\n-gpgpu_max_cycle 3000000\n')
        if a.config_override:
            with cfg.open('a') as stream:
                stream.write('\n'.join(a.config_override) + '\n')
        if not a.full_topology:
            with cfg.open('a') as stream:
                stream.write('-gpgpu_n_clusters 1\n-gpgpu_n_cores_per_cluster 1\n')
        aliases = work / 'runtime'
        aliases.mkdir()
        for n in ('libcudart.so', 'libcudart.so.13', 'libcuda.so', 'libcuda.so.1'):
            (aliases / n).symlink_to(library)
        binary = package / 'bin' / row['binary']
        env = {k: v for k, v in os.environ.items() if not k.startswith(('FLASHGPU_', 'PTX_SIM_', 'GPGPUSIM_')) and k != 'LD_PRELOAD'}
        env.update(CUDA_INSTALL_PATH=str(cuda), OMP_NUM_THREADS='1', FLASHGPU_TEST_LIBRARY_DIR=str(aliases))
        if frontend == 'ptx':
            subprocess.run([cuda / 'bin/cuobjdump', '-xptx', 'all', binary], cwd=work, check=True, stdout=subprocess.DEVNULL)
            env['PTX_SIM_USE_PTX_FILE'] = '1'
        else:
            env.update(FLASHGPU_SASS_AUTO='1', FLASHGPU_SASS_TIMING='1', FLASHGPU_SASS_ARCH='sm100',
                       FLASHGPU_SASS_BINARY=str(binary), FLASHGPU_SASS_PYTHON=shutil.which('python3'),
                       FLASHGPU_SASS_NVDISASM=str(cuda / 'bin/nvdisasm'), FLASHGPU_SASS_CUOBJDUMP=str(cuda / 'bin/cuobjdump'),
                       FLASHGPU_SASS_DECODE_TOOL=str(root / 'src/gpgpu-sim/flash/sass/tools/dump_kernel_sassir.py'))
        cmd = ['bash', '-c', 'ulimit -c 0; cd "$1" && source setup_environment release >/dev/null && cd "$2" && export LD_LIBRARY_PATH="$FLASHGPU_TEST_LIBRARY_DIR:$LD_LIBRARY_PATH" && shift 2 && exec timeout 1200 "$@"', 'sim', str(root), str(work), str(binary), *row['args']]
        (work / 'command.json').write_text(json.dumps(cmd, indent=2))
        start = time.monotonic()
        with (work / 'sim.log').open('w') as log:
            rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        text = (work / 'sim.log').read_text(errors='replace')
        mode = 'mode=execution-driven SASS timing simulation' in text if frontend == 'sass-timing' else 'GPGPU-Sim PTX: Parsing' in text
        valid = ('verification=PASS' in text or 'CHECK,PASS,bad=0' in text) and 'verification=FAIL' not in text
        unchanged = sha(library) == report['runtime_sha256'] and sha(binary) == binary_hashes[row['binary']]
        return dict(name=row['name'], frontend=frontend, family=row['family'], args=row['args'],
                    returncode=rc, passed=rc == 0 and valid and mode and unchanged,
                    wall_seconds=time.monotonic() - start, log=str(work / 'sim.log'),
                    binary_sha256=binary_hashes[row['binary']], config_sha256=sha(cfg))

    def save():
        p = out / 'report.tmp'
        p.write_text(json.dumps(report, indent=2) + '\n')
        p.replace(out / 'report.json')

    save()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(run, j) for j in jobs]):
            row = future.result()
            report['runs'].append(row)
            save()
            print(row['frontend'], row['name'], row['passed'], flush=True)
    report['status'] = 'PASS' if all(r['passed'] for r in report['runs']) else 'FAIL'
    save()


if __name__ == '__main__':
    main()
