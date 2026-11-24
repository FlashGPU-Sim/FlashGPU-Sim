#!/usr/bin/env python3
"""Run identical TCGen05 CUDA fixtures through PTX and SASS timing frontends.

The binary must retain its embedded PTX and sm_100a cubin. Results qualify
only the fixtures listed in the report, not the entire ISA or DeepInfer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--config-dir', type=Path, required=True)
    p.add_argument('--cuda', type=Path, default=Path('/usr/local/cuda'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runtime-library-dir', type=Path, help='Use an immutable simulator library snapshot')
    p.add_argument('--timeout', type=int, default=300)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--frontends', nargs='+', choices=['ptx', 'sass-timing'], default=['ptx', 'sass-timing'])
    p.add_argument('--filter', default='Tcgen05*')
    p.add_argument('--expected-tests', type=int, default=12)
    p.add_argument('--timing-marker', default='gpu_sim_cycle =')
    p.add_argument('--cluster-validation', action='store_true', help='Use a small 2-GPC x 4-SM topology with DSM for cluster semantics')
    p.add_argument('--disable-ptx-reorder', action='store_true', help='Use PTX source order for semantic fixtures without SASS line anchors')
    args = p.parse_args()
    root, binary, cuda = args.root.resolve(), args.binary.resolve(), args.cuda.resolve()
    library = args.runtime_library_dir.resolve() if args.runtime_library_dir else None
    if library and not (library / 'libcudart.so').is_file():
        p.error('--runtime-library-dir must contain libcudart.so')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # CUDA fixtures normally request the versioned SONAME (e.g. libcudart.so.13),
    # not libcudart.so. A snapshot containing only the latter would silently
    # fall through to the mutable build directory for the versioned library.
    runtime_aliases = None
    if library:
        runtime_aliases = output / 'runtime-library-aliases'
        runtime_aliases.mkdir(exist_ok=True)
        for name in ('libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12',
                     'libcudart.so.13', 'libcudart_mod.so', 'libcuda.so', 'libcuda.so.1'):
            alias = runtime_aliases / name
            target = library / 'libcudart.so'
            if alias.is_symlink():
                if alias.resolve() != target.resolve():
                    raise RuntimeError(f'Conflicting runtime alias: {alias}')
            elif alias.exists():
                raise RuntimeError(f'Refusing to replace runtime artifact: {alias}')
            else:
                alias.symlink_to(target)
    config = (args.config_dir / 'gpgpusim.config').read_text()
    if args.disable_ptx_reorder:
        config += '\n-gpgpu_ptx_reorder 0\n-gpgpu_ptx_reorder_sass_guided 0\n'
    if args.cluster_validation:
        config += '\n-gpgpu_n_clusters 2\n-gpgpu_n_cores_per_cluster 4\n-gpgpu_dsm_enable 1\n'
    report = {'binary': str(binary), 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
              'config_sha256': hashlib.sha256(config.encode()).hexdigest(), 'runs': {}}
    if library:
        report['runtime_library_dir'] = str(library)
        report['runtime_alias_dir'] = str(runtime_aliases)
        report['runtime_sha256'] = hashlib.sha256((library / 'libcudart.so').read_bytes()).hexdigest()
    report['timing_marker'] = args.timing_marker
    report['threads'] = args.threads
    report['filter'] = args.filter
    report['expected_tests'] = args.expected_tests
    for frontend in args.frontends:
        if hashlib.sha256(binary.read_bytes()).hexdigest() != report['binary_sha256']:
            raise RuntimeError('Fixture binary changed during dual-frontend run')
        work = output / frontend
        work.mkdir(exist_ok=True)
        for f in args.config_dir.iterdir():
            if f.is_file() and f.suffix in {'.config', '.icnt', '.rules'}:
                shutil.copyfile(f, work / f.name)
        (work / 'gpgpusim.config').write_text(config + '\n-gpgpu_execution_frontend ' + frontend + '\n')
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('FLASHGPU_SASS_', 'PTX_SIM_'))}
        env.pop('GTEST_OUTPUT', None)
        env.update(CUDA_INSTALL_PATH=str(cuda), OMP_NUM_THREADS=str(args.threads),
                   FLASHGPU_TEST_LIBRARY_DIR=str(runtime_aliases) if runtime_aliases else '')
        if frontend == 'ptx':
            subprocess.run([str(cuda / 'bin/cuobjdump'), '-xptx', 'all', str(binary)],
                           cwd=work, check=True, stdout=subprocess.DEVNULL)
            env['PTX_SIM_USE_PTX_FILE'] = '1'
        else:
            env.update(FLASHGPU_SASS_AUTO='1', FLASHGPU_SASS_TIMING='1', FLASHGPU_SASS_ARCH='sm100',
                       FLASHGPU_SASS_BINARY=str(binary), FLASHGPU_SASS_PYTHON=shutil.which('python3'),
                       FLASHGPU_SASS_NVDISASM=str(cuda / 'bin/nvdisasm'),
                       FLASHGPU_SASS_CUOBJDUMP=str(cuda / 'bin/cuobjdump'),
                       FLASHGPU_SASS_DECODE_TOOL=str(root / 'src/gpgpu-sim/flash/sass/tools/dump_kernel_sassir.py'))
        log = output / (frontend + '.log')
        start = time.monotonic()
        with log.open('w') as f:
            command = ['bash', '-c', 'cd "$1" && source setup_environment release >/dev/null && cd "$2" && export LD_LIBRARY_PATH="${FLASHGPU_TEST_LIBRARY_DIR:+$FLASHGPU_TEST_LIBRARY_DIR:}$LD_LIBRARY_PATH" && exec timeout "$4" "$3" --gtest_filter="$5"',
                       'dual-frontend', str(root), str(work), str(binary), str(args.timeout), args.filter]
            rc = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        if library and hashlib.sha256((library / 'libcudart.so').read_bytes()).hexdigest() != report['runtime_sha256']:
            raise RuntimeError('Simulator library changed during dual-frontend run')
        text = log.read_text(errors='replace')
        tests = re.findall(r'\[       OK \] (\S+)', text)
        mode = ('mode=execution-driven SASS timing simulation' in text) if frontend == 'sass-timing' else ('GPGPU-Sim PTX: Parsing' in text)
        passed = mode and rc == 0 and len(tests) == args.expected_tests and args.timing_marker in text and '[  FAILED  ]' not in text and not re.search(r'syntax error|parser error', text, re.I)
        report['runs'][frontend] = {'returncode': rc, 'passed': passed, 'tests': tests,
                                    'wall_seconds': time.monotonic() - start, 'log': str(log),
                                    'gpu_sim_cycle': re.findall(r'^gpu_sim_cycle\s*=\s*(\d+)', text, re.M)}
        print(f'{frontend}: passed={passed}, numerical tests={len(tests)}, seconds={time.monotonic() - start:.2f}', flush=True)
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if all(v['passed'] for v in report['runs'].values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
