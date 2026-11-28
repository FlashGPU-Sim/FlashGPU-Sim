#!/usr/bin/env python3
"""Build a self-contained native package locally; retain sm100a cubin and PTX."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cuda', type=Path, default=Path('/usr/local/cuda-13.3'))
    p.add_argument('--cutlass', type=Path, required=True)
    p.add_argument('--reuse', type=Path, help='Retain executables only when source, headers, flags and toolchain are unchanged')
    p.add_argument('--rebuild', nargs='*', default=[], help='Explicitly rebuild these binaries even with --reuse')
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(here, out / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    for d in ('bin', 'lib', 'artifacts'):
        (out / d).mkdir()
    for name in ('collect.py', 'plan.py', 'summarize.py', 'audit.py', 'qualify.py', 'README.md'):
        if (here / name).exists():
            shutil.copy2(here / name, out / name)
    cuda = a.cuda.resolve()
    nvcc = cuda / 'bin/nvcc'
    (out / 'artifacts/nvcc-version.txt').write_bytes(subprocess.check_output([nvcc, '--version']))
    (out / 'artifacts/cutlass-head.txt').write_bytes(subprocess.check_output(['git', '-C', str(a.cutlass), 'rev-parse', 'HEAD']))
    (out / 'artifacts/cutlass.patch').write_bytes(subprocess.check_output(['git', '-C', str(a.cutlass), 'diff', 'HEAD', '--binary']))
    source = out / 'source/sources'
    jobs = []
    for name, upper, col in (('mma_low', 0, 384), ('mma_high', 1, 384),
                             ('mma_col320', 0, 320), ('mma_col448', 0, 448)):
        jobs.append((name, 'mma.cu', [f'-DSHARED_SCRATCH_UPPER={upper}',
                     f'-DTMEM_BACKGROUND_COLUMN={col}', '-DLDS_LIGHT_CHECKSUM=1',
                     '-DCONTROLLED_SHARED_REGIONS=1']))
    jobs += [('tmem_solo', 'tmem_solo.cu', []), ('transfers', 'transfers.cu', []),
             ('shared32', 'shared_pair.cu', ['-DPAIR_TMA_32K=1']),
             ('shared64', 'shared_pair.cu', [])]

    def build(job):
        name, src, extra = job
        cmd = [str(nvcc), '-std=c++17', '--expt-relaxed-constexpr', '-O3', '-lineinfo',
               '-gencode', 'arch=compute_100a,code=[sm_100a,compute_100a]', '-cudart', 'shared',
               '-I' + str(a.cutlass / 'include'), '-I' + str(source), *extra,
               str(source / src), '-o', str(out / 'bin' / name)]
        if src == 'shared_pair.cu':
            cmd += ['-L' + str(cuda / 'lib64/stubs'), '-lcuda']
        if a.reuse and name not in a.rebuild:
            old = a.reuse.resolve()
            for f in [src, *(h.name for h in source.glob('*.cuh'))]:
                if (old / 'source/sources' / f).read_bytes() != (source / f).read_bytes():
                    raise RuntimeError('Cannot reuse modified fixture source: ' + f)
            old_cmd = json.loads((old / 'artifacts' / (name + '.command.json')).read_text())
            # A reused executable keeps its original compiler invocation, even
            # across several package revisions.
            command_root = Path(old_cmd[old_cmd.index('-o') + 1]).parent.parent
            if [x.replace(str(command_root), str(out)) for x in old_cmd] != cmd:
                raise RuntimeError('Cannot reuse changed compiler arguments: ' + name)
            for f in ('nvcc-version.txt', 'cutlass-head.txt', 'cutlass.patch'):
                if (old / 'artifacts' / f).read_bytes() != (out / 'artifacts' / f).read_bytes():
                    raise RuntimeError('Cannot reuse changed toolchain: ' + f)
            shutil.copy2(old / 'bin' / name, out / 'bin' / name)
            for suffix in ('.sass', '.ptx', '.resources', '.build.log', '.command.json'):
                shutil.copy2(old / 'artifacts' / (name + suffix), out / 'artifacts' / (name + suffix))
            (out / 'artifacts' / (name + '.reuse.json')).write_text(json.dumps(dict(
                original_package=str(old), source_and_toolchain_unchanged=True,
                binary_sha256=hashlib.sha256((out / 'bin' / name).read_bytes()).hexdigest()), indent=2))
            print(name, 'REUSED IDENTICAL BINARY', flush=True)
            return
        (out / 'artifacts' / (name + '.command.json')).write_text(json.dumps(cmd, indent=2))
        with (out / 'artifacts' / (name + '.build.log')).open('w') as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        for option, suffix in (('--dump-sass', '.sass'), ('--dump-ptx', '.ptx'),
                               ('--dump-resource-usage', '.resources')):
            with (out / 'artifacts' / (name + suffix)).open('w') as log:
                subprocess.run([cuda / 'bin/cuobjdump', option, out / 'bin' / name], check=True, stdout=log)
        print(name, 'BUILT', flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(build, jobs))
    shutil.copy2((cuda / 'lib64/libcudart.so.13').resolve(), out / 'lib/libcudart.so.13')
    (out / 'lib/libcudart.so').symlink_to('libcudart.so.13')
    subprocess.run(['python3', str(out / 'audit.py'), str(out)], check=True)
    # Never bundle libcuda from the simulator or toolkit stubs for native use.
    manifest = {str(f.relative_to(out)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in out.rglob('*') if f.is_file()}
    (out / 'artifacts/file-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Built package:', out)


if __name__ == '__main__':
    main()
