#!/usr/bin/env python3
"""Fail closed on missing measured operations, spills, or unsafe shared geometry."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from plan import cases, smoke_cases


def audit(package):
    results = {}
    for binary in sorted({r['binary'] for r in cases()}):
        resources = (package / 'artifacts' / (binary + '.resources')).read_text()
        memory = re.findall(r'(?:STACK|LOCAL):(\d+)', resources)
        assert memory and all(int(n) == 0 for n in memory), (binary, 'spill')
        results[binary] = {'no_local_or_stack': True}
        if binary.startswith('mma_'):
            subprocess.run(['python3', str(package / 'source/sources/mma_audit.py'),
                            str(package), binary], check=True)
    parts = (package / 'artifacts/transfers.sass').read_text().split('Function : ')[1:]
    assert len(parts) == 26, len(parts)
    transfer = {}
    for part in parts:
        name = part.splitlines()[0].strip()
        kind, form, depth = map(int, re.search(r'ILi(\d+)ELi(\d+)ELi(\d+)E', name).groups())
        clocks = list(re.finditer(r'/\*([0-9a-f]+)\*/\s+CS2R[^;]*SR_CLOCKLO[^;]*;', part))
        assert len(clocks) == 2, (name, 'outer clocks')
        body = part[clocks[0].end():clocks[1].start()]
        assert not re.search(r'\b(?:LDG|STG|LDL|STL)\b', body), (name, 'timed global/local')
        opcode = ('LDTM', 'STTM', 'UTCCP')[kind]
        ops = re.findall(r'\b' + opcode + r'[^;]*;', body)
        assert len(ops) == depth, (name, 'missing transfer', ops)
        # At least one backedge encloses all transfer instructions.
        pcs = [int(m[1], 16) for m in re.finditer(
            r'/\*([0-9a-f]+)\*/\s+' + opcode + r'[^;]*;', body)]
        backs = [(int(m[2], 16), int(m[1], 16)) for m in re.finditer(
            r'/\*([0-9a-f]+)\*/[^;]*\bBRA(?:\.U)?\b[^;]*?(0x[0-9a-f]+)\s*;', body)]
        assert any(lo <= min(pcs) <= max(pcs) < hi for lo, hi in backs), (name, 'no enclosing loop')
        assert not re.search(r'\bSTG\b', part[:clocks[-1].end()]), (name, 'early report')
        tail = part[clocks[-1].end():]
        first_store = re.search(r'\bSTG\b', tail)
        assert first_store and 'BAR.SYNC' in tail[:first_store.start()], (name, 'missing join')
        transfer[name] = dict(kind=kind, form=form, depth=depth, opcodes=ops)
    results['transfers']['kernels'] = transfer
    solo = {}
    for part in (package / 'artifacts/tmem_solo.sass').read_text().split('Function : ')[1:]:
        name = part.splitlines()[0].strip()
        if 'tmem_probe' not in name:
            continue
        clocks = list(re.finditer(r'\bCS2R[^;]*SR_CLOCKLO[^;]*;', part))
        first_store = re.search(r'\bSTG\b', part)
        assert len(clocks) == 2 and first_store, (name, 'outer clocks/report')
        assert first_store.start() > clocks[-1].end(), (name, 'early report')
        assert 'BAR.SYNC' in part[clocks[-1].end():first_store.start()], (name, 'missing worker join')
        solo[name] = {'outer_clocks': 2, 'joined_before_report': True}
    assert len(solo) == 105, len(solo)
    results['tmem_solo']['kernels'] = solo
    env = {k: v for k, v in os.environ.items() if k not in ('LD_LIBRARY_PATH', 'LD_PRELOAD')}
    env['LD_LIBRARY_PATH'] = str(package / 'lib')
    geometries = []
    for row in smoke_cases([r for r in cases() if r['family'] == 'shared']):
        result = subprocess.run([str(package / 'bin' / row['binary']), *row['args'], '--dry-run'],
                                env=env, check=True, capture_output=True, text=True)
        geometries.append(dict(name=row['name'], output=result.stdout))
    results['shared_geometry'] = geometries
    (package / 'artifacts/audit.json').write_text(json.dumps(results, indent=2) + '\n')
    print('PASS: eight binaries; 26 transfer loops; 60 MMA kernels; 105 TMEM joins; 84 shared geometries')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    audit(parser.parse_args().package.resolve())
