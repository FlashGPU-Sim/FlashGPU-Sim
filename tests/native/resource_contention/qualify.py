#!/usr/bin/env python3
"""Join local replay evidence by exact executable hash and arguments, not names."""
import argparse
import hashlib
import json
from pathlib import Path
from plan import cases, smoke_cases


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('reports', nargs='+', type=Path)
    args = parser.parse_args()
    matrix = list(smoke_cases(cases()))
    binaries = {r['binary']: digest(args.package / 'bin' / r['binary']) for r in matrix}
    required = {}
    for row in matrix:
        fronts = ('sass-timing', 'ptx')
        for front in fronts:
            key = (front, binaries[row['binary']], tuple(row['smoke_args']))
            required[key] = dict(frontend=front, binary=row['binary'], args=row['smoke_args'])
    observed, receipts = {}, []
    for path in args.reports:
        report = json.loads(path.read_text())
        assert report['status'] == 'PASS', (str(path), report['status'])
        assert len(report['runs']) == report['total']
        assert digest(Path(report['runtime'])) == report['runtime_sha256']
        receipts.append(dict(path=str(path.resolve()), sha256=digest(path),
                             runtime_sha256=report['runtime_sha256']))
        for run in report['runs']:
            assert run['passed'] and run['returncode'] == 0, run['name']
            key = (run['frontend'], run['binary_sha256'], tuple(run['args']))
            if key not in required:
                raise RuntimeError('Replay does not belong to this package: ' + run['name'])
            log = Path(run['log'])
            assert digest(log.parent / 'gpgpusim.config') == run['config_sha256']
            text = log.read_text(errors='replace')
            assert 'verification=PASS' in text or 'CHECK,PASS,bad=0' in text
            assert 'verification=FAIL' not in text
            observed[key] = dict(required[key], runtime_sha256=report['runtime_sha256'],
                                 log=str(log), log_sha256=digest(log),
                                 command_sha256=digest(log.parent / 'command.json'),
                                 config_sha256=run['config_sha256'])
    missing = set(required) - set(observed)
    assert not missing, (len(missing), 'missing qualification entries')
    summary = dict(status='PASS', package=str(args.package.resolve()),
                   binaries=binaries, sass_smoke=sum(k[0] == 'sass-timing' for k in observed),
                   ptx_smoke=sum(k[0] == 'ptx' for k in observed), reports=receipts,
                   hardware_collected=False, calibrated_timing=False,
                   scope='All packaged smoke cases through both SASS and PTX. Per-report frozen runtimes retained. This is numerical/local preparation, not native port calibration.',
                   entries=list(observed.values()))
    args.out.write_text(json.dumps(summary, indent=2) + '\n')
    print(summary['status'], summary['sass_smoke'], 'SASS,', summary['ptx_smoke'], 'PTX')


if __name__ == '__main__':
    main()
