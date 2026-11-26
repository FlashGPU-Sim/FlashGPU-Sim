"""Bounded, matched B200 resource-service and competition matrix."""
import json


def cases():
    rows = []

    def add(binary, args, family, group, profile, smoke_args, **meta):
        rows.append(dict(name=binary + '-' + '-'.join(str(x).replace('--', '').replace('=', '') for x in args),
                         binary=binary, args=list(map(str, args)), family=family,
                         group=group, profile=profile, smoke_args=list(map(str, smoke_args)), **meta))

    for variant in ('mma_low', 'mma_high', 'mma_col320', 'mma_col448'):
        for shape in (0, 1, 2):
            backgrounds = range(5) if variant == 'mma_low' else (0, 1) if variant == 'mma_high' else (2, 3)
            for bg in backgrounds:
                for ratio in ((8, 32, 128) if variant in ('mma_low', 'mma_high') else (128,)):
                    for rounds in (4, 64):
                        for mode in (1, 2, 3):
                            add(variant, [shape, bg, mode, rounds, ratio], 'mma',
                                f'{variant}-s{shape}-b{bg}-r{rounds}-q{ratio}',
                                rounds == 64 and ratio in (8, 128),
                                [shape, bg, mode, 1, 2], shape=shape, background=bg,
                                active=mode, rounds=rounds, ratio=ratio)
    # 32dp single-path pressure and store-drain controls; keep dependent read
    # timing separate from the per-burst service measurements.
    for mode in (1, 2, 3, 4):
        for width in (4, 32, 64):
            for warps in (1, 4):
                depth = 1 if mode == 4 else 2
                for n in (256, 1024):
                    args = [0, mode, width, depth, warps, 1, n, 1]
                    add('tmem_solo', args, 'solo', f'solo-{mode}-{width}-{depth}-{warps}',
                        n == 1024 and warps == 4, args[:6] + [3, 1],
                        mode=mode, width=width, depth=depth, warps=warps, iterations=n)
    for kind in (0, 1, 2):
        for form in range(5 if kind == 2 else 4):
            for warps in ((4,) if kind == 2 else (1, 4)):
                for depth, drain in ((1, 0), (4, 0), (4, 1)):
                    for n in (256, 1024):
                        add('transfers', [kind, form, depth, warps, n, drain], 'transfer',
                            f'transfer-{kind}-{form}-{depth}-{warps}-{drain}',
                            n == 1024 and warps == 4,
                            [kind, form, depth, warps, 3, drain],
                            kind=kind, form=form, depth=depth, warps=warps,
                            iterations=n, drain=drain)
    for size in (32, 64):
        pairs = (('lds', 'sts'), ('sts', 'sts'), ('lds', 'tma'), ('sts', 'tma'), ('tma', 'tma')) if size == 32 else (('lds', 'tma'), ('sts', 'tma'))
        for left, right in pairs:
            for halves in ('LL', 'LH', 'HL', 'HH'):
                slots = (1, 1) if left == right == 'tma' else (1, 2 if right == 'tma' else 1)
                if left == right == 'tma':
                    addresses = {'LL': (16, 64), 'LH': (16, 144), 'HL': (144, 16), 'HH': (136, 176)}[halves]
                elif right == 'tma' and size == 64:
                    addresses = {'LL': (4, 16), 'LH': (4, 80), 'HL': (160, 16), 'HH': (220, 80)}[halves]
                elif right == 'tma':
                    addresses = (8 if halves[0] == 'L' else 136, 32 if halves[1] == 'L' else 144)
                else:
                    addresses = {'LL': (8, 24), 'LH': (8, 144), 'HL': (144, 8), 'HH': (136, 152)}[halves]
                for mib in (8, 64):
                    for mode in (1, 2, 3):
                        args = [f'--a={left}', f'--b={right}', f'--addr-a={addresses[0]}',
                                f'--addr-b={addresses[1]}', f'--slots-a={slots[0]}',
                                f'--slots-b={slots[1]}', '--warps=8', f'--active={mode}', f'--mib={mib}']
                        add(f'shared{size}', args, 'shared', f'{size}-{left}-{right}-{halves}-{mib}',
                            mib == 64, args[:-1] + ['--mib=1'],
                            active=mode, tile_kib=size, operations=[left, right],
                            addresses_kib=addresses, slots=slots, halves=halves,
                            geometry='dominant-half-crosses-boundary' if size == 64 else 'strict-half', mib=mib)
    assert len({r['name'] for r in rows}) == len(rows)
    return rows


def smoke_cases(rows):
    seen = set()
    for row in rows:
        key = (row['binary'], tuple(row['smoke_args']))
        if key not in seen:
            seen.add(key)
            yield dict(row, args=row['smoke_args'], name='smoke-' + row['name'], profile=False)


if __name__ == '__main__':
    matrix = cases()
    print(json.dumps(dict(cases=len(matrix), smoke_cases=len(list(smoke_cases(matrix))),
                          ncu_full=sum(r['profile'] for r in matrix), matrix=matrix), indent=2))
