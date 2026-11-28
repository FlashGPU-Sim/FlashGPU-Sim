#!/usr/bin/env python3
"""Report matched slowdown, worker overlap, and measured pressure, not port guesses."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics


def parse(row, text):
    result = {}
    if row['family'] == 'mma':
        clocks = {int(w): (int(a), int(b)) for w, a, b in re.findall(r'^CLOCK,(\d+),(\d+),(\d+)$', text, re.M)}
        mode = row['active']
        if mode & 1:
            result['mma_cycles'] = clocks[0][1] - clocks[0][0]
            result['mma_cycles_per_instruction'] = result['mma_cycles'] / (row['rounds'] * 32)
        if mode & 2:
            durations = [clocks[w][1] - clocks[w][0] for w in range(1, 5)]
            result['background_cycles'] = max(durations)
            payload = row['rounds'] * row['ratio'] * (16384 if row['background'] < 2 else 32768 if row['background'] < 4 else 0)
            result['background_Bpc'] = payload / (max(clocks[w][1] for w in range(1, 5)) - min(clocks[w][0] for w in range(1, 5)))
        if mode == 3:
            start = max(clocks[w][0] for w in range(5))
            stop = min(clocks[w][1] for w in range(5))
            result['mma_coverage_by_all_background_warps'] = max(0, stop - start) / result['mma_cycles']
            bg_start = min(clocks[w][0] for w in range(1, 5))
            bg_end = max(clocks[w][1] for w in range(1, 5))
            overlap = max(0, min(bg_end, clocks[0][1]) - max(bg_start, clocks[0][0]))
            result['background_coverage_by_mma'] = overlap / (bg_end - bg_start)
    elif row['family'] == 'shared':
        for s, payload, cycles in re.findall(r'stream=(\d+) bytes=(\d+) cycles=(\d+)', text):
            if int(payload):
                result['stream' + s + '_cycles'] = int(cycles)
                result['stream' + s + '_Bpc'] = int(payload) / int(cycles)
        result['wall_cycles'] = int(re.search(r'wall_cycles=(\d+)', text)[1])
        result['combined_Bpc'] = float(re.search(r'combined_Bpc=([0-9.]+)', text)[1])
        if row['active'] == 3:
            result['shorter_stream_fraction'] = min(result['stream0_cycles'], result['stream1_cycles']) / max(result['stream0_cycles'], result['stream1_cycles'])
    elif row['family'] == 'transfer':
        clocks = [(int(a), int(b)) for _, a, b in re.findall(r'^CLOCK,(\d+),(\d+),(\d+)$', text, re.M) if int(b) > int(a)]
        result['cycles'] = max(b for a, b in clocks) - min(a for a, b in clocks)
        result['cycles_per_iteration'] = result['cycles'] / row['iterations']
        if row['kind'] < 2:
            payload = row['warps'] * 32 * (8 if row['form'] == 3 else 4) * 4
            result['payload_bytes_per_iteration'] = payload * row['depth']
        else:
            result['shared_source_bytes_per_iteration'] = (4096, 2048, 1024, 1024, 512)[row['form']] * row['depth']
            result['payload_bytes_per_iteration'] = (4096 if row['form'] == 0 else 2048) * row['depth']
        result['logical_payload_Bpc'] = result['payload_bytes_per_iteration'] / result['cycles_per_iteration']
    else:
        lines = re.findall(r'^RESULT,(.+)$', text, re.M)
        values = list(map(int, lines[-1].split(',')))
        result['cycles'] = max(values[8:])
        result['cycles_per_iteration'] = result['cycles'] / row['iterations']
    return result


def slopes(rows):
    """Two-length slopes retain loop/drain cost, but remove fixed interval overhead."""
    groups = {}
    for row in rows:
        family = row['family']
        if family == 'mma':
            fields, x = ('binary', 'shape', 'background', 'active', 'ratio'), 'rounds'
            ys = ['mma_cycles', 'background_cycles']
        elif family == 'shared':
            fields, x = ('binary', 'operations', 'addresses_kib', 'slots', 'active'), 'mib'
            ys = ['stream0_cycles', 'stream1_cycles', 'wall_cycles']
        else:
            fields = ('binary', 'kind', 'form', 'depth', 'warps', 'drain') if family == 'transfer' else ('binary', 'mode', 'width', 'depth', 'warps')
            x, ys = 'iterations', ['cycles']
        key = json.dumps([row.get(f) for f in fields])
        groups.setdefault(key, (x, ys, []))[2].append(row)
    out = []
    for x, ys, points in groups.values():
        points.sort(key=lambda r: r[x])
        if len(points) < 2:
            continue
        low, high = points[0], points[-1]
        result = dict(low_case=low['name'], high_case=high['name'], sweep=x)
        for y in ys:
            if y in low and y in high:
                slope = (high[y] - low[y]) / (high[x] - low[x])
                result[y + '_per_' + x] = slope
                result[y + '_intercept'] = low[y] - slope * low[x]
        out.append(result)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results', type=Path)
    a = p.parse_args()
    manifest = json.loads((a.results / 'manifest.json').read_text())
    rows, groups = [], {}
    for case in manifest['matrix']:
        records = []
        for log in sorted((a.results / case['name']).glob('native-*.log')):
            text = log.read_text(errors='replace')
            if 'verification=PASS' in text or 'CHECK,PASS,bad=0' in text:
                records.append(parse(case, text))
        if not records:
            continue
        row = dict(case, repetitions=len(records))
        for key in records[0]:
            row[key] = statistics.median(r[key] for r in records)
        rows.append(row)
        if 'active' in row:
            groups.setdefault(row['group'], {})[row['active']] = row
    for modes in groups.values():
        if not all(m in modes for m in (1, 2, 3)):
            continue
        left, right, mix = (modes[m] for m in (1, 2, 3))
        if mix['family'] == 'mma':
            mix['mma_slowdown'] = mix['mma_cycles'] / left['mma_cycles']
            mix['background_slowdown'] = mix['background_cycles'] / right['background_cycles']
            mix['mma_observation_covered'] = mix['mma_coverage_by_all_background_warps'] >= 0.9
            mix['background_observation_covered'] = mix['background_coverage_by_mma'] >= 0.9
            mix['overlap_sufficient'] = mix['mma_observation_covered'] and mix['background_observation_covered']
        else:
            for s, pure in ((0, left), (1, right)):
                mix[f'stream{s}_slowdown'] = mix[f'stream{s}_cycles'] / pure[f'stream{s}_cycles']
            mix['overlap_sufficient'] = mix['shorter_stream_fraction'] >= 0.9
    # Relative pressure gates do not establish a physical peak. Preserve both
    # absolute B/cycle and comparisons with the best observed same operation.
    best = {}
    for row in rows:
        if row['family'] == 'mma' and row['active'] == 2 and row['background'] < 4:
            key = ('mma', row['background'])
            best[key] = max(best.get(key, 0), row['background_Bpc'])
        if row['family'] == 'shared' and row['active'] in (1, 2):
            s = row['active'] - 1
            key = ('shared', row['operations'][s])
            best[key] = max(best.get(key, 0), row[f'stream{s}_Bpc'])
    for modes in groups.values():
        if not all(m in modes for m in (1, 2, 3)):
            continue
        mix = modes[3]
        if mix['family'] == 'mma' and mix['background'] < 4:
            mix['pure_pressure_fraction_of_best'] = modes[2]['background_Bpc'] / best[('mma', mix['background'])]
        elif mix['family'] == 'shared':
            mix['pure_pressure_fraction_of_best'] = min(modes[s + 1][f'stream{s}_Bpc'] / best[('shared', mix['operations'][s])] for s in (0, 1))
        if 'pure_pressure_fraction_of_best' in mix:
            mix['pressure_sufficient_relative'] = mix['pure_pressure_fraction_of_best'] >= 0.9
            mix['no_contention_claim_allowed_by_gates'] = mix['pressure_sufficient_relative'] and mix['overlap_sufficient']
    report = dict(scope='Unprofiled native clocks only. NCU full reports remain separate. Passing pressure/overlap gates is necessary, not proof of a physical port model. Transfer bytes are logical payload; broadcast copies distinguish shared source and TMEM destination bytes. Slopes retain loop, wait and drain cost.', rows=rows, slopes=slopes(rows))
    (a.results / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (a.results / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print('Summarized', len(rows), 'cases')


if __name__ == '__main__':
    main()
