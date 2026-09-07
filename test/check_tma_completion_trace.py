#!/usr/bin/env python3
"""Check that architectural TMA completion follows actual memory completion."""

import csv
import sys
from pathlib import Path


def check(text):
    completed = {}
    arrivals = 0
    for row in csv.DictReader(text.splitlines()):
        if row['event'] == 'event':  # concatenated traces from separate runs
            completed.clear()
            continue
        uid, cycle = row['tx_uid'], int(row['cycle'])
        if row['event'] == 'NEW':
            completed.pop(uid, None)
        if row['event'] == 'COMPLETE':
            completed[uid] = cycle
        if row['event'] != 'ARCH_ARRIVE':
            continue
        assert uid in completed and cycle >= completed[uid], row
        assert int(row['bytes_completed']) >= int(row['size']), row
        assert int(row['tx_inflight']) == 0, row
        arrivals += 1
    assert arrivals, 'trace contains no architectural TMA completion'
    return arrivals


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        header = 'cycle,event,tx_uid,size,bytes_completed,tx_inflight\n'
        complete = '100,COMPLETE,1,256,256,0\n'
        arrive = '101,ARCH_ARRIVE,1,256,256,0\n'
        assert check(header + complete + arrive) == 1
        for bad in (header + arrive + complete,
                    header + complete + '101,ARCH_ARRIVE,1,256,128,1\n'):
            try:
                check(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError('accepted early/incomplete arrival')
        print('self-test passed')
    else:
        print(f'{check(Path(sys.argv[1]).read_text())} TMA completions verified')
