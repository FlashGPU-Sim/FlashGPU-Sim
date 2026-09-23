#!/usr/bin/env python3
"""Inspect bounded WGMMA lifecycle evidence; this does not certify calibration.

In a separately rebuilt diagnostic library, enable:
  GPGPU_SIM_WGMMA_COLLECTOR_DEBUG=1
  GPGPU_SIM_WGMMA_LIFECYCLE_SM=0             (default 0)
  GPGPU_SIM_WGMMA_LIFECYCLE_MIN_CYCLE=0      (default 0, cumulative core cycles)
The first 16 REGISTER UIDs on that SM at/after the cycle threshold are selected.
All subsequent events for those UIDs are retained, independent of the existing
128-line RF debug cap. Run the unchanged matched GEMM through its normal harness,
capture stdout/stderr, require kernel correctness and successful exit separately:
  python3 test/check_wgmma_lifecycle.py RUN.log[.gz]
Use a matched K=2048 case for pipeline overlap; a K=64 smoke need not build queues.

FU_ADMIT is actual tensor-FU dispatch-register admission, not the later pipeline
start stage. WGMMA bypasses the optional classic-MMA queue. All timestamps use
cumulative core cycles. UID is the issued representative instruction's UID.
Missing/duplicate events and impossible registration/dispatch/admission ordering
are errors. Completion before admission is reported, not labelled a simulator
bug: scheduler-start timing may be an intentional abstraction. Positive
post_admit_shortfall means the observed admission-to-completion span is shorter
than configured compute+tail; it is not an exclusive measured compute-overlap
duration (backend queues, RF drain, and tick-boundary conventions also matter).
"""

import argparse
import gzip
import statistics


EVENTS = {"REGISTER", "OC_DISPATCH", "FU_ADMIT", "ASYNC_COMPLETE"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check(lines):
    lifecycles = {}
    for line in lines:
        if not line.startswith("WGMMA_LIFECYCLE "):
            continue
        pairs = [field.split("=", 1) for field in line.split()[1:]]
        require(all(len(pair) == 2 for pair in pairs), "malformed trace fields")
        fields = dict(pairs)
        require(len(fields) == len(pairs), "duplicate trace field")
        event = fields["event"]
        require(event in EVENTS, f"unknown lifecycle event {event}")
        values = {name: int(fields[name]) for name in
                  ("uid", "sm", "cycle", "compute", "tail")}
        require(all(value >= 0 for value in values.values()), "negative trace value")
        key = values["sm"], values["uid"]
        events = lifecycles.setdefault(key, {})
        require(event not in events, f"duplicate {event} for {key}")
        require(event == "REGISTER" or "REGISTER" in events,
                f"{event} without prior REGISTER for {key}")
        require(event != "FU_ADMIT" or "OC_DISPATCH" in events,
                f"FU_ADMIT without prior OC_DISPATCH for {key}")
        events[event] = values
    require(lifecycles, "no WGMMA lifecycle evidence; enable diagnostic logging")
    require(len(lifecycles) <= 16, "more than 16 lifecycles; use one isolated run")
    require(len({sm for sm, _ in lifecycles}) == 1, "multiple selected SMs")
    rows = []
    for (sm, uid), events in sorted(lifecycles.items()):
        require(events.keys() == EVENTS, f"incomplete lifecycle for {(sm, uid)}")
        register, dispatch, admit, complete = (
            events[name]["cycle"] for name in
            ("REGISTER", "OC_DISPATCH", "FU_ADMIT", "ASYNC_COMPLETE"))
        require(register <= dispatch <= admit,
                f"out-of-order registration/dispatch/admission for {(sm, uid)}")
        require(register <= complete, f"completion before registration for {(sm, uid)}")
        compute = events["REGISTER"]["compute"]
        tail = events["REGISTER"]["tail"]
        require(compute > 0, "REGISTER requires positive effective compute time")
        rows.append((sm, uid, admit - register, complete - admit,
                     max(0, compute + tail - (complete - admit))))
    return rows


def self_test():
    good = [f"WGMMA_LIFECYCLE event={event} uid=7 sm=0 cycle={cycle} compute=8 tail=4\n"
            for event, cycle in (("REGISTER", 10), ("OC_DISPATCH", 12),
                                 ("FU_ADMIT", 15), ("ASYNC_COMPLETE", 22))]
    require(check(good) == [(0, 7, 5, 7, 5)], "complete lifecycle failed")
    early = good[:2] + [good[3].replace("cycle=22", "cycle=13"), good[2]]
    require(check(early)[0][3] == -2, "early-completion evidence lost")
    for bad in ([], good[:-1], good + [good[-1]], [good[1], good[0]] + good[2:],
                [good[0], good[1].replace("cycle=12", "cycle=16")] + good[2:],
                [good[0].replace("compute=8", "compute=0")] + good[1:],
                [good[0].replace("uid=7", "uid=7 uid=8")] + good[1:]):
        try:
            check(bad)
        except (ValueError, KeyError):
            pass
        else:
            raise ValueError("invalid lifecycle evidence accepted")
    print("self-test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif not args.log:
        parser.error("provide a completed log or --self-test")
    else:
        try:
            opener = gzip.open if args.log.endswith(".gz") else open
            with opener(args.log, "rt") as stream:
                rows = check(stream)
        except (ValueError, KeyError, OSError) as error:
            parser.error(str(error))
        print("sm uid scheduler_to_admit_cycles admit_to_complete_cycles post_admit_shortfall_cycles")
        for row in rows:
            print(*row)
        print(f"Complete lifecycles: {len(rows)}; completion_before_admit: "
              f"{sum(row[3] < 0 for row in rows)}; mean scheduler_to_admit: "
              f"{statistics.mean(row[2] for row in rows):.3f} cycles")
        print("Diagnostic only: these spans do not establish a timing-model bug or calibration acceptance.")
