#!/usr/bin/env python3
"""Compare canonical H200 calibration CSV rows with simulator metrics."""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

HEADER = ("suite", "metric", "sim_knob", "median", "p90", "mean", "unit", "notes")
KEY = ("suite", "metric", "sim_knob", "unit")


def validate_gemm_correctness(metrics):
    """Reject explicit failures; historical rows need not contain every flag."""
    shapes = set()
    for row in metrics:
        timing = re.fullmatch(r"gemm_(?:unicast_B|mcast_B|notma)_(.+)_ms", row["metric"])
        if not timing:
            continue
        shapes.add(timing[1])
        if re.search(r"\b(?:ok|ref_ok|notma_ok|finite_ok)=0\b", row.get("notes", "")):
            raise ValueError(f"failed GEMM correctness: {row['metric']} in {row.get('source', '')}")
    for row in metrics:
        check = re.fullmatch(r"gemm_(?:reference|c_match|notma_c_match)_(.+)", row["metric"])
        if check and check[1] in shapes and row["unit"] == "bool" and float(row["median"]) != 1:
            raise ValueError(f"failed GEMM correctness: {row['metric']} in {row.get('source', '')}")


def gemm_event_rows(metrics, require_complete=False, cases=None):
    """Select requested event timings (all nine by default), never cycle estimates."""
    validate_gemm_correctness(metrics)
    shapes = {"cyc_1e5": (256, 8192, 2048), "k2k": (512, 16384, 2048),
              "sq_2k": (2048, 2048, 8192)}
    names = [name.strip() for name in cases.split(",")] if cases is not None else list(shapes)
    if not names or len(names) != len(set(names)) or any(name not in shapes for name in names):
        raise ValueError("invalid GEMM cases: use distinct names from cyc_1e5,k2k,sq_2k")
    expected = {f"{variant}_{case}_ms": shape
                for case, shape in shapes.items() if case in names
                for variant in ("gemm_unicast_B", "gemm_mcast_B", "gemm_notma")}
    selected = []
    for row in metrics:
        shape = expected.get(row["metric"])
        if shape is None:
            continue
        dimensions = re.search(r"\bM=(\d+) N=(\d+) K=(\d+)\b", row.get("notes", ""))
        if row["unit"] != "ms" or not dimensions or tuple(map(int, dimensions.groups())) != shape:
            raise ValueError(f"wrong GEMM shape/unit: {row['metric']} in {row.get('source', '')}")
        selected.append(row)
    missing = expected.keys() - {row["metric"] for row in selected}
    if require_complete and missing:
        raise ValueError(f"missing required hardware GEMM event timings: {sorted(missing)}")
    return selected


def dsm_fit_metrics(text):
    for row in csv.DictReader(text.splitlines()):
        if "aggregate_bytes_per_cycle" not in row or "intercept_cycles" not in row:
            return
        value = row["aggregate_bytes_per_cycle"]
        yield dict(suite="dsm_bandwidth", metric=row["label"],
                   sim_knob="DSM aggregate slope", median=value, p90="", mean="",
                   unit="bytes/cycle", notes=f"size-slope fit; samples={row['samples']}; "
                   f"intercept={row['intercept_cycles']} cycles; RMSE={row['rmse_cycles']} cycles")


def tma_latency_metrics(text):
    for row in csv.DictReader(text.splitlines()):
        if not {'case', 'description', 'sequence', 'median_cycles'} <= row.keys():
            return
        yield dict(suite="tma_latency", metric=row['case'], sim_knob="TMA completion",
                   median=row['median_cycles'], p90=row['p90_cycles'], mean="",
                   unit="cycles", notes=f"{row['description']}; {row['sequence']}")


def hbm_metrics(text):
    """Select the vendor hardware points matched by the three simple drivers."""
    if "Benchmark mode: HBM-streaming" not in text:
        return
    metric = None
    names = {"TMA": "tma", "cp.async": "cp_async", "Normal load": "normal_load"}
    for line in text.splitlines():
        if "producer warps | CHUNK=" in line:
            match = re.fullmatch(
                r"(TMA|cp\.async|Normal load) \+ 16 producer warps \| "
                r"CHUNK=8192 B \| total=1073741824 B \| repeat=1 \| "
                r"warmup=\d+ \| iters=\d+", line)
            metric = names[match[1]] if match else None
        if metric is None:
            continue
        match = re.fullmatch(
            r"Passed \| Stages=\s*16 \| Chunk=\s*8192 \| Warmup=\d+ \| "
            r"Iters=\d+ \| Time=([\d.]+) ms \| Data=1024\.00 MiB \| "
            r"BW=([\d.]+) GB/s \| cycles=(\d+)", line)
        if match and int(match[3]) > 0:
            value = str((1 << 30) / int(match[3]))
            yield dict(suite="hbm", metric=metric, sim_knob="HBM aggregate",
                       median=value, p90="", mean="", unit="bytes/cycle",
                       notes=f"1 GiB HBM stream; 16 producer warps, 16 stages, "
                       f"8192 B chunk, repeat=1; vendor aggregate {match[2]} GB/s; "
                       f"bytes / vendor-reported {match[3]} cycles; not a sample median")


def rows(path: Path, simulator=False):
    text = path.read_text(errors="replace")
    if not simulator and "Benchmark mode: HBM-streaming" in text:
        return [dict(row, source=str(path)) for row in hbm_metrics(text)]
    if not simulator and "aggregate_bytes_per_cycle" in text.partition("\n")[0]:
        return [dict(row, source=str(path)) for row in dsm_fit_metrics(text)]
    if not simulator and "clock64_overhead,min_cycles" in text.partition("\n")[0]:
        return [dict(row, source=str(path)) for row in tma_latency_metrics(text)]
    found, active = [], False
    expected = (("case",) + HEADER) if simulator else HEADER
    for row in csv.reader(text.splitlines()):
        if tuple(row) == expected:
            active = True
            continue
        if not active:
            continue
        offset = 1 if simulator else 0
        if len(row) < len(expected):
            active = False
            continue
        try:
            float(row[offset + 3])
        except ValueError:
            active = False
            continue
        item = dict(zip(expected, row[: len(expected)]))
        item["source"] = str(path)
        found.append(item)
    return found


def compare(hardware, simulator, threshold):
    validate_gemm_correctness(hardware)
    validate_gemm_correctness(simulator)
    sim, duplicates = {}, set()
    for row in simulator:
        key = tuple(row[k] for k in KEY)
        if key in sim:
            duplicates.add(key)
        sim[key] = row
    output, seen = [], set()
    for hw in hardware:
        key = tuple(hw[k] for k in KEY)
        if key in seen:
            raise ValueError(f"duplicate hardware metric: {key}")
        seen.add(key)
        if key in duplicates:
            raise ValueError(f"duplicate simulator metric: {key}")
        sr = sim.get(key)
        hv = float(hw["median"])
        sv = float(sr["median"]) if sr else None
        error = None
        if not math.isfinite(hv) or hv <= 0:
            status = "INVALID_REFERENCE"
        elif not sr:
            status = "MISSING_SIM"
        elif not math.isfinite(sv):
            status = "INVALID_SIM"
        else:
            error = abs(sv - hv) / hv * 100
            status = "PASS" if error < threshold else "FAIL"
        output.append({**dict(zip(KEY, key)), "hardware": hw["median"],
                       "sim": sr["median"] if sr else "", "error_pct": f"{error:.3f}" if error is not None else "",
                       "status": status, "hardware_source": hw["source"],
                       "hardware_notes": hw.get("notes", ""),
                       "sim_source": sr.get("source", "") if sr else "",
                       "sim_case": sr.get("case", "") if sr else ""})
    return output


def write(output, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    fields = (*KEY, "hardware", "sim", "error_pct", "status", "hardware_source", "hardware_notes", "sim_source", "sim_case")
    with (directory / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader(); writer.writerows(output)
    lines = ["# H200 hardware / simulator comparison", "",
             "Relative error requires a positive finite hardware reference. "
             "INVALID_REFERENCE rows remain visible but are not percentage-test passes or failures; "
             "they prevent an all-accepted exit status. Knob names are source metadata, not proof "
             "that an individual knob is isolated by the measurement.", "",
             "| Suite | Metric | Knob | Unit | Hardware | Simulator | Error | Status |", "|---|---|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['suite']} | {r['metric']} | `{r['sim_knob']}` | {r['unit']} | {r['hardware']} | {r['sim']} | {r['error_pct'] + '%' if r['error_pct'] else 'N/A'} | {r['status']} |" for r in output]
    lines += ["", "Metric provenance:", ""]
    lines += [f"- `{r['suite']}/{r['metric']}`: H200 `{r['hardware_source']}`; "
              f"simulator `{r['sim_source']}`, case `{r['sim_case']}`. {r['hardware_notes']}"
              for r in output]
    (directory / "comparison.md").write_text("\n".join(lines) + "\n")


def self_test():
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "hw.csv").write_text(",".join(HEADER) + "\na,b,-x,100,100,100,cycles,n\n")
        (base / "sim.csv").write_text("case," + ",".join(HEADER) + "\nc,a,b,-x,108,108,108,cycles,n\n")
        result = compare(rows(base / "hw.csv"), rows(base / "sim.csv", True), 10)
        assert result[0]["status"] == "PASS" and result[0]["error_pct"] == "8.000"
        (base / "hw2.csv").write_text((base / "hw.csv").read_text().replace("a,b,", "a,b2,"))
        (base / "sim2.csv").write_text((base / "sim.csv").read_text().replace("a,b,", "a,b2,"))
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--hardware", str(base / "hw.csv"), str(base / "hw2.csv"),
                   "--output-dir", str(base / "out"), "--sim", str(base / "sim.csv")]
        subprocess.run(command + [str(base / "sim2.csv")], check=True, capture_output=True)
        with (base / "out/comparison.csv").open() as handle:
            combined = list(csv.DictReader(handle))
        assert len(combined) == 2 and all(row["status"] == "PASS" for row in combined)
        assert {row["sim_source"] for row in combined} == {str(base / "sim.csv"), str(base / "sim2.csv")}
        duplicate = subprocess.run(command + [str(base / "sim.csv")], capture_output=True, text=True)
        assert duplicate.returncode != 0 and "duplicate simulator metric" in duplicate.stderr
        assert compare(rows(base / "hw.csv"), rows(base / "sim.csv", True), 8)[0]["status"] == "FAIL"
        hw, sim = rows(base / "hw.csv"), rows(base / "sim.csv", True)
        for invalid in ("0", "-1", "nan", "inf"):
            hw[0]["median"] = invalid
            result = compare(hw, sim, 10)[0]
            assert result["status"] == "INVALID_REFERENCE" and result["error_pct"] == ""
        hw[0]["median"] = "100"
        sim[0]["median"] = "nan"
        assert compare(hw, sim, 10)[0]["status"] == "INVALID_SIM"
        fit = list(dsm_fit_metrics("label,samples,aggregate_bytes_per_cycle,intercept_cycles,rmse_cycles\nload_size_bi,11,30.7,461.4,60.3\n"))
        assert fit[0]["median"] == "30.7" and fit[0]["unit"] == "bytes/cycle"
        latency = list(tma_latency_metrics("case,description,sequence,median_cycles,p90_cycles\nTMALoad4KB,bulk load,load+wait,1649,1862\n"))
        assert latency[0]['median'] == '1649' and latency[0]['unit'] == 'cycles'
        hbm = ("Benchmark mode: HBM-streaming\n"
               "TMA + 16 producer warps | CHUNK=8192 B | total=1073741824 B | repeat=1 | warmup=1 | iters=10\n"
               "Passed | Stages=16 | Chunk=8192 | Warmup=1 | Iters=10 | Time=0.262 ms | Data=1024.00 MiB | BW=4103.61 GB/s | cycles=467059\n")
        target = list(hbm_metrics(hbm))
        assert len(target) == 1 and target[0]['metric'] == 'tma'
        assert float(target[0]['median']) == (1 << 30) / 467059
        for old, new in [('HBM-streaming', 'L2-resident'), ('+ 16 producer', '+ 8 producer'),
                         ('Stages=16', 'Stages=32'), ('total=1073741824', 'total=268435456')]:
            assert not list(hbm_metrics(hbm.replace(old, new)))
        events = [dict(suite="gemm", metric=f"{variant}_{case}_ms", sim_knob=variant,
                       median="1", unit="ms", notes=f"M={m} N={n} K={k}", source="test")
                  for case, m, n, k in (("cyc_1e5", 256, 8192, 2048),
                                       ("k2k", 512, 16384, 2048),
                                       ("sq_2k", 2048, 2048, 8192))
                  for variant in ("gemm_unicast_B", "gemm_mcast_B", "gemm_notma")]
        assert len(gemm_event_rows(events, True)) == 9
        assert not gemm_event_rows([dict(events[0], metric="gemm_unicast_B_cyc_1e5_cycles_est", unit="cycles")])
        assert compare(events, gemm_event_rows(events[:3]), 10)[-1]["status"] == "MISSING_SIM"
        for bad in (events[:3], [dict(events[0], notes="M=1 N=1 K=1"), *events[1:]]):
            try:
                gemm_event_rows(bad, True)
            except ValueError:
                pass
            else:
                raise AssertionError("incomplete or mismatched GEMM reference accepted")
        # Equal timings must not hide a failed numerical check, even when the
        # event-only CLI drops bool rows. Explicit checks also run under -O.
        failures = [
            [*events, dict(events[0], metric=f"gemm_{kind}_cyc_1e5", unit="bool", median="0")]
            for kind in ("reference", "c_match", "notma_c_match")
        ]
        failures += [[dict(events[0], notes=events[0]["notes"] + f" {flag}=0"), *events[1:]]
                     for flag in ("ok", "ref_ok", "notma_ok", "finite_ok")]
        for failed in failures:
            for hardware_bad in (False, True):
                hw, sim = (failed, events) if hardware_bad else (events, failed)
                try:
                    compare(hw, sim, 10)
                except ValueError as error:
                    if "failed GEMM correctness" not in str(error):
                        raise
                else:
                    raise AssertionError("failed GEMM correctness accepted by compare")
                for path, data, fields in ((base / "gemm_hw.csv", hw, HEADER),
                                           (base / "gemm_sim.csv", sim, ("case",) + HEADER)):
                    with path.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
                        writer.writeheader()
                        writer.writerows(data)
                rejected = subprocess.run(
                    [sys.executable, *(["-O"] if not __debug__ else []), str(Path(__file__).resolve()),
                     "--hardware", str(base / "gemm_hw.csv"), "--sim", str(base / "gemm_sim.csv"),
                     "--gemm-events-only", "--output-dir", str(base / "gemm_out")],
                    capture_output=True, text=True)
                if rejected.returncode == 0 or "failed GEMM correctness" not in rejected.stderr:
                    raise AssertionError("event-only CLI accepted failed GEMM correctness")
        historical = [dict(row, notes=row["notes"] + " ok=1 ref_ok=1") for row in events]
        validate_gemm_correctness([
            *historical,
            dict(events[0], metric="gemm_notma_c_match_cyc_1e5", unit="bool", median="1"),
            dict(events[0], metric="gemm_notma_c_match_unselected", unit="bool", median="0"),
        ])
        if any(row["status"] != "PASS" for row in compare(
                gemm_event_rows(historical, True), gemm_event_rows(historical), 10)):
            raise AssertionError("missing historical notma_ok must not imply failure")
        six = gemm_event_rows(events, True, cases="cyc_1e5,k2k")
        if len(six) != 6 or any("sq_2k" in row["metric"] for row in six):
            raise AssertionError("deferred square must be excluded")
        if len(gemm_event_rows(events, True)) != 9:
            raise AssertionError("default must still require all nine timings")
        missing = compare(six, gemm_event_rows(events[:5], cases="cyc_1e5,k2k"), 10)
        if len(missing) != 6 or missing[-1]["status"] != "MISSING_SIM":
            raise AssertionError("missing selected simulator timing must remain visible")
        for selected in ("", "unknown", "cyc_1e5,", "cyc_1e5,cyc_1e5"):
            try:
                gemm_event_rows(events, cases=selected)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid case selector accepted")
        try:
            gemm_event_rows(events[:5], True, cases="cyc_1e5,k2k")
        except ValueError as error:
            if "missing required hardware" not in str(error):
                raise
        else:
            raise AssertionError("missing selected hardware timing accepted")
        # Exercise argument wiring with only six hardware rows and nine sim rows.
        for path, data, fields in ((base / "gemm_hw.csv", six, HEADER),
                                   (base / "gemm_sim.csv", events, ("case",) + HEADER)):
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(data)
        command = [sys.executable, *(["-O"] if not __debug__ else []), str(Path(__file__).resolve()),
                   "--hardware", str(base / "gemm_hw.csv"), "--sim", str(base / "gemm_sim.csv"),
                   "--gemm-events-only", "--output-dir", str(base / "gemm_out")]
        subprocess.run(command + ["--gemm-cases", "cyc_1e5,k2k"], check=True, capture_output=True)
        with (base / "gemm_out/comparison.csv").open() as handle:
            selected_result = list(csv.DictReader(handle))
        if len(selected_result) != 6 or any(row["status"] != "PASS" for row in selected_result):
            raise AssertionError("six-timing CLI comparison failed")
        default = subprocess.run(command, capture_output=True, text=True)
        if default.returncode == 0 or "missing required hardware" not in default.stderr:
            raise AssertionError("default CLI accepted missing square hardware timings")
    print("self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", type=Path, nargs="+")
    parser.add_argument("--sim", type=Path, nargs="+",
                        help="Simulator metrics CSVs; duplicate comparison keys are rejected.")
    parser.add_argument("--output-dir", type=Path, default=Path("calibration/results/h200_comparison_final"))
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--gemm-events-only", action="store_true",
                        help="Compare required GEMM event timings; validate shape metadata (all nine by default).")
    parser.add_argument("--gemm-cases",
                        help="With --gemm-events-only: comma-separated cases (default: cyc_1e5,k2k,sq_2k).")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return 0
    if args.gemm_cases is not None and not args.gemm_events_only:
        parser.error("--gemm-cases requires --gemm-events-only")
    if not args.hardware or not args.sim:
        parser.error("--hardware and --sim are required")
    hardware = []
    for path in args.hardware:
        found = rows(path)
        if not found:
            parser.error(f"no supported hardware metrics in {path}")
        hardware.extend(found)
    simulator = []
    for path in args.sim:
        found = rows(path, True)
        if not found:
            parser.error(f"no supported simulator metrics in {path}")
        simulator.extend(found)
    if args.gemm_events_only:
        hardware = gemm_event_rows(hardware, require_complete=True, cases=args.gemm_cases)
        simulator = gemm_event_rows(simulator, cases=args.gemm_cases)
    result = compare(hardware, simulator, args.threshold)
    write(result, args.output_dir)
    return int(any(row["status"] != "PASS" for row in result))


if __name__ == "__main__":
    sys.exit(main())
