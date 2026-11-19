"""Summarize five-pass NCU medians versus FlashGPU-Sim cycles."""
from __future__ import annotations
import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "reference" / "SM89_RTX4060" / "gemm"
OUTPUT = ROOT / "gemm_cycle_comparison.csv"


def main() -> None:
    rows = []
    for manifest_path in sorted(ROOT.glob("**/experiment-manifest.json")):
        case_dir = manifest_path.parent
        case = json.loads(manifest_path.read_text(encoding="utf-8"))["case"]
        paths = [case_dir / "ncu" / f"reference_{rep}.csv" for rep in range(1, 6)]
        sim_log = case_dir / "simulator.log"
        if not all(path.is_file() for path in paths) or not sim_log.is_file():
            print(f"INCOMPLETE {case['case_id']}: missing NCU reports or simulator log")
            continue
        values, frequencies = [], []
        for path in paths:
            with path.open(newline="", encoding="utf-8") as stream:
                raw = list(csv.reader(stream))
            data = dict(zip(raw[0], raw[2]))
            values.append(float(data["sm__cycles_elapsed.avg"].replace(",", "")))
            frequencies.append(float(data["sm__cycles_elapsed.avg.per_second"].replace(",", "")))
        text = sim_log.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^gpu_tot_sim_cycle\s*=\s*(\d+)", text, re.MULTILINE)
        if not match or "Validation PASSED" not in text:
            print(f"INVALID {case_dir}: missing simulator cycles or passing output validation")
            continue
        hw = statistics.median(values)
        sim = int(match.group(1))
        rows.append({"split": case["split"], "case_id": case["case_id"],
                     "dtype": case["dtype"], "M": case["M"], "N": case["N"], "K": case["K"],
                     "hw_cycles_median": hw, "sim_cycles": sim,
                     "signed_error_pct": (sim - hw) / hw * 100,
                     "hw_frequency_ghz_median": statistics.median(frequencies),
                     "hw_cycles_min": min(values), "hw_cycles_max": max(values)})
    if not rows:
        raise SystemExit("No complete case measurements found")
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE={OUTPUT}")
    for split, dtype in sorted({(row["split"], row["dtype"]) for row in rows}):
        values = [abs(row["signed_error_pct"]) for row in rows if row["split"] == split and row["dtype"] == dtype]
        print(f"{split}/{dtype}: cases={len(values)} MAPE={statistics.mean(values):.3f}%")


if __name__ == "__main__":
    main()
