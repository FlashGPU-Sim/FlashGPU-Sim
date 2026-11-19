#!/usr/bin/env python3
"""Summarize NCU median cycles vs FlashGPU-Sim cycles for the paper GEMM set."""

from __future__ import annotations

import csv
import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "paper_gemm_cases_4060_fullset.csv"
ARTIFACTS = Path(__file__).resolve().parents[3] / "reference/SM89_RTX4060/gemm/full_set"


def ncu_cycles(path: Path) -> float:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    header = rows[0]
    index = header.index("sm__cycles_elapsed.avg")
    return float(rows[2][index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    args = parser.parse_args()
    cases_path = args.cases
    results_path = HERE / f"{cases_path.stem}_results.csv"
    summary_path = HERE / f"{cases_path.stem}_summary.md"
    with cases_path.open(newline="", encoding="utf-8-sig") as stream:
        cases = list(csv.DictReader(stream))

    result_rows = []
    groups: dict[str, list[float]] = defaultdict(list)
    missing = []
    errors: dict[str, str] = {}
    for case in cases:
        case_id, dtype = case["case_id"], case["dtype"]
        directory = ARTIFACTS / dtype / case_id
        reports = sorted((directory / "ncu").glob("reference_*.csv"))
        sim_log = directory / "simulator.log"
        if len(reports) != 5 or not sim_log.exists():
            missing.append(case_id)
            continue
        try:
            hardware = [ncu_cycles(path) for path in reports]
            text = sim_log.read_text(encoding="utf-8", errors="replace")
            sim_match = re.search(r"^gpu_tot_sim_cycle\s*=\s*(\d+)", text, re.MULTILINE)
            valid = "Validation PASSED" in text
            if not sim_match or not valid:
                raise ValueError("sim cycle or output validation missing")
            sim = int(sim_match.group(1))
            hw_median = statistics.median(hardware)
            signed = (sim / hw_median - 1.0) * 100.0
            ape = abs(signed)
            group = case["mape_group"]
            groups[group].append(ape)
            enriched = dict(case)
            enriched.update({
                "run_status": "pass",
                "hw_cycles_median": f"{hw_median:.3f}",
                "hw_cycles_min": f"{min(hardware):.3f}",
                "hw_cycles_max": f"{max(hardware):.3f}",
                "sim_cycles": str(sim),
                "signed_error_pct": f"{signed:.6f}",
                "abs_error_pct": f"{ape:.6f}",
                "output_status": "pass",
            })
            result_rows.append(enriched)
        except Exception as exc:  # retain failed rows for a complete audit trail
            missing.append(f"{case_id}: {exc}")
            errors[case_id] = str(exc)

    successful_rows = list(result_rows)
    successful_ids = {row["case_id"] for row in successful_rows}
    for case in cases:
        case_id = case["case_id"]
        if case_id not in successful_ids:
            incomplete = dict(case)
            incomplete.update({
                "run_status": "incomplete",
                "hw_cycles_median": "", "hw_cycles_min": "", "hw_cycles_max": "",
                "sim_cycles": "", "signed_error_pct": "", "abs_error_pct": "",
                "output_status": errors.get(case_id, "missing artifacts"),
            })
            result_rows.append(incomplete)

    fieldnames = list(cases[0]) + ["run_status"]
    with results_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    lines = [
        "# RTX 4060 paper GEMM full-set cycle comparison",
        "",
        f"Completed cases: **{len(successful_rows)} / {len(cases)}**",
        "",
        "Hardware reference is the median of five NCU `sm__cycles_elapsed.avg` measurements. "
        "Per-case error is `abs(sim_cycles / hw_median - 1) * 100`; MAPE is the arithmetic mean of those errors.",
        "",
        "| Group | Completed | MAPE | Signed mean error | Max APE |",
        "|---|---:|---:|---:|---:|",
    ]
    for group, values in sorted(groups.items()):
        selected = [r for r in successful_rows if r["mape_group"] == group]
        signed_values = [float(r["signed_error_pct"]) for r in selected]
        lines.append(
            f"| {group} | {len(values)} | {statistics.mean(values):.3f}% | "
            f"{statistics.mean(signed_values):+.3f}% | {max(values):.3f}% |"
        )
    all_apes = [float(row["abs_error_pct"]) for row in successful_rows]
    all_signed = [float(row["signed_error_pct"]) for row in successful_rows]
    if all_apes:
        label = "Full set" if len(successful_rows) == len(cases) else "Completed subset (provisional)"
        emphasis = "**" if len(successful_rows) == len(cases) else ""
        lines.extend([
            f"| {emphasis}{label}{emphasis} | {emphasis}{len(all_apes)}{emphasis} | "
            f"{emphasis}{statistics.mean(all_apes):.3f}%{emphasis} | "
            f"{emphasis}{statistics.mean(all_signed):+.3f}%{emphasis} | "
            f"{emphasis}{max(all_apes):.3f}%{emphasis} |",
        ])
    lines.extend(["", f"Incomplete/invalid cases: **{len(missing)}**"])
    if missing:
        lines.extend(["", *[f"- `{item}`" for item in missing]])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"RESULTS={results_path}")
    print(f"SUMMARY={summary_path}")
    print(f"COMPLETED={len(successful_rows)}/{len(cases)}")
    print(f"INCOMPLETE={len(missing)}")


if __name__ == "__main__":
    main()
