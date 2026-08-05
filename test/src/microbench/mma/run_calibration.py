#!/usr/bin/env python3
"""
GPGPU-Sim calibration runner for the MMA microbenchmarks.

This script runs the committed MMA calibration benches on native hardware and
under GPGPU-Sim, captures the per-variant output files, and generates
comparison plots/CSVs in test/calibration_results/.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DIR = SCRIPT_DIR.parents[2]
PROJECT_ROOT = TEST_DIR.parent
OUTPUT_DIR = TEST_DIR / "calibration_results"
RUN_BASE_DIR = TEST_DIR / "run"
SETUP_ENVIRONMENT = PROJECT_ROOT / "setup_environment"
DEFAULT_CONFIG = "SM120_RTX5090"

RUN_TARGETS = [
    {
        "pattern": "ILPMinimal",
        "file_prefix": "MMAIssueTest.ILPMinimal.",
    },
    {
        "pattern": "MultiWarpMinimal",
        "file_prefix": "MMAIssueTest.MultiWarpMinimal.",
    },
]

PLOT_OUTPUTS = [
    "ilp_issue_gap_calibration.png",
    "multiwarp_throughput_calibration.png",
    "ilp_issue_gap_calibration.csv",
    "multiwarp_throughput_calibration.csv",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_dir_for_config(config_name: str) -> Path:
    return RUN_BASE_DIR / config_name


def is_simulator_lib_path(path: str) -> bool:
    resolved = os.path.realpath(path)
    return resolved.startswith(str(PROJECT_ROOT / "lib"))


def native_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "GPGPUSIM_ROOT",
        "GPGPUSIM_CONFIG",
        "GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN",
        "OPENCL_REMOTE_GPU_HOST",
    ):
        env.pop(key, None)

    ld_library_path = env.get("LD_LIBRARY_PATH", "")
    if ld_library_path:
        filtered = [
            path
            for path in ld_library_path.split(":")
            if path and not is_simulator_lib_path(path)
        ]
        if filtered:
            env["LD_LIBRARY_PATH"] = ":".join(filtered)
        else:
            env.pop("LD_LIBRARY_PATH", None)

    return env


def shell_quote(path: Path) -> str:
    return shlex.quote(str(path))


def run_command(cmd: str, env: dict[str, str] | None = None) -> None:
    print(f"⚡ Running: {cmd}")
    subprocess.run(
        cmd,
        shell=True,
        check=True,
        cwd=TEST_DIR,
        env=env,
        executable="/bin/bash",
    )


def clear_run_outputs(run_dir: Path) -> None:
    ensure_dir(run_dir)
    for target in RUN_TARGETS:
        for path in run_dir.glob(f"{target['file_prefix']}*.txt"):
            path.unlink()


def clear_captured_outputs() -> None:
    ensure_dir(OUTPUT_DIR)

    capture_patterns = [
        "MMAIssueTest.ILPMinimal.*_HARDWARE.txt",
        "MMAIssueTest.ILPMinimal.*_SIM.txt",
        "MMAIssueTest.MultiWarpMinimal.*_HARDWARE.txt",
        "MMAIssueTest.MultiWarpMinimal.*_SIM.txt",
    ]
    for pattern in capture_patterns:
        for path in OUTPUT_DIR.glob(pattern):
            path.unlink()

    for filename in PLOT_OUTPUTS:
        path = OUTPUT_DIR / filename
        if path.exists():
            path.unlink()


def capture_outputs(run_dir: Path, suffix: str) -> list[Path]:
    captured: list[Path] = []
    for target in RUN_TARGETS:
        for src_path in sorted(run_dir.glob(f"{target['file_prefix']}*.txt")):
            dst_path = OUTPUT_DIR / f"{src_path.stem}_{suffix}{src_path.suffix}"
            shutil.copy2(src_path, dst_path)
            captured.append(dst_path)
            print(f"Captured: {dst_path}")
    return captured


def variant_from_file(file_path: Path) -> str:
    with file_path.open("r", encoding="utf-8") as infile:
        first_line = infile.readline().strip()

    prefix = "MMA Variant: "
    if first_line.startswith(prefix):
        return first_line[len(prefix) :].strip()

    stem = file_path.stem.removesuffix("_HARDWARE").removesuffix("_SIM")
    parts = stem.split(".")
    return parts[-1] if parts else stem


def parse_box_table_file(
    file_path: Path,
    x_index: int,
    y_index: int,
    skip_tokens: tuple[str, ...],
) -> dict[str, object]:
    data: dict[str, object] = {"variant": variant_from_file(file_path), "x": [], "y": []}

    with file_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            normalized = line.replace("│", "|")
            if "|" not in normalized:
                continue
            if any(token in normalized for token in skip_tokens):
                continue

            parts = [part.strip() for part in normalized.split("|") if part.strip()]
            if len(parts) <= max(x_index, y_index):
                continue

            try:
                data["x"].append(int(parts[x_index]))
                data["y"].append(float(parts[y_index]))
            except ValueError:
                continue

    return data


def parse_ilp_minimal(file_path: Path) -> dict[str, object]:
    return parse_box_table_file(
        file_path,
        x_index=0,
        y_index=4,
        skip_tokens=("ILP", "Cycles/MMA", "MMA Variant"),
    )


def parse_multiwarp_minimal(file_path: Path) -> dict[str, object]:
    return parse_box_table_file(
        file_path,
        x_index=0,
        y_index=3,
        skip_tokens=("Warps", "Cycles/MMA", "MMA Variant"),
    )


def collect_results(prefix: str, suffix: str, parser) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for path in sorted(OUTPUT_DIR.glob(f"{prefix}*_{suffix}.txt")):
        parsed = parser(path)
        if parsed["x"] and parsed["y"]:
            results[str(parsed["variant"])] = parsed
    return results


def plot_variant_comparison(
    hw_results: dict[str, dict[str, object]],
    sim_results: dict[str, dict[str, object]],
    title: str,
    xlabel: str,
    ylabel: str,
    output_filename: str,
) -> None:
    if not hw_results or not sim_results:
        print(f"Skipping {output_filename} due to missing data")
        return

    plt.figure(figsize=(12, 7))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    variants = sorted(set(hw_results.keys()) | set(sim_results.keys()))

    for index, variant in enumerate(variants):
        color = color_cycle[index % len(color_cycle)] if color_cycle else None

        if variant in hw_results:
            plt.plot(
                hw_results[variant]["x"],
                hw_results[variant]["y"],
                marker="o",
                linestyle="-",
                color=color,
                label=f"{variant} HW",
            )
        if variant in sim_results:
            plt.plot(
                sim_results[variant]["x"],
                sim_results[variant]["y"],
                marker="x",
                linestyle="--",
                color=color,
                label=f"{variant} Sim",
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path)
    plt.close()
    print(f"Generated plot: {output_path}")


def export_variant_csv(
    hw_results: dict[str, dict[str, object]],
    sim_results: dict[str, dict[str, object]],
    output_filename: str,
    x_label: str,
) -> None:
    if not hw_results or not sim_results:
        print(f"Skipping {output_filename} due to missing data")
        return

    output_path = OUTPUT_DIR / output_filename
    variants = sorted(set(hw_results.keys()) | set(sim_results.keys()))

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "variant",
                x_label,
                "HW_Cycles_per_MMA",
                "Sim_Cycles_per_MMA",
                "Diff_Percent",
            ],
        )
        writer.writeheader()

        for variant in variants:
            hw_map = {}
            sim_map = {}
            if variant in hw_results:
                hw_map = dict(zip(hw_results[variant]["x"], hw_results[variant]["y"]))
            if variant in sim_results:
                sim_map = dict(zip(sim_results[variant]["x"], sim_results[variant]["y"]))

            for x_value in sorted(set(hw_map.keys()) | set(sim_map.keys())):
                hw_value = hw_map.get(x_value)
                sim_value = sim_map.get(x_value)
                diff_percent = "N/A"
                if hw_value is not None and sim_value is not None and hw_value != 0:
                    diff_percent = f"{((sim_value - hw_value) / hw_value) * 100:.2f}%"

                writer.writerow(
                    {
                        "variant": variant,
                        x_label: x_value,
                        "HW_Cycles_per_MMA": hw_value if hw_value is not None else "N/A",
                        "Sim_Cycles_per_MMA": sim_value if sim_value is not None else "N/A",
                        "Diff_Percent": diff_percent,
                    }
                )

    print(f"Generated CSV: {output_path}")


def run_hardware_phase(config_name: str, run_dir: Path) -> None:
    print("\n[Phase 1] Running Hardware Tests (Native)...")
    clear_run_outputs(run_dir)

    for target in RUN_TARGETS:
        run_command(
            f"./run_tests.py -c {config_name} run --arch sm120 "
            f"--test-group microbench --profile mma \"{target['pattern']}\"",
            env=native_env(),
        )

    capture_outputs(run_dir, "HARDWARE")


def run_sim_phase(config_name: str, run_dir: Path) -> None:
    print("\n[Phase 2] Running GPGPU-Sim Tests...")
    clear_run_outputs(run_dir)

    setup_cmd = f"source {shell_quote(SETUP_ENVIRONMENT)}"
    for target in RUN_TARGETS:
        run_command(
            f"{setup_cmd} && ./run_tests.py -c {config_name} run --arch sm120 "
            f"--test-group microbench --profile mma \"{target['pattern']}\""
        )

    capture_outputs(run_dir, "SIM")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        default=DEFAULT_CONFIG,
        help=f"GPU configuration name under configs/ (default: {DEFAULT_CONFIG})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.environ.get("CUDA_INSTALL_PATH"):
        raise SystemExit(
            "Set CUDA_INSTALL_PATH to the CUDA Toolkit root before running calibration."
        )

    ensure_dir(OUTPUT_DIR)
    clear_captured_outputs()

    config_name = args.config
    run_dir = run_dir_for_config(config_name)

    run_command(f"./run_tests.py -c {config_name} refresh", env=native_env())
    run_hardware_phase(config_name, run_dir)
    run_sim_phase(config_name, run_dir)

    print("\n[Phase 3] Generating Plots...")

    hw_ilp = collect_results("MMAIssueTest.ILPMinimal.", "HARDWARE", parse_ilp_minimal)
    sim_ilp = collect_results("MMAIssueTest.ILPMinimal.", "SIM", parse_ilp_minimal)
    plot_variant_comparison(
        hw_ilp,
        sim_ilp,
        title="MMA Throughput vs ILP (Issue Gap Analysis)",
        xlabel="ILP (Instruction Level Parallelism)",
        ylabel="Cycles per MMA Instruction",
        output_filename="ilp_issue_gap_calibration.png",
    )
    export_variant_csv(
        hw_ilp,
        sim_ilp,
        output_filename="ilp_issue_gap_calibration.csv",
        x_label="ilp",
    )

    hw_multiwarp = collect_results(
        "MMAIssueTest.MultiWarpMinimal.", "HARDWARE", parse_multiwarp_minimal
    )
    sim_multiwarp = collect_results(
        "MMAIssueTest.MultiWarpMinimal.", "SIM", parse_multiwarp_minimal
    )
    plot_variant_comparison(
        hw_multiwarp,
        sim_multiwarp,
        title="MMA Throughput vs Warp Count (Pipeline Independence)",
        xlabel="Number of Warps",
        ylabel="Cycles per MMA Instruction",
        output_filename="multiwarp_throughput_calibration.png",
    )
    export_variant_csv(
        hw_multiwarp,
        sim_multiwarp,
        output_filename="multiwarp_throughput_calibration.csv",
        x_label="warps",
    )

    print(f"\nCalibration finished. Results in {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
