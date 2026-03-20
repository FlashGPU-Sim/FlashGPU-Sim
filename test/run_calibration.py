#!/usr/bin/env python3
"""
GPGPU-Sim calibration runner for the MMA microbenchmarks.

This script runs the committed MMA calibration benches on native hardware and
under GPGPU-Sim, captures the per-variant output files, and generates
comparison plots/CSVs in test/calibration_results/.
"""

import csv
import glob
import os
import shutil
import subprocess

import matplotlib.pyplot as plt


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
OUTPUT_DIR = os.path.join(TEST_DIR, "calibration_results")
RUN_BASE_DIR = os.path.join(TEST_DIR, "run")
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


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def run_dir_for_config(config_name):
    return os.path.join(RUN_BASE_DIR, config_name)


def native_env():
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
            if "gpgpu-sim_distribution/lib" not in path
        ]
        if filtered:
            env["LD_LIBRARY_PATH"] = ":".join(filtered)
        else:
            env.pop("LD_LIBRARY_PATH", None)

    return env


def run_command(cmd, env=None):
    print(f"⚡ Running: {cmd}")
    subprocess.run(
        cmd,
        shell=True,
        check=True,
        cwd=TEST_DIR,
        env=env,
        executable="/bin/bash",
    )


def clear_run_outputs(run_dir):
    ensure_dir(run_dir)
    for target in RUN_TARGETS:
        pattern = os.path.join(run_dir, f"{target['file_prefix']}*.txt")
        for path in glob.glob(pattern):
            os.remove(path)


def clear_captured_outputs():
    ensure_dir(OUTPUT_DIR)

    capture_patterns = [
        "MMAIssueTest.ILPMinimal.*_HARDWARE.txt",
        "MMAIssueTest.ILPMinimal.*_SIM.txt",
        "MMAIssueTest.MultiWarpMinimal.*_HARDWARE.txt",
        "MMAIssueTest.MultiWarpMinimal.*_SIM.txt",
    ]
    for pattern in capture_patterns:
        for path in glob.glob(os.path.join(OUTPUT_DIR, pattern)):
            os.remove(path)

    for filename in PLOT_OUTPUTS:
        path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(path):
            os.remove(path)


def capture_outputs(run_dir, suffix):
    captured = []
    for target in RUN_TARGETS:
        pattern = os.path.join(run_dir, f"{target['file_prefix']}*.txt")
        for src_path in sorted(glob.glob(pattern)):
            base, ext = os.path.splitext(os.path.basename(src_path))
            dst_path = os.path.join(OUTPUT_DIR, f"{base}_{suffix}{ext}")
            shutil.copy2(src_path, dst_path)
            captured.append(dst_path)
            print(f"Captured: {dst_path}")
    return captured


def variant_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as infile:
        first_line = infile.readline().strip()

    prefix = "MMA Variant: "
    if first_line.startswith(prefix):
        return first_line[len(prefix) :].strip()

    stem = os.path.splitext(os.path.basename(file_path))[0]
    stem = stem.removesuffix("_HARDWARE")
    stem = stem.removesuffix("_SIM")
    parts = stem.split(".")
    return parts[-1] if parts else stem


def parse_box_table_file(file_path, x_index, y_index, skip_tokens):
    data = {"variant": variant_from_file(file_path), "x": [], "y": []}

    with open(file_path, "r", encoding="utf-8") as infile:
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


def parse_ilp_minimal(file_path):
    return parse_box_table_file(
        file_path,
        x_index=0,
        y_index=4,
        skip_tokens=("ILP", "Cycles/MMA", "MMA Variant"),
    )


def parse_multiwarp_minimal(file_path):
    return parse_box_table_file(
        file_path,
        x_index=0,
        y_index=3,
        skip_tokens=("Warps", "Cycles/MMA", "MMA Variant"),
    )


def collect_results(prefix, suffix, parser):
    results = {}
    pattern = os.path.join(OUTPUT_DIR, f"{prefix}*_{suffix}.txt")
    for path in sorted(glob.glob(pattern)):
        parsed = parser(path)
        if parsed["x"] and parsed["y"]:
            results[parsed["variant"]] = parsed
    return results


def plot_variant_comparison(
    hw_results,
    sim_results,
    title,
    xlabel,
    ylabel,
    output_filename,
):
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

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Generated plot: {output_path}")


def export_variant_csv(hw_results, sim_results, output_filename, x_label):
    if not hw_results or not sim_results:
        print(f"Skipping {output_filename} due to missing data")
        return

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    variants = sorted(set(hw_results.keys()) | set(sim_results.keys()))

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
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


def run_hardware_phase(config_name, run_dir):
    print("\n[Phase 1] Running Hardware Tests (Native)...")
    clear_run_outputs(run_dir)

    for target in RUN_TARGETS:
        run_command(
            f"./run_tests.sh -c {config_name} bench \"{target['pattern']}\"",
            env=native_env(),
        )

    capture_outputs(run_dir, "HARDWARE")


def run_sim_phase(config_name, run_dir):
    print("\n[Phase 2] Running GPGPU-Sim Tests...")
    clear_run_outputs(run_dir)

    setup_script = os.path.join(PROJECT_ROOT, "setup_environment")
    for target in RUN_TARGETS:
        run_command(
            f"source {setup_script} && ./run_tests.sh -c {config_name} bench "
            f"\"{target['pattern']}\""
        )

    capture_outputs(run_dir, "SIM")


def main():
    ensure_dir(OUTPUT_DIR)
    clear_captured_outputs()

    config_name = DEFAULT_CONFIG
    run_dir = run_dir_for_config(config_name)

    run_command(f"./run_tests.sh -c {config_name} refresh", env=native_env())
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

    print("\nCalibration finished. Results in test/calibration_results/")


if __name__ == "__main__":
    main()
