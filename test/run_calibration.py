#!/usr/bin/env python3
"""
GPGPU-Sim Calibration Runner & Plotter

Runs microbenchmarks on both Native Hardware and GPGPU-Sim,
collects performance data, and generates comparison plots.
"""

import os
import shutil
import subprocess
import re
import matplotlib.pyplot as plt
import sys
import glob

# Configuration
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
OUTPUT_DIR = os.path.join(TEST_DIR, "calibration_results")
RUN_BASE_DIR = os.path.join(TEST_DIR, "run")
DEFAULT_CONFIG = "SM120_RTX5090"

TEST_CASES = [
    "MMAILPIssueGapTest.ILPMinimal",
    "MMAILPIssueGapTest.MultiWarpMinimal"
]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_run_dir():
    """Determine where run_tests.sh is outputting files"""
    # First check default
    default_run = os.path.join(RUN_BASE_DIR, DEFAULT_CONFIG)
    if os.path.exists(default_run):
        return default_run
    
    # If not found, try to find any directory in run/
    if os.path.exists(RUN_BASE_DIR):
        subdirs = [os.path.join(RUN_BASE_DIR, d) for d in os.listdir(RUN_BASE_DIR) if os.path.isdir(os.path.join(RUN_BASE_DIR, d))]
        if subdirs:
            # Return the one with the latest modification time
            return max(subdirs, key=os.path.getmtime)
    
    return TEST_DIR 

def run_command(cmd, env=None, shell=True):
    """Run a shell command and stream output"""
    print(f"⚡ Running: {cmd}")
    process = subprocess.Popen(
        cmd, 
        shell=shell, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        universal_newlines=True,
        env=env,
        executable="/bin/bash" # Use bash for source command
    )
    
    # Stream output
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        print(f"Error: Command failed with return code {process.returncode}")
        # Don't exit immediately, let the script try to recover or show what happened
        # sys.exit(1) 

def parse_ilp_minimal(file_path):
    """
    Parses MMAILPIssueGapTest.ILPMinimal.txt
    Format:
    |   ILP   | Iterations | Total MMAs   |  Total Cycles  |  Cycles/MMA     |
    |    1    |      16    |        16    |         105     |        6.56     |
    """
    data = {'ilp': [], 'cycles_per_mma': []}
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found")
        return data
        
    with open(file_path, 'r') as f:
        for line in f:
            line = line.replace('│', '|') # Handle box-drawing chars
            if "|" in line and "ILP" not in line and "Cycles" not in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 5:
                    try:
                        data['ilp'].append(int(parts[0]))
                        data['cycles_per_mma'].append(float(parts[4]))
                    except ValueError:
                        continue
    return data

def parse_multiwarp_minimal(file_path):
    """
    Parses MMAILPIssueGapTest.MultiWarpMinimal.txt
    Format:
    |  Warps  | Total MMAs |   Cycles   | Cycles/MMA |  Speedup  |
    |     1   |     1000   |      6500  |      6.50  |     0.00x |
    """
    data = {'warps': [], 'cycles_per_mma': []}
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found")
        return data

    with open(file_path, 'r') as f:
        for line in f:
            line = line.replace('│', '|') # Handle box-drawing chars
            if "|" in line and "Warps" not in line and "Cycles" not in line:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 5:
                    try:
                        data['warps'].append(int(parts[0]))
                        data['cycles_per_mma'].append(float(parts[3]))
                    except ValueError:
                        continue
    return data

def plot_ilp_minimal(hw_data, sim_data):
    if not hw_data['ilp'] or not sim_data['ilp']:
        print("Skipping ILP plot due to missing data")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(hw_data['ilp'], hw_data['cycles_per_mma'], 'bo-', label='Hardware (Native)')
    plt.plot(sim_data['ilp'], sim_data['cycles_per_mma'], 'rs--', label='GPGPU-Sim')
    
    plt.title('MMA Throughput vs ILP (Issue Gap Analysis)')
    plt.xlabel('ILP (Instruction Level Parallelism)')
    plt.ylabel('Cycles per MMA Instruction')
    plt.legend()
    plt.grid(True)
    
    output_file = os.path.join(OUTPUT_DIR, "ilp_issue_gap_calibration.png")
    plt.savefig(output_file)
    print(f"Generated plot: {output_file}")
    plt.close()

def plot_multiwarp_minimal(hw_data, sim_data):
    if not hw_data['warps'] or not sim_data['warps']:
        print("Skipping MultiWarp plot due to missing data")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(hw_data['warps'], hw_data['cycles_per_mma'], 'bo-', label='Hardware (Native)')
    plt.plot(sim_data['warps'], sim_data['cycles_per_mma'], 'rs--', label='GPGPU-Sim')
    
    plt.title('MMA Throughput vs Warp Count (Pipeline Independence)')
    plt.xlabel('Number of Warps')
    plt.ylabel('Cycles per MMA Instruction')
    plt.legend()
    plt.grid(True)
    
    output_file = os.path.join(OUTPUT_DIR, "multiwarp_throughput_calibration.png")
    plt.savefig(output_file)
    print(f"Generated plot: {output_file}")
    plt.close()

def main():
    ensure_dir(OUTPUT_DIR)
    os.chdir(TEST_DIR) # Make sure we are in test/ dir
    
    # Identify Run Directory
    run_dir = get_run_dir()
    print(f"Detected Run Directory: {run_dir}")

    # 1. Run Hardware Tests
    print("\n[Phase 1] Running Hardware Tests (Native)...")
    # Clean old results in Run Dir
    for test in TEST_CASES:
        res_file = os.path.join(run_dir, f"{test}.txt")
        if os.path.exists(res_file):
            os.remove(res_file)

    # Run tests without sourcing setup_environment (uses system CUDA)
    # We clear LD_LIBRARY_PATH just in case to avoid simulator libs
    hw_env = os.environ.copy()
    hw_env.pop("GPGPUSIM_ROOT", None)
    
    cmd_hw = "./run_tests.sh run \"MMAILPIssueGapTest.*\""
    run_command(cmd_hw, env=hw_env)

    # Rename results
    for test in TEST_CASES:
        src_file = os.path.join(run_dir, f"{test}.txt")
        dst_file = os.path.join(OUTPUT_DIR, f"{test}_HARDWARE.txt")
        if os.path.exists(src_file):
            shutil.move(src_file, dst_file)
            print(f"Captured: {dst_file}")
        else:
            print(f"Error: Hardware test {test} did not produce partial output at {src_file}.")

    # 2. Run Simulator Tests
    print("\n[Phase 2] Running GPGPU-Sim Tests...")
    
    # Setup for Sim
    setup_script = os.path.join(PROJECT_ROOT, "setup_environment")
    cmd_sim = f"source {setup_script} && ./run_tests.sh run \"MMAILPIssueGapTest.*\""
    
    # Clean old results again (just in case)
    for test in TEST_CASES:
        res_file = os.path.join(run_dir, f"{test}.txt")
        if os.path.exists(res_file):
            os.remove(res_file)

    run_command(cmd_sim)

    # Rename results
    for test in TEST_CASES:
        src_file = os.path.join(run_dir, f"{test}.txt")
        dst_file = os.path.join(OUTPUT_DIR, f"{test}_SIM.txt")
        if os.path.exists(src_file):
            shutil.move(src_file, dst_file)
            print(f"Captured: {dst_file}")
        else:
             print(f"Error: Simulator test {test} did not produce partial output at {src_file}.")

    # 3. Parse and Plot
    print("\n[Phase 3] Generating Plots...")
    
    hw_ilp = parse_ilp_minimal(os.path.join(OUTPUT_DIR, "MMAILPIssueGapTest.ILPMinimal_HARDWARE.txt"))
    sim_ilp = parse_ilp_minimal(os.path.join(OUTPUT_DIR, "MMAILPIssueGapTest.ILPMinimal_SIM.txt"))
    plot_ilp_minimal(hw_ilp, sim_ilp)

    hw_warp = parse_multiwarp_minimal(os.path.join(OUTPUT_DIR, "MMAILPIssueGapTest.MultiWarpMinimal_HARDWARE.txt"))
    sim_warp = parse_multiwarp_minimal(os.path.join(OUTPUT_DIR, "MMAILPIssueGapTest.MultiWarpMinimal_SIM.txt"))
    plot_multiwarp_minimal(hw_warp, sim_warp)

    print("\nCalibration finished. Results in test/calibration_results/")

if __name__ == "__main__":
    main()
