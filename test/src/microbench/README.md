# MMA Microbenchmark Suite

This directory contains microbenchmarks for characterizing GPU instruction latencies and throughputs, specifically focusing on Tensor Core (MMA) instructions and general scalar instructions.

## Supported Tests

The suite is built on GoogleTest. You can run specific tests using the `./run_tests.sh` script from the `test/` directory.

### 1. MMA Issue Gap & Throughput Test (`mma_issue_bench.cc`)
Analyzes the pipeline throughput (issue gap) of Tensor Core instructions by varying Instruction-Level Parallelism (ILP).

*   **Test Name**: `MMAIssueTest.ILPMinimal`
    *   **Description**: Runs single-warp MMA chains with varying ILP (1, 2, 4, 8) to determine the maximum instruction issue rate.
    *   **Run Command**:
        ```bash
        ./run_tests.sh run "MMAIssueTest.ILPMinimal"
        ```

*   **Test Name**: `MMAIssueTest.MultiWarpMinimal`
    *   **Description**: Runs multi-warp workloads to verify if throughput scales with the number of warps (checking for independent Tensor Core pipelines).
    *   **Run Command**:
        ```bash
        ./run_tests.sh run "MMAIssueTest.MultiWarpMinimal"
        ```

### 2. Instruction Latency Calibration (`inst_latency_bench.cc`)
Measures the pure execution latency (dependent chain) of various scalar and floating-point instructions.

*   **Test Name**: `InstLatencyTest.IntegerAdd` (and others)
    *   **Description**: Measures latency for Integer ADD, MUL, MAD, etc.
    *   **Run Command**:
        ```bash
        ./run_tests.sh run "InstLatencyTest.*"
        ```

## Configuration Guide

### How to Switch MMA Instruction Type

To test different MMA instruction types (e.g., FP16, TF32, INT8) in the Issue Gap benchmark, you need to modify the source code in [`mma_issue_bench.cc`](mma_issue_bench.cc).

1.  Open `mma_issue_bench.cc`.
2.  Locate the section marked `// SELECT ACTIVE MMA TYPE HERE` (around line 144-151).
3.  Uncomment the `using CurrentMmaOp = ...` line corresponding to the instruction you want to test and comment out others.

**Example:**

To switch from the default **TF32** (M16N8K8) back to **FP16** (M16N8K16):

```cpp
// SELECT ACTIVE MMA TYPE HERE
using CurrentMmaOp = MmaOp_F16_M16N8K16; // <--- Uncomment this for FP16
// using CurrentMmaOp = MmaOp_F16_M16N8K8; 
// using CurrentMmaOp = MmaOp_BF16_M16N8K8; 
// using CurrentMmaOp = MmaOp_TF32_M16N8K8; // <--- Comment this out
```

4.  Recompile the tests before running.

### Supported Strategies

*   `MmaOp_F16_M16N8K16`: FP16 input, FP32 accum. (Standard Tensor Core op)
*   `MmaOp_TF32_M16N8K8`: TF32 input, FP32 accum. (Ampere+ default)
*   `MmaOp_S8_M16N8K32`: Int8 input, Int32 accum. (Integer Tensor Core)
*   `MmaOp_F16_M16N8K8`: FP16 input, FP32 accum. (Smaller K dimension)
*   `MmaOp_BF16_M16N8K8`: BF16 input, FP32 accum. (Ampere+ only)
*   `MmaOp_TF32_M16N8K4`: TF32 input, FP32 accum. (Hopper+ specific shape)
*   `MmaOp_S8_M16N8K16`: Int8 input, Int32 accum. (Half-K dimension)

### How to change ILP and multi-warp level

Change vectors in [`mma_issue_bench.cc`](mma_issue_bench.cc):

#### For ILPMinimal

```c++
//line 409-413
    const int mma_count = 16;  // iterations per kernel
    const int warmup = 3;
    const int iterations = 10;
    
    std::vector<int> ilp_values = {1, 2, 4, 8};
```


#### For MultiWarpMinimal

```c++
// line 474-476
    const int ilp = 1;
    const int mma_count = 1000;  // Increased from 16 to 1000 to minimize overhead impact
    std::vector<int> warp_counts = {1, 2, 4, 8, 16, 32};
```

## Running Calibration

To calibrate the simulator against hardware, use the provided Python script:

```bash
# From test/ directory
./run_calibration.py
```

This script will:
1.  Run `MMAIssueTest.ILPMinimal` and `MMAIssueTest.MultiWarpMinimal` on **native hardware** (ignoring `setup_environment`).
2.  Source `setup_environment` and run the same tests on **GPGPU-Sim**.
3.  Parse the output tables and generating comparison plots in `test/calibration_results/`.

**Output:**
- `calibration_results/ilp_scaling_calibration.png`: Issue Gap comparison.
- `calibration_results/multiwarp_scaling_calibration.png`: Pipeline throughput comparison.
