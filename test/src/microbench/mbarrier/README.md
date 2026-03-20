# MBarrier Microbenchmark Suite

This directory contains `mbarrier`-focused microbenchmarks for characterizing polling latency, visibility, and multi-warp contention behavior on native GPUs.

## Supported Tests

The suite is built on GoogleTest. You can run specific tests using the `./run_tests.sh` script from the `test/` directory.

### 1. Try-Wait Latency (`mbarrier_trywait_latency_bench.cc`)
Measures dependent-chain latency for `mbarrier.try_wait.parity` and the matched `ld.shared` baseline.

*   **Test Name**: `MBarrierLatencyTest.P1`
    *   **Description**: Measures false-path `try_wait.parity` latency with an instruction-count sweep.
    *   **Run Command**:
        ```bash
        ./run_tests.sh bench "MBarrierLatencyTest.P1"
        ```
    *   **Output**:
        *   Console summary table.
        *   `MBarrierLatencyTest.P1.csv`

*   **Test Name**: `MBarrierLatencyTest.P2`
    *   **Description**: Measures true-path `try_wait.parity` latency with the same sweep.
    *   **Run Command**:
        ```bash
        ./run_tests.sh bench "MBarrierLatencyTest.P2"
        ```
    *   **Output**:
        *   Console summary table.
        *   `MBarrierLatencyTest.P2.csv`

*   **Test Name**: `MBarrierLatencyTest.PLd`
    *   **Description**: Measures dependent `ld.shared` latency as the matched baseline for `P1/P2`.
    *   **Run Command**:
        ```bash
        ./run_tests.sh bench "MBarrierLatencyTest.PLd"
        ```
    *   **Output**:
        *   Console summary table.
        *   `MBarrierLatencyTest.PLd.csv`

### 2. Visibility (`mbarrier_visibility_bench.cc`)
Measures how quickly `arrive_expect_tx` and TMA completion become observable to polling warps.

*   **Test Name**: `MBarrierVisibilityTest.P3Arrive`
    *   **Description**: Calibrates filler-step cost, then sweeps the observer delay window to locate the `arrive_expect_tx(0)` visibility threshold.
    *   **Run Command**:
        ```bash
        ./run_tests.sh bench "MBarrierVisibilityTest.P3Arrive"
        ```
    *   **Output**:
        *   Console summary table.
        *   `MBarrierVisibilityTest.P3Arrive.FillerCalibration.csv`
        *   `MBarrierVisibilityTest.P3Arrive.Threshold.csv`

*   **Test Name**: `MBarrierVisibilityTest.P3Tma`
    *   **Description**: Sweeps TMA tile size and compares data visibility, barrier visibility, and matched control latency.
    *   **Run Command**:
        ```bash
        ./run_tests.sh bench "MBarrierVisibilityTest.P3Tma"
        ```
    *   **Output**:
        *   Console summary table.
        *   `MBarrierVisibilityTest.P3Tma.csv`

### 3. Contention (`mbarrier_contention_bench.cc`)
Measures multi-warp polling overhead and layout sensitivity.

*   **Test Name**: `MBarrierContentionTest.LayoutScan`
    *   **Description**: Scans candidate barrier strides and selects stable high-conflict and low-conflict layouts.
    *   **Run Command**:
        ```bash
        ./run_tests.sh bench "MBarrierContentionTest.LayoutScan"
        ```
    *   **Output**:
        *   Console summary table.
        *   `MBarrierContentionTest.LayoutScan.csv`

*   **Test Name**: `MBarrierContentionTest.PContention`
    *   **Description**: Measures the extra cost of false-path polling under increasing warp count for both selected layouts.
    *   **Run Command**:
        ```bash
        ./run_tests.sh bench "MBarrierContentionTest.PContention"
        ```
    *   **Output**:
        *   Console summary table.
        *   `MBarrierContentionTest.PContention.csv`

## Notes

*   These microbenchmarks are intended for native GPU mode. In simulator mode they may skip or lose the timing meaning the CSV files rely on.
*   All CSV files are written into the active `test/run/<GPU_CONFIG>/` directory because `run_tests.sh` executes each bench from inside that config directory.
*   The shared helpers used by both `integration` and `microbench` now live under `test/common/mbarrier/`.
