# Experimental RTX 4060 FP16 GEMM calibration

This workflow runs a fixed Triton GEMM on the physical GPU, records inputs, outputs and kernel code with TritonTrace, profiles the standalone launcher five times with Nsight Compute, and replays it under FlashGPU-Sim. The experimental configuration and its limitations are documented in [configs/SM89_RTX4060](../../../configs/SM89_RTX4060/README.md). This PR configuration and case manifest cover FP16 GEMM only.

Requirements: a built simulator, CUDA 12.8 tooling including `ptxas` and `nvdisasm`, PyTorch and Triton, an SM 8.9 GPU, `ncu` with permission to access counters, and the dependencies listed in the project and TritonTrace READMEs. Install TritonTrace in the active Python environment with `python -m pip install -e tools`, from the repository root.

Start from a fresh shell that has **not** sourced `setup_environment`; capture and NCU must use the native CUDA stack. From the repository root:

```bash
bash tests/triton_trace/calibration/run_gemm_calibration.sh capture square_256 fp16
bash tests/triton_trace/calibration/run_gemm_calibration.sh ncu square_256 fp16
bash tests/triton_trace/calibration/run_gemm_calibration.sh sim square_256 fp16
python tests/triton_trace/calibration/compare_gemm_results.py
```

`capture` checks the Triton result against a PyTorch reference, saves the exact inputs and GPU output, builds a launch-specific CUDA harness, and records artifact hashes. It refuses to replace a completed capture. `ncu` executes that same harness in five separate profiler runs, using base clock control and cache clearing; it saves each `.ncu-rep`, CSV and log. `sim` copies the experimental config **and the dtype-specific SASS hint rules** into the launcher directory, activates the simulator only in that subprocess, and requires both the simulated output check and cycle count. Existing reports and simulator logs are not overwritten. For a fresh repeat, use another case ID or archive the existing case directory first.

Outputs live under `reference/SM89_RTX4060/gemm/{calibration,holdout}/fp16/<case>/`. The summary script writes `gemm_cycle_comparison.csv`, including the NCU median, range, frequency, simulator cycles and signed error. It calculates MAPE separately for calibration and holdout cases that have all five reports and a passing simulator log. Incomplete cases are printed explicitly and excluded; inspect these messages before interpreting an aggregate. Do not tune on a holdout and continue to call it independent. Define new case IDs and freeze the model before using additional cases as a holdout.

`--clock-control base` is an NCU request, not proof that unprofiled execution uses the same fixed frequency. Profiling may replay a kernel when metrics require multiple passes; verify the collection log. Simulator total cycles and NCU SM elapsed cycles may not have identical start/end boundaries. SASS-guided scheduling depends on the captured code, tooling and rules and is only validated here for the listed FP16 GEMM path.
