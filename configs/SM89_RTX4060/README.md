# SM89_RTX4060 Configuration

PTX execution-driven configuration for the measured GeForce RTX 4060 (`sm_89`, AD106). It combines reported device limits with modeled cache, memory, and timing parameters; it does not claim support for every Ada product or arbitrary kernels. See [gpgpusim.config](gpgpusim.config) for the complete parameter set and [sass_primary_hints.rules](sass_primary_hints.rules) for the tested PTX-to-SASS scheduling hints.

## Modeled device

| Property | Configuration |
| --- | --- |
| Compute capability | 8.9 |
| SMs | 24 |
| Resident threads / warps per SM | 1536 / 48 |
| Shared memory per SM | 100 KiB |
| Default / opt-in shared memory per block | 48 / 99 KiB |
| L2 topology | 8 memory channels, 2 slices per channel |
| Modeled L2 capacity | 24 MiB |
| Memory interface | 128-bit GDDR6, 272 GB/s nominal bandwidth |
| Core / interconnect / L2 / DRAM clocks | 1809 / 1200 / 1200 / 4250.5 MHz |

The measured device reports PCI ID `0x2808`. The 1809 MHz core setting follows the NCU base-clock protocol; it does not mean the physical application clock was locked. L2 topology, service capacity, and several timing values are model assumptions.

## Model scope

FP16 MMA and ordinary `cp.async` are enabled. TMA and TensorMap are disabled. Device resource limits are fixed where reported; undocumented timing and memory behavior remain approximations. The model currently targets the FP16 GEMM and FlashAttention forward workloads listed below.

## Validation summary

| Workload | Cases | Output validation | Cycle MAPE |
|---|---:|---:|---:|
| Paper GEMM shapes, FP16, fixed 64×64×32 tile, current config | 40 | 40/40 pass | 14.817% |
| Triton pointer-load FlashAttention forward | 8 | 8/8 pass | 10.158% |
| Official FA1 + FA2 forward, B=8 and S=128/256 | 16 | 16/16 pass | 16.382% combined (FA1 22.904%, FA2 9.859%) |

MAPE compares NCU `sm__cycles_elapsed.avg` with simulator `gpu_tot_sim_cycle`; GEMM uses the median of five NCU samples. Correctness and cycle agreement are separate checks. GEMM and FA sets use different implementations and are not aggregated together. Results are workload-specific and remain above the paper's 5.24% aggregate.

## Calibration workload definitions

The three CSV manifests in [workloads/](workloads/) define the exact cases used for the reported MAPE values:

- [gemm_paper_fp16_cases.csv](workloads/gemm_paper_fp16_cases.csv): 40 paper GEMM shapes. FP16 inputs/output, FP32 accumulation; contiguous row-major A/B/C, no transpose; alpha=1, beta=0, no activation; `BLOCK_M/N/K=64/64/32`, `GROUP_SIZE_M=8`, 4 warps, 3 stages, 1 CTA/program; seed 20261024; no TMA. NCU uses five runs per case with `--clock-control base --cache-control all`; the median cycle count is compared with simulator `gpu_tot_sim_cycle`. MAPE: 14.817%.
- [fa_triton_pointer_fp16_cases.csv](workloads/fa_triton_pointer_fp16_cases.csv): 8 custom pointer-load online-softmax Triton forward cases. FP16 BHSD, scale `1/sqrt(D)`, `BLOCK_M/N=64/64`, 4 warps, 3 stages, seed 20261117, no TMA. NCU uses one launch per case with base clock control, all-cache control, and stable pipeline boost. MAPE: 10.158%.
- [fa_official_fa1_fa2_fp16_cases.csv](workloads/fa_official_fa1_fa2_fp16_cases.csv): 8 workloads each run in FA1 and FA2, for 16 implementation-case rows. B=8, S=128/256, (H,D)=(32,64)/(16,128), full/causal; FP16 input/output, FP32 accumulation, no dropout, scale `1/sqrt(D)`, contiguous BSHD, identical deterministic Q/K/V for each paired workload. NCU uses one launch per implementation-case with base clock control, all-cache control, and stable pipeline boost. MAPE: FA1 22.904%, FA2 9.859%, combined 16.382%.

For all three sets, per-case absolute percentage error is `abs(sim_cycles / hardware_cycles - 1) * 100`; MAPE is the arithmetic mean over the listed cases. Numerical output validation is a separate pass/fail check. No calibration/holdout split is used for these reported full-set MAPE values. The official FA source revisions were FA1 v1.0.9 (`6d48e14a6c2f551db96f0badc658a6279a929df3`), its CUTLASS dependency (`319a389f42b776fae5701afcb943fc03be5b5c25`), and FA2 (`d80a77103021c4e980f8cbbf85774f6a19e6474a`).

These manifests specify workloads and settings; they are not one-command runners. Generated PTX/CUBIN files, profiler reports, captured tensors, and simulator logs are not included.

The GEMM experiment used TritonTrace capture and its generated CUDA harness. The Triton FlashAttention experiment used a pointer-load Triton kernel. The official FA1/FA2 experiment used the upstream CUDA implementations and a separate harness. The FA2 test entry point remains [fa2_fwd_fp16_test.cu](../../tests/src/fa2/fa2_fwd_fp16_test.cu); its built-in test profiles are not identical to the custom full-set manifest.
