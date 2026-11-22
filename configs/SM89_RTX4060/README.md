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

## Reproduction

The fixed GEMM capture, NCU, simulation workflow, and full-set case manifest are under [tests/triton_trace/calibration](../../tests/triton_trace/calibration/README.md). The integrated FA2 correctness case is [fa2_fwd_fp16_case.cuh](../../tests/src/fa2/fa2_fwd_fp16_case.cuh). The official FA1/FA2 cycle set used pinned external source; its detailed per-case artifacts and profiler reports remain in the local experiment backup.
