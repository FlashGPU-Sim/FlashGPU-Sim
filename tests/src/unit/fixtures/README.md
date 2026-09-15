# SASS unit-test cubins

This directory keeps two small controlled SM120 CUDA sources, cubins, and
physical-GPU runners. It does not keep generated SASSIR files.

- `sass_gemm_sm120.cubin` executes a scalar `2x2x4` F32 GEMM.
- `sass_mma_gemm_sm120.cubin` executes one `HMMA.1688.F32` and emits a full
  `16x8` F32 tile.

The SM120 unit build decodes these cubins plus the repository's Triton GEMM
and GQA-attention cubins with CUDA's official `nvdisasm`/`cuobjdump`. Generated
SASSIR is written under the ignored `tests/build/generated/sass-unit/`
directory, then the unit tests load and execute it by dynamic PC. The generated
Triton inputs come from:

- `tests/ci/perf/traces/SM120_RTX5090/gemm-m4096-n128-k4096/`
- `tests/ci/perf/traces/SM120_RTX5090/llama3-prefill-b2-s128/`

Regenerate the small cubins with CUDA 13.3 when their sources change:

```bash
nvcc -std=c++17 -O3 --fmad=true -arch=sm_120a -cubin \
  sass_gemm_sm120.cu -o sass_gemm_sm120.cubin
nvcc -std=c++17 -O3 --fmad=true -arch=sm_120a -cubin \
  sass_mma_gemm_sm120.cu -o sass_mma_gemm_sm120.cubin
```

The adjacent driver-API runners can execute the exact cubins on a physical
GPU. Functional SASS checks compare 4/4 scalar outputs and 128/128 tensor-core
outputs with CPU GEMM references; the generated Triton gates retain their
complete TMA/WGMMA GEMM and causal-attention numerical checks.
