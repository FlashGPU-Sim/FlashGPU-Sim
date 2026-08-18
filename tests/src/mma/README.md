# MMA Integration Tests

These CUDA/GoogleTest workloads execute inline PTX `mma.sync` instructions
through FlashGPU-Sim and compare the results with CPU reference calculations.
The SM120 architecture manifest selects this test group.

## Coverage

| Source | Input type | Instruction shapes |
| --- | --- | --- |
| `mma_f16_test.cu` | F16 | `m16n8k8`, `m8n8k4`, `m16n8k16` |
| `mma_bf16_test.cu` | BF16 | `m16n8k8`, `m16n8k16` |
| `mma_tf32_test.cu` | TF32 | `m16n8k4`, `m16n8k8` |
| `mma_s8_test.cu` | S8 | `m16n8k16`, `m16n8k32`, `m8n8k16` |

The floating-point tests round inputs to the instruction format before
computing their references and compare with a format-appropriate tolerance.
The integer tests use exact S32 accumulation checks.

## Run

From the repository root:

```bash
./tests/run_tests.py list-cases --arch sm120 --group mma
./tests/run_tests.py run --arch sm120 --group mma
./tests/run_tests.py run --arch sm120 --group mma \
  --gtest-filter 'MMAF16M16N8K8IntegrationTest.AllOnesTest'
```

The `sm120` manifest supplies the NVCC target and simulator configuration; no
per-test configuration is required.

## Adding coverage

Add the CUDA source below this directory, register it in
[`tests/arch/sm120.toml`](../../arch/sm120.toml), and implement a CPU reference
that applies the input format's rounding and accumulation rules. Keep the
fragment-to-lane mapping consistent with the PTX ISA definition.

See the [test framework documentation](../../README.md), the
[MMA implementation](../../../src/gpgpu-sim/flash/mma/), and NVIDIA's
[PTX MMA instruction reference](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma).
