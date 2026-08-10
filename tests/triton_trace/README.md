# TritonTrace Examples and Validation

This directory contains examples and validation workloads for
[TritonTrace](../../tools/README.md). See the tool documentation for
installation, capture internals, generated artifacts, standalone replay, and
limitations.

## Layout

- `examples/`: single-workload examples demonstrating TritonTrace.
- `validation/`: systematic workload sweeps and comparison utilities.
- `triton_kernel_tracking/`: generated artifacts and validation results.

## Environment

Create a local Python environment and install TritonTrace:

```bash
cd tests/triton_trace
python3.12 -m venv .venv
.venv/bin/python -m pip install -U pip uv
.venv/bin/uv pip install --python .venv/bin/python torch triton numpy
.venv/bin/python -m pip install -e ../../tools
source .venv/bin/activate
```

Select a CUDA Toolkit compatible with the PTX version and target emitted by
the installed Triton release.

## Examples

The bundled examples are:

- `examples/example_vector_add.py`
- `examples/example_tensor_add.py`
- `examples/example_tma_gemm.py`
- `examples/example_gemm.py`
- `examples/example_flash_attn.py`
- `examples/example_gpt2_triton.py`

Run the vector-add example and build its generated harness:

```bash
python examples/example_vector_add.py
cd triton_kernel_tracking/example_vector_add/launchers
make -f add_kernel_launch1_Makefile
./add_kernel_launch1
```

Expected validation output:

```text
Validating outputs...
Validation PASSED for arg[2]: all 1024 elements match within tolerance 1.00e-05
```

All examples write generated artifacts under the root
`triton_kernel_tracking/` directory, not under `examples/`.

## Validation Sweeps

The `validation/` directory contains:

- `test_tma_gemm.py`: TMA GEMM trace generator for shape sweeps.
- `test_flash_attn.py`: FlashAttention trace generator for shape sweeps.
- `sweep_tests.sh`: trace, simulate, and profile shape sweeps.
- `compare_cycles.py`: compare NCU and FlashGPU-Sim cycle summaries.
- `extract_metrics.py`: extract simulator and NCU memory/cache metrics.
- `plot_cta_lifecycle.py`: plot CTA lifecycle timelines from simulation logs.
- `configs/`: workload-shape CSV files.

Typical commands:

```bash
# Run a TMA GEMM inference sweep with FlashGPU-Sim
./validation/sweep_tests.sh tma_gemm run \
  --csv configs/gemm_shapes_inference_server.csv

# Compare current simulation cycles against NCU summaries
python3 validation/compare_cycles.py test_tma_gemm \
  --csv configs/gemm_shapes_inference_server.csv

# Copy the selected simulator config into existing launcher directories
./validation/sync_config.sh test_tma_gemm
```

Validation outputs are stored under:

```text
triton_kernel_tracking/
├── test_tma_gemm/
│   ├── m512_n3000_k1536/
│   └── results/
└── test_flash_attn/
    ├── b32_h32_seq512_d64/
    └── results/
```
