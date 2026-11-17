# TMA microbenchmarks

Build-only CUDA benches. The runner target is

```bash
./tests/run_tests.py build --arch sm90 --group microbench --profile tma
```

| Source | What it times |
|---|---|
| `tma_completion_latency_bench.cu` | 2D tensor TMA load completion, fixed GEMM tile sizes |
| `tma_descriptor_setup_bench.cu` | Tensor-map replace and publish |
| `tma_fa3_m3_bench.cu` | 4D `shared::cluster` tensor load of an FA3-shaped tile |

The linear `clock64` bulk-copy gtest is not in this tree.
