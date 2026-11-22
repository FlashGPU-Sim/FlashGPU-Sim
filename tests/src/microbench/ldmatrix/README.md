# LDMATRIX microbenchmark

This standalone probe measures a loop using `ldmatrix.sync.aligned.m8n8.x4[.trans].shared.b16` followed by integer accumulation. It was used as one constraint when calibrating the RTX 4060 GEMM model. Its per-loop slope includes the loop, instructions that consume the loaded registers, scheduling, and timing overhead. It is neither a direct single-instruction latency measurement nor a complete bandwidth test.

Build with CUDA 12.8, then execute one example on an appropriate SM 8.9 GPU:

```bash
nvcc -std=c++17 -arch=sm_89 -cudart shared -O3 -lineinfo \
  -o ldmatrix_probe ldmatrix_probe.cu
./ldmatrix_probe 8 1 0 1 native_8.bin
./ldmatrix_probe 32 1 0 1 native_32.bin
```

Arguments are steps, warps per CTA (1–4), transposed mode (0 or 1), number of CTAs, and output file. The program prints the `clock64` window for the first warp. Use repetitions and compare `(median_window_32 - median_window_8) / 24`; this still measures the complete loop. The input is deterministic and every lane writes four accumulation results. Compare all output bytes between hardware and simulator for the same arguments; no built-in CPU reference is provided.

The optional NCU profiling pass should be used separately to check dynamic `LDSM` count and shared-memory wavefront/conflict counters. Some NCU counter collections can perturb in-kernel `clock64` measurements substantially; measure timing separately and check `--clock-control` and `--cache-control` behavior. This probe was checked with base-clock NCU timing and repeated native timing; application frequency can vary. Report the selected method and raw results.
