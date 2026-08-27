# Experimental RTX 5090 Instruction Cache

This configuration enables the bounded instruction stream buffer and replaces
the perfect instruction cache with an experimental 64 KiB per-SM cache while
leaving the default perfect constant-cache behavior unchanged. It is used by
the instruction-prefetch GEMM regression cases in
`tests/ci/perf/cases.toml`.

This is not a complete RTX 5090 instruction hierarchy. In particular, ICC
misses currently go through the simulator's normal lower-memory path; there is
no GPC-scoped GCC between ICC and L2. The 128-byte simulator line and eight-line
prefetch depth are tunable model parameters, not measured hardware values.

Do not replace `configs/SM120_RTX5090` with this configuration for normal cycle
validation. Hardware evidence, measured bounds, and the remaining GCC probes
are documented in `docs/rtx5090-instruction-cache.md`.
