# Experimental RTX 5090 Instruction Cache

This configuration enables the bounded instruction stream buffer and replaces
the perfect instruction cache with an experimental 64 KiB per-SM cache while
leaving the default perfect constant-cache behavior unchanged. It is used by
the approximately 10 us instruction-prefetch GEMM and FlashAttention
regression cases in `tests/ci/perf/cases.toml`.

This is not a complete RTX 5090 instruction hierarchy. In particular, ICC
prefetch misses in the first 512 launch-relative lines use a 16-cycle local GCC
return path. The path preserves the ordinary ICC tag, MSHR, sector-fill, and
fill-port behavior. Requests outside that fixed 64 KiB window still use the
normal lower-memory hierarchy.

This local path approximates the measured launch-time code preload; it is not a
GPC-scoped shared GCC. GCC sharing, lookup-miss/tag-hit latency, capacity, and
replacement are still unmodeled. The 128-byte simulator line and four-line
prefetch depth also remain tunable model parameters rather than measured
physical values.

`-gpgpu_icache_address_scale 2` maps each eight-byte PTX PC slot onto the
sixteen-byte address spacing used by Blackwell SASS before looking up the
instruction cache. This preserves functional PTX PCs and only changes cache
addresses. It is still an approximation because one PTX instruction does not
always lower to exactly one SASS instruction.

Do not replace `configs/SM120_RTX5090` with this configuration for normal cycle
validation. Hardware evidence, measured bounds, and the remaining GCC probes
are documented in `docs/rtx5090-instruction-cache.md`.
