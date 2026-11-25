# Scalar PTX signed-byte load extension

The scalar `ld`/`ldu` path extended signed 16- and 32-bit loads but omitted
signed 8-bit loads. Negative bytes consequently became positive values when
loaded into wider registers. The correction includes `S8_TYPE` in the existing
sign-extension path; unsigned loads and timing are unchanged.

The independent `PtxSignedByteLoad` CUDA launch checks signed and unsigned byte
loads into both 32- and 64-bit destinations, including -128, -1, 0, 1 and 127.
It requires no SASS, TMEM, MMA or TMA operations. Frozen before/after runtimes
run the same executable and configuration. Public-main-only qualification is
still required before presenting the stacked development commit as a public PR.

This patch is limited to scalar loads; it does not claim to repair the separately
noticed vector-load extension path.

Reference: [PTX relaxed-type destination operands](https://docs.nvidia.com/cuda/parallel-thread-execution/#relaxed-type-checking-rules),
which requires extension according to the instruction's signedness.
