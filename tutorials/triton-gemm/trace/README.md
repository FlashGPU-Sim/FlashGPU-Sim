# Bundled GEMM Replay

This directory contains the selected launch from the default online capture:

- shape: `M=2560, N=64, K=2560`;
- target: `sm_120a`;
- grid: `(40, 1, 1)`;
- block: `(128, 1, 1)`;
- Triton config: `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=64`,
  `BLOCK_SIZE_K=64`, `GROUP_SIZE_M=8`, `num_stages=4`, and `num_warps=4`.

Triton autotuning requires a physical GPU, so `capture.sh` remains the
regeneration path for this bundle. `run.sh` copies the checked-in files into
`run/tracking/`, builds the standalone launcher, and replays it with
FlashGPU-Sim without running Triton or accessing a physical GPU.

When refreshing the bundle, retain the launch-specific harness, Makefile, PTX,
CUBIN, ptxinfo, captured inputs, and reference output. Generated executables,
fatbins, simulator configuration files, logs, and unselected autotune
candidates belong under `run/`.
