# TCGen05 Support

This directory contains the functional and execution-driven TCGen05/TMEM
support used by the Blackwell FA4 bring-up path.

Current scope:

- `cta_group::1` only.
- Dense `.kind::f16` MMA with f16/bf16 inputs and f32 accumulator output.
- TMEM allocation, deallocation, register load/store, and accumulator storage.
- Parser coverage for the common TCGen05 instruction family needed during FA4
  bring-up.
- A per-SM asynchronous timing unit with thread-stream MMA/CP/shift ordering,
  warp-stream LD/ST waits, commit-to-mbarrier completion, issue intervals,
  queue backpressure, and timing statistics.
- Dense FP16 MMA service time derived from decoded `2*M*N*K` work. The full and
  reduced B200 configs use an initial 13,764 FLOP/SM-cycle reference estimate
  derived from the published 2.2 PFLOP/s device throughput at the configured
  1.08 GHz; it is not an instruction-level calibration result.
- Optional inline-PTX surface validation through
  `tests/dev/tcgen05/run_phase1_smoke.sh --check-inline`, which requires a
  Blackwell-capable CUDA toolchain.

Implemented but not calibrated yet:

- MMA fixed completion tail and finite asynchronous queue depth.
- CP/shift and vector-width-specific LD/ST latency/initiation tables; defaults
  are one cycle until instruction-level B200 measurements are available.
- TMEM banking and contention. LD/ST still share the existing tensor-pipeline
  and register-file infrastructure; there is no independent TMEM byte-rate
  knob.

Unsupported by design for the current FA4 forward target:

- `cta_group::2`.
- Sparse, warp-specialized MMA variants, block scaling, and narrow precision
  kinds beyond the parser surface needed to reject unsupported instructions
  cleanly.
