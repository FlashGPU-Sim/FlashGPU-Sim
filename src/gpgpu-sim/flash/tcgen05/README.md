# TCGen05 Support

This directory contains the functional and execution-driven TCGen05/TMEM
support for Blackwell PTX execution.

Current scope:

- `cta_group::1` only.
- Dense `.kind::f16` MMA with f16/bf16 inputs and f32 accumulator output.
- TMEM allocation, deallocation, register load/store, and accumulator storage.
- Parsing and validation for the TCGen05 instruction family; functional
  support is limited to the variants described here.
- A per-SM asynchronous timing unit with thread-stream MMA/CP/shift ordering,
  warp-stream LD/ST waits, commit-to-mbarrier completion, issue intervals,
  queue backpressure, and timing statistics.
- Dense FP16 MMA service time derived from decoded `2*M*N*K` work. The B200
  config uses 8192 FLOP/SM-cycle, a one-cycle issue interval, and a 160-cycle
  completion tail. These combine a counter-informed service rate with an
  effective completion overhead; they do not establish timing accuracy for
  every TCGen05 variant.

Model limits:

- MMA completion uses a fixed tail. The asynchronous queue depth is
  configurable; `0` leaves it unlimited, as in the B200 configuration.
- CP/shift and vector-width-specific LD/ST timing use configurable tables
  with provisional one-cycle defaults.
- TMEM banking and contention are not modeled. LD/ST share the tensor-pipeline
  and register-file infrastructure; there is no independent TMEM byte-rate
  knob.

Unsupported execution variants:

- `cta_group::2`.
- Sparse, warp-specialized MMA variants, block scaling, and narrow precision
  kinds beyond the parser surface needed to reject unsupported instructions
  cleanly.
