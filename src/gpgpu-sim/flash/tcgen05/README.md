# TCGen05 Support

This directory contains the functional TCGen05/TMEM support used by the
Blackwell FA4 bring-up path.

Current scope:

- `cta_group::1` only.
- Dense `.kind::f16` MMA with f16/bf16 inputs and f32 accumulator output.
- TMEM allocation, deallocation, register load/store, and accumulator storage.
- Parser coverage for the common TCGen05 instruction family needed during FA4
  bring-up.

Not calibrated yet:

- TCGen05 MMA issue throughput and completion latency.
- TMEM load/store bandwidth, banking, and queueing.
- `tcgen05.cp`/`shift`/`commit` timing.

Unsupported by design for the current FA4 forward target:

- `cta_group::2`.
- Sparse, warp-specialized MMA variants, block scaling, and narrow precision
  kinds beyond the parser surface needed to reject unsupported instructions
  cleanly.
