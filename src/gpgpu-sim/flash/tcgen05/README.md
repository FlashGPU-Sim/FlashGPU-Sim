# TCGen05 support

The shared functional backend and asynchronous timing engine support both PTX
and SASS adapters. Current qualified scopes and explicit remaining work are in
[Blackwell TMEM](../../../../docs/features/blackwell-tmem.md) and
[Blackwell MMA](../../../../docs/features/blackwell-mma.md).

The MMA extraction is a private draft: complete-launch coverage includes
single-CTA F16/BF16, TF32 TMEM-A, dense FP8 and selected block-scaled FP4 forms.
Component arithmetic tests do not establish end-to-end coverage for other
modifiers, paired CTA or Rubin.

Timing retains the existing configured work-based MMA service rate, fixed
completion tail, asynchronous queues and CP/LD/ST tables. No new hardware
calibration or physical TMEM/shared bandwidth model is implied here.
