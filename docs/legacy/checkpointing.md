# Inherited Checkpoint Support

FlashGPU-Sim retains the upstream GPGPU-Sim checkpoint/resume path. It writes
functional state under `checkpoint_files/`, including global, local and
shared memory, thread registers and SIMT stacks.

The inherited controls include `-checkpoint_option`, `-checkpoint_kernel`,
`-checkpoint_CTA`, `-checkpoint_CTA_t`, `-checkpoint_insn_Y`, `-resume_option`,
`-resume_kernel` and `-resume_CTA`. Checkpointing and resume are disabled by
default.

Current CI does not validate this workflow for FlashGPU-Sim's asynchronous
TMA, memory-barrier or TMEM execution. The inherited state format should not
be assumed to capture a complete modern timing-model snapshot. Use ordinary
workload replay for the documented validation flows in the
[test guide](../../tests/README.md).
