# FlashAttention Patches

`prepare_flash_attention.sh` applies this patch on top of upstream
FlashAttention commit `d80a77103021c4e980f8cbbf85774f6a19e6474a`.

Patch:

- `flash-attention-fa2-fa3-hooks.patch`

The patch preserves the local FA2/FA3 profiling hooks, sensitivity macros, and
task/globaltimer tracing used by the isolated debug targets. It also exposes
`FLASH_FWD_PACKGQA_CPASYNC_NOINC`, which lets the PackGQA regression build the
PTX `.noinc` form while accounting for its arrival in the barrier init count.
