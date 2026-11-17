# FlashAttention Patches

`../prepare.sh` applies this patch on top of upstream
FlashAttention commit `d80a77103021c4e980f8cbbf85774f6a19e6474a`.

Patch:

- `flash-attention-fa2-fa3-hooks.patch`

The patch supplies FA2/FA3 profiling and sensitivity controls for the
benchmark profiles. It also exposes
`FLASH_FWD_PACKGQA_CPASYNC_NOINC`, which lets the PackGQA regression build the
PTX `.noinc` form while accounting for its arrival in the barrier init count.
