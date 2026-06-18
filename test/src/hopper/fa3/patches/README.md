# FlashAttention Patches

`prepare_flash_attention.sh` applies these patches on top of upstream
FlashAttention commit `d80a77103021c4e980f8cbbf85774f6a19e6474a`.

Apply order:

1. `flash-attention-fa3-profile-hooks.patch`
2. `flash-attention-fa2-fa3-sensitivity-hooks.patch`

The first patch preserves the local FA3 profiling hooks that were previously
stored as the `wzr/fa3-profile-hooks` submodule commit. The second patch
preserves the uncommitted FA2/FA3 sensitivity macros used by the isolated debug
targets.
