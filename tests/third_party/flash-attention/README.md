# FlashAttention Test Dependency

This directory owns the upstream FlashAttention checkout shared by the FA2 and
FA3 test workloads. The checkout is generated under `checkout/`; it is not
tracked by this repository.

From `tests/`, `./run_tests.py build` and `./run_tests.py run` prepare the
dependency automatically when an FA2 or FA3 build needs it. Preparation clones
the upstream repository, checks out the pinned revisions below, and applies the
patches in `patches/` before compilation starts. There is no separate
user-facing prepare command.

- FlashAttention: `d80a77103021c4e980f8cbbf85774f6a19e6474a`
- CUTLASS: `7127592069c2fe01b041e174ba4345ef9b279671`

The preparation script accepts these environment overrides:

- `FLASH_ATTENTION_DIR` - use a checkout outside the default `checkout/` path
- `FLASH_ATTENTION_REPO` - clone from a different repository URL
- `FLASH_ATTENTION_BASE_COMMIT` - select a different FlashAttention commit
- `FLASH_ATTENTION_CUTLASS_COMMIT` - select a different CUTLASS commit
- `FLASH_ATTENTION_FORCE=1` - reset an existing checkout before patching
