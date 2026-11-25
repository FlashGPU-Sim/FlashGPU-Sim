# Full-width PTX integer literals

Decimal immediates above INT64_MAX previously saturated through `strtoll`.
Unsigned suffixes, octal and binary literals also aborted during lexing.
The lexer now parses a checked unsigned 64-bit magnitude for each radix,
preserves all bits in the existing signed semantic carrier, and applies
negative signs modulo 2^64 without host signed overflow. Overflow is rejected
instead of silently saturating. Instruction-specific conversion is unchanged.

`PtxIntegerLiteral` covers decimal values on both sides of INT64_MAX, UINT64_MAX,
four radices, unsigned suffixes, negative forms and out-of-range rejection.
A separate no-suffix decimal test exposes saturation without the old unsigned
suffix abort masking it. It also keeps INT64_MIN as a signed boundary control.

Local CUDA 13.3 / SM100 qualification used the same frozen test binary before
and after the production patch. Public base b3378cb9 fails all four regressions;
the corrected runtime passes all four and three existing PTX isolation tests.
Tests are registered in the SM90, SM100 and SM120 manifests; only the SM100
execution was qualified locally. No hosted CI or hardware run is claimed.

Workspace evidence: `runs/public-integer-literals-20261125/` (frozen runtimes,
binary hashes, logs, configuration and qualification.json). Files prefixed
`setup-error-` record an initial missing runtime environment and are excluded
from the before/after qualification.
