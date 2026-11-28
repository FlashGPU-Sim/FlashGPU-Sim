# Shared BF16 min/max

PTX scalar and packed two-input BF16 `min`/`max` use the same pair primitive
as native `HMNMX2.BF16_V2` and the retained finite-only three-source native
form. Numeric operands win over either kind of NaN, two NaNs produce a NaN,
and negative zero is ordered below positive zero regardless of operand order.

Two dual-frontend numerical tests cover independent packed halves, aliased
outputs, both input orders, signed zeros, subnormals, infinities and both
quiet/signaling NaNs. NaN payloads are not required to match hardware bits.

This extraction qualifies the plain two-input PTX forms needed by MHC. PTX
`.NaN`, `.xorsign.abs`, `.ftz` and additional source operands are rejected;
this is not a claim that all PTX min/max variants are implemented. The existing
native three-input NaN rejection remains explicit. Timing parameters are
unchanged.
