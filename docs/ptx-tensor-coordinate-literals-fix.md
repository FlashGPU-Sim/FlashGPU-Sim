# Integer literals in ordinary TMA coordinates

CUDA emits tensor coordinates such as `{0, 1}`, but the texture/tensor operand
grammar accepted only register identifiers. Reuse the existing vector-component
grammar and validate ordinary TMA coordinates against the instruction rank.
Each component must be a 32-bit integer register or a representable 32-bit
integer literal. The signed lower bound and unsigned bit-pattern upper bound
are supported. Ordinary load and store coordinate positions are both covered.

This branch starts directly from public `flash` at `b3378cb9`. It does not add
gather4, narrow `mov.b16/b32` packing, TCGen05, or SASS semantics. The existing
public `mov.b64` literal-pack validation remains unchanged. Malformed tensor
ranks/types are rejected before predecode rather than accepted by the broader
grammar accidentally.

Test-only commits `6610e2bb` and `904c5e6e` cover ordinary load/store ranks 1–5,
negative and maximum-width literals, invalid float/width/count/register forms,
and a CUDA launch that copies the second row of a 2D tensor using `{0, 1}`.
Both the parser fixture and the CUDA launch abort on the public baseline.
The launch is in existing SM90/SM100/SM120 integration manifests; host parser
coverage is in SM100 unit CI. Evidence is retained in workspace
`runs/public-tensor-coordinates-20261125/`; hosted CI and hardware measurements
are separate from this local qualification.

Local qualification passes both host tests (ten load/store forms plus rejection
checks), the numerical tensor load, and three existing vector/pack numerical
controls. The before/after runs use the same frozen test executables. Public
baseline aborts while parsing the positive fixture, so no before/after timing
comparison is claimed for this new syntax.
