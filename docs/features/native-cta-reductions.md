# Native CTA reductions

Native `BAR.RED.{AND,OR,POPC}.DEFER_BLOCKING` and `B2R.RESULT` support
compiler-lowered CTA predicate reductions. PTX and native execution use one
arithmetic accumulator with operation-specific identities. Each frontend keeps
its existing barrier arrival/completion interface; PTX scheduling is unchanged.

A native reduction records contributions by barrier generation. Completion
publishes a result separately for each participating warp, so a fast warp
starting the next generation cannot erase a slow peer's completed result.
Nonparticipant retirement can release an implicit-count barrier. B2R observes
both functional completion and the existing deferred barrier timing handoff.
Timing projection treats its integer/predicate operands as destinations.

Qualified forms use an immediate barrier id, optional immediate thread count,
a source predicate (including negation), and converged full-warp participation.
POPC reads an integer destination; AND/OR read `RZ,Pn`. Register-selected
reduction barriers and additional native B2R forms remain unqualified. Repeated
B2R reads return the latest completed per-warp value; this deterministic model
choice is not an independent hardware validation of repeated reads.

The `sm100-cta-reduction` job combines seven native component checks with the same
compiled CUDA executable through PTX and native timing (four numerical tests,
six launches per frontend). It covers repeated
zero/partial/all reductions, multiple CTAs, explicit-count subgroups with
separate barrier ids, predicate negation, guarded instructions and result
isolation across generations. Focused tests also cover invalid operation/type
combinations without partial updates, retired nonparticipants and dependency
projection. Full retained routing/top-k acceptance is a separate gate.

This feature introduces no timing constants or production config changes.
