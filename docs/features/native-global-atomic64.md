# Native returned-value 64-bit atomic add (private draft)

`ATOMG.E.ADD.64.STRONG.GPU` and `.SYS` use the existing integer RMW backend
and atomic timing path. This extends the native adapter, not the arithmetic
or memory subsystem. PTX already supports the corresponding operations.

The adapter reads and writes both halves of the source and returned-old-value
register pairs. The scoreboard and register-file projection retain both halves
as well. Aliasing the destination with the source or address pair is supported;
the address and increment are consumed before writing the returned value.
The zero-register destination discards the result. Unqualified operand
modifiers and invalid register pairs are rejected.

Two complete CUDA launch tests cover GPU and system scope, four CTAs contending
on the same counter, 64-bit overflow, returned values and adjacent memory guards.
Returned values are compared as a multiset, so the test requires atomicity
without assuming an inter-CTA scheduling order. Component tests cover aliases,
predication, discarded results, dependencies and malformed register pairs.
The inherited `sm100-global-reductions` job runs all eleven launch tests through
both frontends, retaining the nine existing reduction and ordering tests.

This is functional and dependency qualification. GPU/system scope share the
current simulator memory domain; no host/GPU coherence or separate system-scope
latency calibration is claimed. Full extracted DeepInfer acceptance remains
separate.
