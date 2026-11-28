# Shared addresses used by Blackwell cluster kernels

PTX shared-state pointers carry a cluster rank above the low 24-bit offset.
Generic pointers retain the simulator's separate owner-SM address windows.
The rank encoding is required by captured SM100 code that clears bit 24 to
select its even-rank partner before issuing a paired TMA completion.

`shared_address.h` defines the bit representation. `tb_cluster` resolves a rank
against the live CTA group; no address may silently alias a different group.
PTX symbol addresses, cvta/mapa, scoped shared accesses, asynchronous copies,
TMA, matrix loads, TCGen05 and barrier operands consume that representation.
CTA-scoped consumers discard rank bits; cluster-scoped consumers resolve them.
Both u32 and u64 shared mapa forms remain state-space addresses; generic mapa
retains generic addressing. SASS TMA reuses the same pure encoding helpers.

## Qualification

`SharedAddress.RankBitsMapWidthsAndLocalAsyncCopy` exercises two clusters and two
phases through both frontends: u32/u64 shared mapa, generic mapa/cvta roundtrip,
rank-bit comparisons, asynchronous local copies and local loads. Every output
is checked. `TmaTensorCluster.PairedCompletionWithRankBitMask` checks the captured
bit-clear idiom with real paired TMA completion and complete output checks.
The existing Driver TMA CI gate runs both fixtures.

Ordinary remote SASS shared load/store is not claimed here. Its retained
reproducer is `tests/dev/shared_address/remote_read_test.cu`: PTX passes; CUDA
13.3 emits a bare native LD.E form that the extracted frontend does not support.
That fixture is deliberately outside the qualified test manifest, not counted
as a dual-frontend pass. DSM instruction execution is separate from the cluster
TMA/TCGen05 operations validated here.

The fresh retained projection now passes the address-related blocker. Full
operator acceptance additionally requires the separate suspended-barrier wake
fix. Historical numerical residuals and process-exit failures must be reported
separately from instruction-level qualification. No hardware timing calibration
or complete DeepInfer acceptance is implied by these tests.
