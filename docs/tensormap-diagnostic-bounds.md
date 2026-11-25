# Invalid tensor-map diagnostic bounds

The descriptor printer is also used to diagnose rejected descriptors. It
previously trusted the encoded rank as the array loop bound; a corrupt large
rank caused the diagnostic itself to read beyond the descriptor. Rank plus
one could also wrap at UINT32_MAX and misleadingly report zero dimensions.

Printing now bounds descriptor-array reads to their five physical entries,
labels invalid ranks, computes the reported dimension count without 32-bit
wraparound, and omits a derived tile size for an invalid rank. Descriptor
validation and valid memory-transfer semantics are unchanged.

A guard-page regression positions the complete descriptor at the end of a
readable page and prints ranks 5, 2581408 and UINT32_MAX. The public baseline
segfaults on the large-rank case. The same test binary passes with the fix.
All valid ranks 0 through 4 remain printable and their full descriptor text
is identical before and after. Three ordinary PTX isolation tests also pass.

Local CUDA 13.3 / SM100 qualification is recorded under workspace
`runs/public-tensormap-diagnostic-20261125/qualification.json`. Tests are wired
into all three architecture manifests; hosted CI has not been run. No
TCGen05 or private instruction feature is required.
