# Native exit preparation

SM100 cubins can contain `PREEXIT` before further computation or a guarded
`EXIT`. Treating it as retirement loses those instructions and active lanes;
rejecting it prevents otherwise supported kernels from starting.

Register `PREEXIT` with the existing zero-operand advance handler and project
it as an issued ALU instruction with no data destinations. Actual retirement
continues through the existing `EXIT` and shared CTA lifecycle. This is a
native instruction compatibility change, with no new PTX exit path and no
change to PTX scheduling or timing defaults. Its hardware scheduling cost has
not been separately calibrated.

`SassPreexit` verifies unguarded and predicated preparation, computation after
preparation, partial retirement at a guarded exit, and timing projection that
does not retire a warp. The same component binary fails on the previous
runtime and passes after this change. The `sm100-native-exit` CI job runs these
checks. Retained DeepInfer operators provide the full launch acceptance;
component tests alone do not establish that acceptance.
