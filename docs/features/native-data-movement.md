# Native data movement compatibility (private draft)

This extraction connects instruction forms used by the retained SM100 DeepInfer
cubins to the existing functional memory and timing backend. It does not add a
second memory subsystem or change cache policy calibration.

Supported additions:

- `LDG.E.ENL2.256`, `STG.E.ENL2.256`, and `STG.E.NA.EFL2.256`: two explicit
  four-register groups, one aligned 32-byte transfer per executing lane.
  Disassembly lists the upper group first for loads and the lower group first
  for stores. Register dependencies include both groups, including noncontiguous
  groups. Masked four-operand forms remain rejected.
- `LDG.E.S8`: sign extension and a one-byte timing access.
- `LDG.E.LTC128B`, `STG.E.NA.128`, and generic `ST.E.64`: existing address,
  predication and memory-space handling with the corresponding transfer width.
- `MOV.SPILL` / `R2UR.FILL`: unguarded full-warp and single-active-lane forms.
  Fill requires uniform source values among active lanes. Arbitrary partial
  masks, operand modifiers and guarded forms remain rejected. Existing SM120
  hardware probes establish these limited functional forms; this is not an
  SM100 latency calibration or support for arbitrary spill encodings.

Global load/store handlers are moved out of `memory.cc` into
`global_memory.cc`. Existing and new 256-bit operations share one handler.
The separate PTX scalar signed-byte extension fix is documented in
[ptx-signed-byte-load.md](../fixes/ptx-signed-byte-load.md).

`sm100-native-data-movement` runs host checks for values, lane guards, register
ranges, transfer widths, zero-register groups and timing dependencies, followed
by compiled PTX/SASS launch checks for 256-bit copies and signed-byte loads.
The retained memory-forms suite is a separate before/after regression. Full
DeepInfer qualification remains necessary; these small tests alone do not
establish complete operator coverage.
