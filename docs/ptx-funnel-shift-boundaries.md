# PTX funnel-shift boundary counts

The previous `shf` implementation performed 32-bit C++ shifts by 32 at PTX
counts zero and 32. Those operations are undefined in C++ and produced the
wrong result in the public runtime. A shared unsigned 64-bit primitive now
implements the concatenated low/high word operation with explicit wrapping
or clamping. Instruction scheduling and latency are unchanged.

The numerical regression executes all four PTX left/right and wrap/clamp
forms for counts 0 through 127, using lane-varying input words and an independent
host oracle. The same frozen executable fails on public base b3378cb9 and
passes all 512 output checks after the correction. Local CUDA 13.3 / SM100
was qualified; the fixture is also registered for SM90 and SM120.

This public fix contains the common primitive and existing PTX adapter only.
New native wrapping/right-shift opcode registrations remain in their feature;
that adapter uses this same primitive. No new SASS form is enabled by this fix.
Workspace evidence is `runs/public-funnel-shift-20261125/qualification.json`.
No hosted CI or public PR result is claimed.
