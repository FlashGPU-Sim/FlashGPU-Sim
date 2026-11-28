# PTX aliased shuffle sources

PTX lanes execute serially on the host. When a shuffle source and destination
name the same register, later lanes could read values already overwritten by
earlier lanes. Snapshot active source lanes once per dynamic warp instruction.
Scratch storage belongs to the core and resets before each instruction; it is
not shared across simulator workers. Existing lane dispatch and timing remain.

This branch is based directly on public `flash` at `b3378cb9`. It does not need
the private TCGen05 or SASS feature stack. `WarpShuffle.*` is included in the
existing SM90/SM100/SM120 integration CI jobs.

Five numerical launch tests cover up/down/bfly/idx across segment widths
32/16/8/4 and all 32 deltas, plus a predicated repeated-instruction case.
On the public base, up, bfly and the repeated case fail; after the fix all five
pass through PTX on SM100_B200 using the same executable. No hosted CI or new
GPU measurement is claimed. Workspace evidence, including frozen runtimes:
`runs/public-shuffle-20261125/`.

Reproduce with the normal test runner after building and sourcing the simulator:

```sh
./tests/run_tests.py run --arch sm100 --group integration --gtest-filter 'WarpShuffle.*'
```
