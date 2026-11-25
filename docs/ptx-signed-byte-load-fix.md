# Scalar signed-byte load extension

Scalar PTX `ld.s8` previously omitted sign extension into wider destination
registers. For example, a byte containing -128 became +128 in a 32-bit or
64-bit register. Include S8 in the existing S16/S32 extension path. This changes
load semantics without adding an instruction or changing timing parameters.

The fix is based directly on public `flash` at `b3378cb9`; it does not require
TCGen05, TMEM, MMA, gather4, or the private SASS instruction extensions.

`PtxSignedByteLoad` checks all 256 byte patterns for global and shared loads,
with signed and unsigned loads into both 32-bit and 64-bit destinations.
The fixture belongs to the ordinary integration group on SM90, SM100 and
SM120, and is therefore included by their existing CI jobs.

On SM100_B200, the test-only commit fails both tests on the public base; the
fix passes both tests using the same executable and configuration. Both runs
use 5768 cycles for global and 5804 cycles for shared. Frozen binaries, runtimes
and logs are retained in the workspace `runs/public-signed-byte-20261125/`.
This is local PTX qualification, not hosted CI, hardware calibration, or proof
of coverage for vector loads and every address space. The shared ld/ldu helper
is changed, but this fixture exercises ld, not uniform ldu.

Reproduce after building the simulator and sourcing `setup_environment release`:

```sh
./tests/run_tests.py run --arch sm100 --group integration --gtest-filter 'PtxSignedByteLoad.*'
```
