# SM90_H100_REDUCED Configuration

Single-SM Hopper/H100 configuration for FA3 and Hopper instruction debugging.

This config keeps the full H100 per-SM FlashAttention execution features,
including ordinary `cp.async`, TMA, mbarrier, calibrated MMA/WGMMA issue,
PTX allocation/reordering, VOQ, and WGMMA RF-pressure modeling. It reduces the
simulated SM count from 132 to 1 and retains a reduced memory topology, so it is
intended for correctness/debugging and smaller logs, not performance or scaling
validation.

Key differences from `SM90_H100`:

- `-gpgpu_n_clusters`: 132 -> 1
- `-trace_sampling_core`: -1 -> 0
- Interconnect topology uses the reduced `k = 2` single-SM layout.
- Memory-controller count, L2 geometry/address mapping, DRAM timing, and clock
  domains remain reduced-model choices rather than full-H100 performance values.

Full/reduced parity is checked by
`test/scripts/check_reduced_config_parity.py`. Every field must match
`SM90_H100` unless it is explicitly listed as an allowed reduced-model
difference in that checker.

Known limitation: the full `fa3-smoke` registry group includes the
`Fa3PrefillFp16BackwardSmokeTest` suite, whose main backward kernel currently
hits the simulator liveness timeout. The failure still reproduces when the
newly synchronized execution knobs are reverted, so PR CI gates the FA3 forward
smoke suite and `Fa3FwdHdim128Fp16IntegrationTest.FixedForwardCase`; backward
coverage remains in the registry for explicit runs and follow-up debugging.

Example:

```bash
./test/run_tests.sh -c SM90_H100_REDUCED run test \
  --target sm90 --group fa3-smoke 'Fa3PrefillFp16SmokeTest.*'
```
