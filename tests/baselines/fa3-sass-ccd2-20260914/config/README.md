# H100 SASS-frontend validation baseline

This configuration is intentionally narrow in scope: it is the stable H100
baseline for validating execution-driven FA-3 SASS functionality and timing.
It combines:

- the current `SM90_H100` core model, including SASS scheduling, WGMMA,
  register-file pressure, barriers, scoreboard behavior, and instruction
  latencies; and
- the historical memory/SoC settings used before commit `9fb3c149` introduced
  the experimental non-power-of-two partition mapping and remote-L2 latency.

Relative to `../SM90_H100/gpgpusim.config`, the frozen SoC differences are:

| Option | SASS validation value | `SM90_H100` value |
| --- | ---: | ---: |
| `-gpgpu_l2_partition_count` | `1` | `2` |
| `-gpgpu_l2_partition_extra_latency` | `0` | `150` |
| `dramid@` mapping bit | `8` | `9` |
| DRAM `CCD` | `1` | `2` |
| `-gpgpu_ipoly_non_power2_balanced` | `0` | `2` |

Both configurations keep 80 memory channels, two subpartitions per channel,
the same cache geometry, interconnect topology, clocks, and current core
timing. TMA OOB traffic remains disabled because the validation baseline must
never turn an out-of-bounds fill into a memory access.

Each of the 160 L2 subpartitions has an explicit 32 B/cycle data port.  At the
1.5 GHz core clock used for validation this represents 7.68 TB/s of aggregate
L2 data bandwidth; do not reinterpret the two subpartitions in one memory
channel as one 64 B/cycle subpartition.

Do not use this configuration to infer H100 SoC organization. Changes made to
improve the SASS core model should be applied to both SM90 configurations;
memory/SoC experiments belong in `SM90_H100`.
