# Runtime Tracing

Runtime traces record instruction, synchronization, and memory events in the
simulator's stdout log. Tracing is compiled in with `TRACE=1`, the default
build setting, and disabled at runtime by default.

## Enable and Select

Add these options to the run directory's `gpgpusim.config`:

```text
-trace_enabled 1
-trace_components WARP_SCHEDULER,INSTRUCTION_ISSUE
-trace_sampling_core 0
-trace_sampling_memory_partition -1
```

| Option | Default | Meaning |
| --- | --- | --- |
| `-trace_enabled` | `0` | Enable selected trace components |
| `-trace_components` | Empty | Comma-separated, exact component names |
| `-trace_sampling_core` | `0` | Sample one SM; `-1` selects all SMs |
| `-trace_sampling_memory_partition` | `-1` | Sample one memory partition, or one L2 subpartition for L2 request events; `-1` selects all |

Core sampling applies to core-scoped events. PTX module-loading events are
independent of SM selection. Component names are independent: enabling
`WGMMA_RF_TRAFFIC` does not enable `MMA`.

## Components

| Component | Records |
| --- | --- |
| `WARP_SCHEDULER` | Warp scheduling and issue decisions |
| `SCOREBOARD` | Register dependency reservation and release |
| `INSTRUCTION_ISSUE` | `ISSUE`, `STALL_SCOREBOARD`, and `STALL_READY_NO_ISSUE`, with warp, PC, instruction class/text and scoreboard producer class |
| `PTX_IR` | PTX extraction commands, selected files and loading overrides |
| `PTX_INST_EXEC` | Functional PTX instruction execution |
| `MMA` | MMA/WMMA shapes, completion, fragment values, lane mappings and matrix load/store details |
| `WGMMA_RF_TRAFFIC` | Register-file traffic token additions, collector drains and remaining backlog |
| `NAMED_BARRIER` | Named-barrier arrivals, release decisions and delayed warp releases |
| `MBAR` | Memory-barrier operations, per-lane wait results, rechecks and completion |
| `TCGEN05` | TMEM allocation, MMA and shared descriptors, register/TMEM transfers |
| `TMA` | Bulk/tensor transactions, memory requests and responses, completion and barrier arrivals |
| `MEMORY_PARTITION_UNIT` | Memory-partition activity |
| `MEMORY_SUBPARTITION_UNIT` | L2 activity and request events `REQ`, `CACHE_ACCEPT`, `RESP` |
| `INTERCONNECT` | Interconnect activity, request NoC events and gem5 bridge memory events |
| `LIVENESS` | CTA admission, thread exit and resource release |

`INSTRUCTION_ISSUE` does not cover every stall cause or guarantee one record
per instruction in a multi-issue scheduling iteration. Matrix/fragment dumps
and tracing every SM can produce substantial output and slow simulation.

## Capture and Interpret

Redirect the workload's stdout and stderr to retain a complete run log:

```bash
./workload > simulation.log 2>&1
```

L2, request NoC, TMA and gem5 events use key/value records that can be
post-processed into tables. Request records include identifiers, addresses,
request types, sizes and sector masks where applicable.

- L2 events use GPU cycle timestamps and identify the L2 subpartition.
- NoC events use `icnt_cycle` for the network event and `gpu_push_cycle` for
  the request's status timestamp. `PRE_ARB` reports outputs with at least two
  competing inputs. Its request metadata represents a sampled request;
  requester and queue counts include all contenders.
- The gem5 bridge reports `GEM5` events `PUSH`, `DRAIN_SEND` and `POP`, with
  an explicit `gem5_tick`. These events appear only when the gem5 backend is
  active; `INTERCONNECT` also serves the regular interconnect.

Use the timestamp field associated with each subsystem when comparing events.
The [configuration guide](../configs/README.md) describes the corresponding
execution and memory model controls.
