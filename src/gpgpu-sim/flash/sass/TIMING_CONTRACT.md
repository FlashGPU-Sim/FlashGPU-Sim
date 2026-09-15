# ISA-neutral timing contract

The timing backend models hardware effects, not source-ISA syntax. Every
execution-driven frontend must translate its instruction into the common
`inst_t`/`warp_inst_t` contract before the instruction enters issue or an
execution unit.

## Frontend lifecycle and scope

`flash/frontend/execution_frontend.h` owns the SIMT execution interface:
thread/CTA initialization, instruction fetch, architectural execution, control
flow, collective participant updates and barrier completion. The PTX adapter
is in `flash/frontend/ptx_shader_adapter.cc`; the native adapter is in
`sass/timing/shader_adapter.cc`. The shader delegates these operations rather
than selecting an ISA at every callback. A shader cannot switch frontends
while it has live CTAs.

Kernel frontend identity is explicit. Occupancy uses `kernel_resource_usage`
(registers, static shared memory, local memory and stack size), independent of
the loader. PTX retains lazy resource lookup because register allocation can
finalize metadata after launch construction; native loaders supply a snapshot.

This separates ISA semantics from timing policy, not GPU architecture from
all possible accelerators. The contract still uses warps, lanes, CTAs and
`warp_inst_t`; adapters have privileged access to shader state. Another SIMT
ISA can implement the adapter and project common instruction metadata. A
non-SIMT NPU needs its own execution/scheduling model, though memory and
interconnect models may be reusable. CUDA loading and legacy PTX checkpointing
are not generalized by this interface.

## Static decode metadata

The frontend owns syntax and encoding. At static decode it must populate the
ordinary instruction fields (operation class, pipeline, registers, memory
space, width, latency, and initiation interval) plus the hardware descriptions
needed by the newer asynchronous units:

- `wgmma_static_info_t`: MMA versus group-control operation, accumulator bytes,
  register-A traffic, wait-group index, and an optional native end-of-group
  marker;
- `mbarrier_static_info_t`: barrier action and arrive/expect-tx behavior;
- `async_copy_static_info_t`: optional source size and cp.async.mbarrier NOINC
  behavior; and
- `tma_static_info_t`: TMA action, source/destination spaces, dimensions,
  reduction, multicast, and group-control data.

Hardware latency/issue options are owned by `instruction_timing_config`, not
the PTX functional simulator. Legacy `-ptx_opcode_*` spellings are preserved
for configuration compatibility and apply to both frontends.

The PTX frontend fills these fields in `ptx_instruction::pre_decode()`. The SASS
frontend must derive the same descriptions from decoded SM90/SM120 opcode fields. A
timing unit must not recover them by inspecting a frontend instruction object.

Native dependency control and execution-unit occupancy are separate inputs.
The SASS frontend projects the instruction's encoded stall, yield, and barrier
fields as warp readiness constraints. It obtains latency and initiation tables
from the same GPU configuration consumed by the backend pipelines. Encoded
stall cycles do not reserve an SP, SFU, or tensor execution unit, and configured
pipeline occupancy must not be replaced by frontend-specific constants.
Instructions carrying explicit dependency control bypass the inferred register
scoreboard used by source ISAs such as PTX; applying both dependency models to
the same instruction would count a fixed-latency RAW dependency twice. The
explicit wait barriers remain pending until backend instruction completion, so
variable-latency dependencies still observe real cache and execution timing.

## Dynamic functional results

TMA stores/reductions consume shared memory and must observe outstanding
deferred CTA barriers before this frontend snapshots their source bytes.
`orders_deferred_cta_barrier()` is shared by functional execution and timing
readiness, including the configured named-barrier visibility delay. This
ordering does not assert a newly measured hardware issue latency; SM90 TMA
loads retain their independently calibrated penetration behavior.
Publishing `SYNCS.ARRIVE*` must also wait for a deferred barrier: otherwise
FA3's no-TMA producer can publish another Q completion before a delayed
consumer observes the prior phase. Independent SM90 phase polling remains
allowed. This is a shared-state ordering constraint, not a measured extra
latency or a parity-history workaround.

Addresses, byte counts, mbarrier addresses, register-selected named-barrier
ids, and other data-dependent values are known only after functional
execution. They live on the dynamic `warp_inst_t`; lane-specific TMA and
mbarrier values use `tma_dyn_info_t` and `mbarrier_info_t` arrays. The PTX
execution context copies these values, and the TMA static description finalized
by its functional handler, into the pipeline instruction in
`ptx_shader_frontend::execute()`.

A SASS execution context must write the same generic fields directly. Timing
units receive only the generic pipeline instruction; they must neither locate
the static instruction again nor access frontend-owned dynamic objects.

SASS try-wait results are recorded per executing lane. The timing adapter
coalesces identical address/parity pairs into one warp-wide wait; nonuniform
waits panic because the backend cannot release lanes independently. Functional
execution still evaluates those lanes individually. Other mbarrier operations
retain their per-lane effects, including arrival counts.

Warpgroup issue creates one timing operation for four participating warps. PTX
evaluates the instruction's functional semantics once for that collective;
SASS evaluates it for each participant because every warp owns a distinct
native register file and PC. The virtual `execute_collective_participant()` hook
applies the already-issued collective to the other participants: PTX advances
their PCs, while SASS also evaluates their architectural outputs. Neither path
may create an additional timing operation. Native GMMA's final `gsb`
operand is projected to the common end-of-group marker, while PTX continues to
use its separate `wgmma.commit_group` instruction.

## Shared atomic timing approximation

`ATOMS.ADD` performs its RMW once in CTA shared memory and returns the old
32-bit value, including destination/source aliasing and discarded RZ results.
It observes deferred CTA barriers like other shared-memory operations.

For now its timing uses the same response-driven L2 read proxy as `ATOMG`:
L1 is bypassed, completion releases the destination dependency, and no PTX
atomic callback executes. The adapter maps the shared offset into the existing
per-SM shared generic-address window for this timing request only. Functional
memory events still identify shared memory; no global-memory value is read or
modified by this proxy. This is an explicitly uncalibrated approximation, not
a claim that hardware shared atomics visit L2. It does not model per-CTA
physical shared placement, shared-atomic bank contention or `ATOMS.POPC` timing.

## Allowed and forbidden dependencies

Allowed in a timing path:

- `inst_t`, `warp_inst_t`, active masks, addresses, register identifiers, and
  the generic static/dynamic metadata above;
- shader/core/cache/interconnect configuration and runtime resource state; and
- frontend-neutral memory services when the timing model actually transfers
  data rather than consuming addresses produced by functional execution.

Forbidden in a timing path:

- casts from a pipeline instruction to `ptx_instruction` or a future SASS
  instruction class;
- access to PTX or SASS frontend-owned thread/register state;
- PTX opcode constants, operand positions, option tokens, or text parsing; and
- fetching a frontend static instruction to reconstruct information that should
  already be carried by `warp_inst_t`.

PTX functional semantic handlers in `cuda-sim`, `flash/tma.cc`,
`flash/mbarrier.cc`, and `flash/wgmma/tensor_wgmma.cc` are intentionally outside
this rule. The single PTX cast in `exec_shader_core_ctx::func_exec_inst()` is an
adapter at the PTX frontend boundary, not a timing-unit dependency. It should
disappear when PTX functional execution itself writes generic dynamic metadata.

## Review invariant

The audited handoff points are:

| Effect | Frontend boundary | Backend consumer |
| --- | --- | --- |
| WGMMA shape, RF traffic, commit/wait | `timing_projection.cc`; `execute_collective_participant()` advances participant state | `shader.cc` uses `wgmma_static_info_t`, not native operands |
| Named barriers and mbarrier lanes | `timing_runtime::step()` and `project_mbarrier_effects()` | `barrier_set_t` consumes barrier ids/counts and `mbarrier_*_info_t` |
| Tensor/linear TMA | Runtime copies lane effects, including the descriptor snapshot | `tma_unit_t::warp_reaches_tma()` creates transactions from static/dynamic info; it does not fetch frontend instructions |
| Memory space, address, width, predication | Runtime translates functional accesses and local offsets | Shared coalescer, bank/data-path and memory pipelines consume `warp_inst_t` |
| Native dependencies | Projection supplies control fields; shader issue records completion dependencies | Encoded dependency tracking is separate from the PTX inferred scoreboard |

Opcode recognition in the SASS adapter is intentional. In particular, its
async-proxy epoch tracking currently recognizes `UTMALDG` and
`FENCE.VIEW.ASYNC.S`; this is not evidence that every linear bulk-copy opcode
has the same modeled visibility behavior. The shared `async_proxy_timing`
helper itself accepts epochs and completion-token flags, not ISA syntax.

When adding a new instruction feature, first extend the generic contract, then
teach each frontend to populate it, and finally consume it in timing. Do not add
an ISA-specific shortcut to shader issue, WGMMA, TMA, cp.async, or mbarrier
timing code.
