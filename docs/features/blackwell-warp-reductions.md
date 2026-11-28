# Warp reductions through PTX and SASS

This private feature slice extends the retained Blackwell reductions through
both frontends. `flash/warp_reduce.h` owns the numeric operation and member-group
validation. `ptx_warp_reduce.cc` reads/writes PTX registers; the existing SASS data
movement module adapts REDUX/CREDUX. No numeric operation is duplicated in either adapter.

Qualified operations follow [PTX redux.sync](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-redux-sync):

- `.add/.min/.max` for `.s32` and `.u32`, with modulo-32-bit addition.
- `.and/.or/.xor.b32`.
- `.min/.max.f32`, independently with `.abs` and `.NaN`.
- FP32 signed zeros, infinities, subnormals, mixed NaN/numeric and all-NaN inputs.
  NaN propagation yields canonical `0x7fffffff`; absent `.NaN`, numeric inputs win.
- Co-issued full/partial masks, disjoint member groups, predicated execution,
  source/destination aliasing, and filtering PTX lanes that already exited.

PTX executes once per participating warp instruction, after snapshotting all
source values. A destination cannot change later lanes' reduction inputs.
Native results are uniform registers. CUDA 13.3 emits `WARPSYNC.EXCLUSIVE` inside
BSSY/BSYNC for variable member masks; disjoint groups serialize through the
existing reconvergence paths so each group consumes its own uniform result.

The two frontends use the existing SHFL/reduction timing class. This change does
not claim new hardware calibration. In particular, supporting floating CREDUX
does not mean its physical throughput has been independently measured.

## Validation

`CI_JOB=sm100-warp-reduce` runs host component tests and eight numerical launch
tests from one binary through PTX and SASS. The main kernel exercises all 17
operation/type/modifier combinations. Tests check overflow, aliasing, a partial
warp excluding lane zero, four disjoint groups, FP32 special values, and explicit
PTX predication. An eighth launch covers wrapping funnel shifts needed by
compiler-generated member-mask setup. The host reference is independent of the simulator helper.

## Boundaries

Rendezvous across separately issued divergent paths at a redux instruction is
not implemented here. PTX rejects a group containing live non-co-issued members;
SASS exclusive masks must partition the co-issued active lanes and use the compiler-emitted
BSSY/BSYNC region or a terminal region resumed at EXIT. These restrictions are checked, not treated as
completed synchronization. PTX exited-lane filtering is implemented; its separate
full-launch qualification is still pending.

Real DeepInfer MegaMoE still requires complete launch acceptance after extraction;
passing these instruction tests alone is not full workload acceptance. C4's
current independent blocker is BF16x2 arithmetic, confirmed at `mul.bf16x2` by a
debugger on the clean preceding branch. It is not missing FP32x2 dispatch.

The funnel-shift launch also exposed an independent PTX zero/full-width shift
bug (C++ shifts by 32). It is fixed separately with a shared wide-intermediate
helper and both clamp/wrap directions checked at boundary counts.

The disjoint-group launch exposed a separate cached-load completion defect:
multiple dynamic loads targeting the same register could share a register-level
pending count, and only the last load notified its dependency token. Cached
ordinary loads now count responses per dynamic instruction as well as keeping
the existing register-level count. A redux-free predicated-load launch and response-counter unit test accompany
this backend fix. The independent launch is a regression check, not a standalone
reproducer: it passes on the old runtime because compiler scheduling serializes
its loads. The failing-before/passing-after integration reproducer is the
disjoint-group redux launch. Public-main-only reproduction remains pending. No bypass,
atomic, shared-memory or LDGSTS completion path is changed by that fix.
