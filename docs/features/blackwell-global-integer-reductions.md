# Global integer reductions (private draft)

The operator acceptance stack needs ordinary 32/64-bit integer ADD, 64-bit
XOR and Blackwell's `red.async.release.{gpu,sys}.global.add` publication form.
PTX and native REDG/UREDGR adapters reuse the common integer RMW operation,
atomic serialization and LSU request/completion path. The existing native
32-bit ATOMG.ADD handler shares the vector reduction adapter.

The PTX adapter validates the supported integer forms and global address
alignment. Native UREDGR performs one operation for the active warp, whereas
REDG performs one for each executing lane. Both implicit high address words
and wide addend words participate in native dependencies. Lanes without a
uniform memory effect do not generate spurious requests at address zero.

The common backend distinguishes asynchronous reductions from ordinary atomic
operations. A release instruction waits for the issuing warp's preceding
tracked memory operations and shared-store visibility. Its own request does
not stop independent warp progress. PTX operands are captured at issue so
register updates and function returns cannot change a delayed RMW. Outstanding
requests still prevent CTA retirement, and fences account for them. The
release gate does not invalidate L1 or create another memory completion engine.

Tests include same-binary PTX/SASS launches, overflow and high-word addends,
conditional XOR, concurrent CTAs, GPU/SYS scopes, and a publishing function
that immediately overwrites its source register and returns. Clock checks
separately test nonblocking issue and ordering behind an artificial 512-cycle
shared-store visibility delay. These checks validate simulator behavior;
they do not calibrate real atomic latency. System scope here means ordering
within the simulated device, not newly implemented CPU/GPU coherence.

Other RED operations, floating reductions, shared-cluster asynchronous
reductions with mbarrier completion, MMIO, and multimem are not qualified by
this extraction and are not silently treated as integer global ADD.

Reference: [NVIDIA PTX red.async](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-red-async).
