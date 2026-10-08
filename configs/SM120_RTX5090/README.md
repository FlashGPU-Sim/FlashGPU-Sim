# SM120_RTX5090 Configuration

PTX execution-driven configuration for RTX 5090 (`sm_120` / `sm_120a`).
It combines device resource limits with effective timing and bandwidth models.
See [gpgpusim.config](gpgpusim.config) for the complete parameter set.

## Modeled device

| Property | Configuration |
| --- | --- |
| Compute capability | 12.0 |
| SMs | 170 |
| Resident threads / warps per SM | 1536 / 48 |
| Shared memory per SM | 100 KiB |
| Default / opt-in shared memory per block | 48 / 99 KiB |
| Unified L1/shared capacity per SM | 128 KiB |
| L2 topology | 16 channels, 3 slices per channel |
| Modeled L2 capacity | 96 MiB |
| Core / interconnect / L2 / DRAM clocks | 2580 / 2580 / 2317 / 14001 MHz |

## Execution and synchronization

ALU scoreboard forwarding exposes dependent results before physical writeback
while retaining execution occupancy. SETP/SELP result latency is five core
cycles. Classic `m16n8k16.f16` MMA uses a 34-cycle result latency and a
32-cycle initiation interval; other shape/type entries have separate timing.

Ordinary shared loads and stores have a four-cycle same-warp issue interval
and at least two dispatch cycles. LDMATRIX has a four-cycle minimum dispatch
time; STMATRIX uses its decoded collective width. Shared-memory latency is
29 core cycles. These service constraints are distinct from bank conflicts.

MIO uses a two-cycle SM-wide service interval and a 16-entry frontend queue.
Shared/shuffle mixed-workload measurements inform the shared service model;
its application to global loads, TMA, ordinary `cp.async`, and memory barriers
is a modeling assumption. The queue depth is an effective run-ahead capacity.

TMA permits 16 active transactions and 1024 outstanding child requests per SM.
An active TMA copy delays the issuing warp's next instruction by 168 core
cycles, separately from asynchronous data completion. The outstanding-request
cap is a capacity assumption, rather than a measured physical queue depth.

CTA `bar.sync` release uses 20 core cycles. No-hint `mbarrier.try_wait` uses a
34-cycle predicate-ready latency and maximum suspension bound; a successful
phase-notified suspended wait has an additional 110-cycle wakeup delay.
See the [mbarrier model](../../src/gpgpu-sim/flash/mbarrier.md) for wait semantics.

## Memory and instruction hierarchy

The L2 topology abstracts 48 service instances with 2 MiB per instance.
Lookup/data/fill widths are 2/4/1 32-byte sectors per L2 tick. Hit and fill
traffic share data-array service, while interconnect request and response
budgets remain separate. Instance count and associativity describe the model;
they do not reconstruct the physical cache organization or address hash.

Simple DRAM service uses one 32-byte atom per four DRAM ticks per controller,
giving 1.792128 TB/s at the configured clock. L2/ROP and DRAM fixed delays are
322 and 500 core cycles respectively. Dependent-load measurements inform
the miss-path delay. Each controller permits 684 outstanding requests and
each L2 slice has 256 sector MSHRs; these capacities cover the fixed-delay
bandwidth product with buffering headroom. Detailed bank timing is bypassed
in simple mode and is not independently calibrated by this configuration.

The instruction model uses a 64 KiB cache with 12 miss entries, address scale
2, one prefetch stream of depth 4, and one prefetch issue per cycle. GCC preload
uses 512 lines and a 16-core-cycle hit latency. These hierarchy parameters
are effective approximations. Register allocation and SASS-guided PTX
reordering are enabled.

The model uses fixed clocks. Hardware L2 clocks can vary under load, and
core-cycle delays require reassessment when clock domains change. Agreement
on one workload can include compensating errors among components.

## Usage

Build the simulator and source `setup_environment` as described in the
repository setup instructions. Run tests from the repository root:

```bash
./tests/run_tests.py -c SM120_RTX5090 run --arch sm120 --group unit
./tests/run_tests.py -c SM120_RTX5090 run --arch sm120 --group integration
```

See the [configuration guide](../README.md) for option meanings and the
[test guide](../../tests/README.md) for workload selection.
