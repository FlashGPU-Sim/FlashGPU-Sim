# SM100_B200 Configuration

PTX execution-driven configuration for datacenter Blackwell
(`sm_100` / `sm_100a`), including TCGen05 and TMA modeling.
The configuration combines architectural limits with effective timing and
bandwidth models; it is not a complete reconstruction of the B200
microarchitecture.

## Modeled device

| Property | Configuration |
| --- | --- |
| Compute capability | 10.0 |
| SMs | 148 |
| Resident threads / warps per SM | 2048 / 64 |
| Shared memory per SM | 228 KiB |
| Default / opt-in shared memory per block | 48 / 227 KiB |
| Unified L1/shared capacity per SM | 256 KiB |
| L2 topology | 16 channels, 12 slices per channel |
| Modeled L2 capacity | 126 MiB |
| Core / interconnect / L2 / DRAM clocks | 1080 / 1080 / 1155 / 3996 MHz |

See [gpgpusim.config](gpgpusim.config) for the complete parameter set.

## Compute timing

ALU scoreboard forwarding is enabled. Dependent instructions may consume a
ready result before physical writeback; execution queue delays and execution
and writeback resource occupancy remain modeled.

The following values are in core cycles. Latency specifies the modeled result
timing, while initiation interval controls execution-unit issue spacing.

| Instruction class | Latency | Initiation interval |
| --- | --- | --- |
| Scalar FP32 ADD / MUL / MAD | 4 | 1 |
| FP32 MIN / MAX | 5 | 1 |
| Packed `f32x2` arithmetic | Corresponding scalar FP latency | 2 |
| Packed `cvt.f16x2.f32` | 4 | 2 |
| EX2 | 18 | 8 |
| Other instructions using generic SFU timing | 28 | 8 |
| SETP / SELP | 5 | 1 |

Packed conversion and predicate instructions use the INT execution path in
this configuration. Packed arithmetic uses SP; EX2 uses SFU. No separate
packed-conversion subpipeline is modeled.

These values include Blackwell-family approximations informed by SM120
measurements and SM100 compiler scheduling. They are not all direct B200
measurements; compiler scheduling alone does not establish hardware latency.
The common packed-arithmetic and conversion rules extend beyond individually
measured variants, and FP min/max timing has limited independent validation.

## Memory and synchronization model

Shared loads use a 19-cycle shared-memory pipeline. A named barrier waits for
earlier shared stores from its warp to leave LD/ST dispatch plus four core
cycles. This is a timing fence; functional shared-memory updates occur at
instruction execution.

A nonblocking named-barrier arrive has two overlapping delays measured from
issue: the warp can resume ordinary instructions after 18 core cycles, and
the arrival is visible to waiters after 24 core cycles. These delays are not
added together. Later named-barrier operations remain ordered behind pending
arrivals, and warp retirement waits for delivery. These values are effective
timing approximations. They do not change the separate mbarrier/try-wait model.

- L2 lookup and fill service each allow three 32-byte sectors per slice per
  L2 cycle. Data service allows five sectors per three L2 cycles, modeling
  shared bandwidth of 20.111 TB/s at 1964 MHz L2 and 11.827 TB/s at the default
  1155 MHz. ROP-delay output has a separate budget of two sectors per L2 cycle.
  Delayed ingress uses a bounded pipeline budget derived from its configured
  bandwidth and latency, plus the existing output FIFO capacity.
- Memory-channel selection uses the configured IPOLY mapping. The 12 L2 slices
  per channel use a separate stable-rotation policy. These mappings and service
  widths are modeling assumptions, not measured physical hashes or port counts.
- TMA and ordinary `cp.async` use 32-byte requests with issue and response
  widths of four. Mixed traffic shares the cluster's transport budget.
- TMA permits up to 3200 outstanding child requests per SM. This is a
  bandwidth-delay-product bound for four responses per core cycle and an
  approximately 800-cycle memory endpoint, not a measured hardware queue depth.
- The simple DRAM model is enabled. Its delay and service parameters model
  aggregate latency and bandwidth; detailed DRAM timing and physical bank
  mapping are not independently calibrated.
- Each L2 instance tracks up to 768 outstanding 32-byte sector misses. The
  12 instances per memory channel provide 9216 entries to cover the configured
  7220 DRAM in-flight requests and buffering. This is an effective capacity
  approximation, not a measurement of the physical MSHR organization.
- CTA slot reuse has a 1200-core-cycle transition after outstanding work and
  resources drain. First use of a slot is not delayed. TCGen05 MMA has a
  160-cycle completion tail. Both are effective timing approximations rather
  than intrinsic hardware instruction latencies.

Application timing can be affected by interactions among these models.
Functional correctness and agreement on individual workloads do not establish
cycle accuracy for other workloads.

## Usage

Build the simulator and source `setup_environment` as described in the
repository setup instructions. Run tests from the repository root:

```bash
./tests/run_tests.py -c SM100_B200 run --arch sm100 --group unit
./tests/run_tests.py -c SM100_B200 run --arch sm100 --group integration
```

For PTX versions requiring a newer assembler, set `PTXAS_CUDA_INSTALL_PATH`
to a compatible CUDA toolkit while keeping `CUDA_INSTALL_PATH` on the toolkit
used to build the simulator.

The standalone TMA throughput benchmark can be run with:

```bash
make -C tests/src/microbench/tma ARCH=sm_100a PTX_PROFILE=compute_100a \
  throughput-sim
```

This target creates its own run configuration with clock domains
`1930:1930:1964:3996`; it does not use the base-clock tuple unchanged.
`THROUGHPUT_TMA_MAX_INFLIGHT` overrides its outstanding-request limit
(`0` means unlimited). Elapsed kernel time includes startup and drain.
For steady-state response bandwidth in bytes per core cycle, divide
`gpgpu_cluster_response_dispatch_transport_accepted_data_sectors` by
`gpgpu_cluster_response_dispatch_transport_service_ticks` and multiply by 32.

See the [CuTe DSL examples and validation guide](../../tests/dsl/cutedsl/README.md)
for frontend capture and replay workflows.
