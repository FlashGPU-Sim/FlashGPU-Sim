# Blackwell paired-CTA extraction (private draft)

`feature/blackwell-paired-cta` extends `feature/blackwell-mma`. Backend changes,
PTX/SASS adapters, and complete numerical launch fixtures travel together.
This branch does not introduce Rubin or change production presets.

## Implementation boundaries

- The common cluster scheduler resolves peer ownership. Neither adapter guesses
  peers from SM numbers. One CTA has one physical TMEM image regardless of the
  instruction's `cta_group` qualifier.
- Both adapters use the existing common descriptor, copy-address, scale, MMA,
  and accumulator-layout implementations. Group 2 collects operands from the
  two resident CTAs; it does not introduce another arithmetic implementation.
- Commit carries its CTA-group stream and multicast mask through the common
  TCGen05 completion engine. Each selected rank receives the completed arrival.
- Native allocator lowering additionally needs peer shared stores (`STAS`) and
  remote `SYNCS` notifications. Notifications use the existing DSM message
  transport; native architectural payload writes remain synchronous.
- The CUDA 13.3 native paired allocator uses the compiler-reserved shared
  barrier at offset `0x48`, with initial parity determined by rank. This ABI
  initialization is gated on the paired allocator opcode; ordinary cluster
  kernels do not acquire that hidden barrier.
- Native physical allocation chooses a free block common to the two peer
  allocation maps. PTX allocation remains a per-CTA collective; the qualified
  fixtures use matched allocation sequences in both CTAs. Arbitrary unequal
  prior allocation histories are not yet qualified.

## Numerical acceptance

`tests/src/tcgen05/mma_pair_test.cu` embeds PTX and sm_100a code in one executable.
The `sm100-pair` CI job runs component tests, then four numerical tests with
14 complete launches through each frontend:

| Family | Launches per frontend | Coverage |
|---|---:|---|
| Shared-A dense | 4 | FP16/BF16, M128/M256, accumulation and zero output-lane masks |
| TMEM-A dense | 8 | FP16/BF16, M128/M256, N32/N64, nonzero TMEM base and distinct physical halves |
| Block-scaled | 1 | FP4 x FP8, distinct A/B values and scale factors, multicast completion |
| Cluster isolation | 1 | Two independent CTA pairs in one launch |

SFB is the complete N-wide image replicated in both CTAs, not two scale halves
concatenated from the CTAs. The mixed fixture checks distinct column scales in
that replicated image, plus distinct per-CTA A scales and different A values in
each cluster. Host references are
computed from logical operands independently of simulator layout helpers.

Tests explicitly set cluster dimensions through the CUDA launch API. Scratch
configs use two GPCs of four SMs with DSM enabled, perfect instruction cache as
in the inherited preset, and disable PTX guided reorder for inline-PTX probes.
Production configuration files remain unchanged.

Component cases cover selected multicast destinations, physical storage aliasing
across group qualifiers, group-specific completion cutoffs, paired allocation
intersection, illegal owners, remote barrier effects, all STAS payload widths
and dependencies, and over-completion rejection before architectural stores.

## Limits requiring separate qualification

- Group-2 compact FP4, dense FP8, TF32, other TMEM-copy layouts/modifiers, remote
  unicast TCGen05 commit, and implicit cubin required-cluster launch metadata.
  Complete-launch acceptance above must not be read as full ISA coverage.
- Paired throughput/recurrence, multicast latency, and STAS payload bandwidth
  are not calibrated here. In particular, work is accounted on the issuing
  TC engine, multicast commit directly notifies peers at completion, and STAS
  times the completion message rather than a full payload transfer.
- CLC/Rubin extraction and fresh assembled DeepInfer acceptance remain later
  work. Historical private-stack results are not acceptance for this branch.

Reference: [NVIDIA PTX TCGen05](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions).

The explicit contiguous scale-stride follow-up adds a paired block-scaled MMA
fixture. The paired CI gate now requires 5 tests / 15 launches per frontend;
PTX and SASS reuse the common copy descriptor decoder.
