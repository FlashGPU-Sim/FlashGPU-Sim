# PTX bit search and native compiler lowering

This slice extracts `fns.b32` from the retained private implementation. The
operation follows [PTX fns](https://docs.nvidia.com/cuda/parallel-thread-execution/#integer-arithmetic-instructions-fns):
a signed offset selects a set bit relative to an inclusive base; zero tests
only the base. Missing bits return `0xffffffff`. The adapter rejects bases
outside the defined 0..31 range. Its 64-bit offset magnitude avoids host
signed overflow at INT_MIN.

CUDA 13.3 sm_100a has no single native FNS in this lowering: it emits branch,
bit reverse, shifts, population counts and integer selection. SASS already
executes those primitives. Adding a second native FNS implementation would not
exercise the generated program. PTX uses the existing generic integer timing
class; its single virtual instruction is not a calibrated estimate of this
multi-instruction native sequence.

`CI_JOB=sm100-bit-search` runs the same CUDA executable through both frontends:
three launches check every base, offsets -33..33, eight mask patterns,
large positive/negative offsets, immediate operands, predication and source /
destination aliasing. A separate PTX-only launch covers INT_MIN. The CPU oracle
enumerates set-bit positions independently of the simulator's scan loop.

The original full test found a CUDA 13.3 lowering discrepancy for INT_MIN.
The generated sequence negates the offset in a 32-bit register, leaving
0x80000000, then uses a signed comparison against the population count.
The native simulator consequently returns 0 where PTX requires not-found.
The failed same-binary comparison and standalone probe disassembly are retained
in `runs/fns-extraction-20261125`. No native instruction is changed to mask this
edge case. This diagnosis is based on disassembly and simulation, not a new
hardware measurement. The qualified dual-frontend domain excludes INT_MIN;
the PTX implementation still tests it explicitly.

MegaMoE's retained PTX now parses beyond fns and stops at
`st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4`. This is a further
extraction dependency, not a claim that the prior private full-stack workload
was unsupported. Full DeepInfer acceptance remains pending.
