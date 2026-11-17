# Warp-Level MMA Support

FlashGPU-Sim functionally simulates `mma.sync` register-fragment operations
and applies the selected configuration's tensor-core timing model. This
instruction family is separate from WMMA, WGMMA and TCGen05.

## Validated Variants

The SM120 integration suite checks these input formats and shapes against CPU
references:

| Input type | Shapes |
| --- | --- |
| F16 | `m16n8k8`, `m16n8k16`, `m8n8k4` |
| BF16 | `m16n8k8`, `m16n8k16` |
| TF32 | `m16n8k4`, `m16n8k8` |
| S8 | `m16n8k16`, `m16n8k32`, `m8n8k16` |

The functional implementation also contains U8 and additional F16 shape paths;
the table above identifies the integration-tested scope. FP64, S4/U4 and B1
MMA execution are not implemented in this path. A shape or type recognized by
the parser may still lack functional execution support.

Floating-point results use the simulator's format conversions and accumulation
model. See [numerical behavior](tensor_mma.md) for precision limitations and
[matrix execution controls](../../../../configs/README.md#matrix-execution)
for timing settings. Functional correctness checks do not establish timing
accuracy for every shape.

## Validate

From the repository root:

```bash
./tests/run_tests.py list-cases --arch sm120 --group mma
./tests/run_tests.py run --arch sm120 --group mma
```

See the [MMA test guide](../../../../tests/src/mma/README.md) for individual
case selection and reference checks, and [runtime tracing](../../../../docs/tracing.md)
for the `MMA` component.
