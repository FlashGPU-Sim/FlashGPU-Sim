# ldmatrix/stmatrix Support

FlashGPU-Sim models shared-memory matrix loads into register fragments and
stores from register fragments back to shared memory.

## Validated Variants

Integration tests cover both `ldmatrix` and `stmatrix` for:

| Shape and element width | Matrix count | Transpose |
| --- | --- | --- |
| `m8n8.b16` | `x1`, `x2`, `x4` | No |
| `m8n8.b16` | `x1` | Yes |

The transpose path also accepts `x2` and `x4`, but those combinations are not
covered by the current integration tests.

The Blackwell byte-matrix extension covers `stmatrix.m16n8.trans.b8` with
`x1`, `x2`, and `x4` through both PTX and native `STSM.8.MT168[.2/.4]`.
`matrix_byte_layout.h` is the shared lane/element mapping; the two frontends
only decode operands and record accesses for the common shared-memory backend.
It does not add the other Blackwell `ldmatrix` packing/conversion forms.

## Memory and Execution

These are warp collective instructions. The `.aligned` qualifier requires
converged execution; it does not imply a 128-byte address alignment. For
`m8n8.b16`, row fragments require natural 16-byte alignment. The simulator
does not validate address alignment; the integration tests use aligned shared
storage.

Non-transpose transfers access two adjacent `b16` elements per lane together;
transpose transfers access the two elements separately. Timing and shared
memory contention follow the selected configuration's matrix-transfer and
shared-memory controls.

See [matrix execution controls](../../../configs/README.md#matrix-execution).
Run the validated cases from the repository root:

```bash
./tests/run_tests.py run --arch sm120 --group integration '*Matrix*'
```

Coverage is defined in
[`ldst_matrix_test.cu`](../../../tests/src/integration/ldst_matrix_test.cu).

## Blackwell byte-matrix qualification

`CI_JOB=sm100-byte-matrix bash tests/ci/run_ci_tests.sh` runs native component
checks followed by nine same-binary PTX/SASS launches (CUDA 13.3, SM100a).
The launch matrix covers x1/x2/x4 with shared32, shared64 and generic64 addresses,
two CTAs with two warps each, noncontiguous permuted column bases, distinct
matrix payloads and untouched-byte guards. Expected output is computed from
logical row/column coordinates independently of the simulator helper.

The byte-matrix path rejects partial warps and unaligned column addresses.
The backend stores up to sixteen per-lane addresses to represent x4's sixteen
noncontiguous bytes without dropping accesses. Both frontends record byte accesses and use the common shared-memory
transaction path for this new form; existing m8n8 PTX dispatch is unchanged. No new hardware
latency/bandwidth calibration is claimed by these functional tests.

ISA reference: [NVIDIA PTX stmatrix](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-stmatrix).
For m16n8, `.trans` is mandatory according to the normative text and ptxas;
the manual's last x4 example omits it and should not be used as an allowed form.
