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
covered by the current integration tests. Other shapes and element widths
are outside this documented support scope.

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
