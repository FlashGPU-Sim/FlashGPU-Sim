# Integration Tests

These CUDA/GoogleTest workloads validate PTX execution and CUDA runtime
behavior across the architectures selected by the test runner.

| Source | Coverage |
| --- | --- |
| `address_operand_test.cu` | 64-bit address operands and large offsets |
| `vector_operand_test.cu` | Vector operands, compiler patterns and PTX transformations |
| `vector_add_test.cu` | Allocation, kernel launch and memory copies |
| `shared_memory_optin_test.cu` | Dynamic shared-memory opt-in |
| `ldst_matrix_test.cu` | `ldmatrix`/`stmatrix` variants |
| `ldst_multi_sector_test.cu` | Loads and stores spanning multiple memory sectors |
| `cp_async_src_size_test.cu` | `cp.async` source-size forms |
| `shared_atomic_address_test.cu` | Explicit shared addresses for atomic and ordinary accesses |
| `integer_multiply_test.cu` | Integer multiplication |
| `float_minmax_three_input_test.cu` | Three-input FP32 min/max |

Architecture selections determine which cases run. The three-input FP32
min/max case requires CUDA 12.9+ and an SM100+ compilation target; it reports
a skip with older toolchains or targets.

From the repository root:

```bash
./tests/run_tests.py list-cases --arch sm120 --group integration
./tests/run_tests.py run --arch sm120 --group integration
./tests/run_tests.py run --arch sm90 --group integration CpAsyncSrcSizeTest
```

See the [test guide](../../README.md) for configuration overrides, filters,
other architectures and native-GPU validation.
Run all 23 original integration assertions through strict execution-driven
SASS with:

```bash
./tests/run_sass_suite.py tests/src/integration/sass_functional_suite.json
```

Each case decodes the exact cubin registered by `integration_tests` with
NVIDIA `nvdisasm`/`cuobjdump` into the ignored content-addressed cache. The
suite stores filters and expected test counts only; it has no generated
manifest or kernel-name mapping. Unsupported decode or semantics fail closed
without PTX functional fallback.

Feature families that require multiple sources, shared headers, special build
flags, or architecture-restricted compilation use their own sibling test-group
directory instead.
