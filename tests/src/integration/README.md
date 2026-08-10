# Portable Integration Tests

This directory aggregates self-contained CUDA GoogleTest translation units
that can be compiled for every architecture supported by the test system.

A test belongs here only when it:

- consists of one ordinary `.cu` test source;
- uses the shared GoogleTest/CUDA compile and link rules;
- needs no test-specific flags or external dependencies;
- compiles for every supported architecture without hiding its main body; and
- safely links into the architecture's `integration_tests` binary.

Architecture manifests still decide which cases run. Build portability does
not imply that every architecture must execute every case.

Current sources:

- `address_operand_test.cu`: 64-bit PTX address operands and large offsets;
- `vector_operand_test.cu`: PTX vector discard operands;
- `vector_add_test.cu`: CUDA runtime allocation, launch, and copy coverage;
- `shared_memory_optin_test.cu`: opt-in dynamic shared-memory behavior;
- `ldst_matrix_test.cu`: `ldmatrix` and `stmatrix` variants;
- `cp_async_src_size_test.cu`: `cp.async` source-size forms.

Run the test group with:

```bash
tests/run_tests.py run --arch sm120 --group integration
tests/run_tests.py run --arch sm90 --group integration CpAsyncSrcSizeTest
```

Feature families that require multiple sources, shared headers, special build
flags, or architecture-restricted compilation use their own sibling test-group
directory instead.
