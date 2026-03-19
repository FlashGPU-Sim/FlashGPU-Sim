# Test Configuration Matrix

This document provides a comprehensive reference for all test suites in GPGPU-Sim, their recommended GPU configurations, current status, and usage notes.

## Overview

The test framework supports multiple GPU configurations to balance validation thoroughness with resource usage. This matrix helps developers select the appropriate configuration for each test type.

## Active Test Suites

| Test File | Description | Recommended Config | Status | Notes |
|-----------|-------------|-------------------|--------|-------|
| `cuda_mma_f16_test.cc` | F16/BF16 MMA instruction tests | `SM120_RTX5090_REDUCED` | ✅ Active | Single-block functionality tests, fast execution |
| `cuda_mma_bf16_test.cc` | BF16 MMA instruction tests | `SM120_RTX5090_REDUCED` | ✅ Active | Single-block functionality tests, fast execution |
| `cuda_mma_tf32_test.cc` | TF32 MMA instruction tests | `SM120_RTX5090_REDUCED` | ✅ Active | Single-block functionality tests, fast execution |
| `cuda_mma_s8_test.cc` | S8/U8 MMA instruction tests (M16N8K16/K32, M8N8K16) | `SM120_RTX5090_REDUCED` | ✅ Active | Single-block functionality tests, fast execution |
| `cuda_tensor_mma_test.cc` | General tensor MMA tests | `SM120_RTX5090_REDUCED` | ✅ Active | Single-block functionality tests, fast execution |
| `cuda_tma_test.cc` | Tensor Memory Accelerator (TMA) tests | `SM120_RTX5090_REDUCED` | ⚠️ Partial | Contains 2 excluded tests (see below) |
| `cuda_ld_st_matrix_test.cc` | ldmatrix/stmatrix instruction tests | `SM120_RTX5090_REDUCED` | ✅ Active | Matrix load/store operations |
| `cuda_vector_add_test.cc` | Basic CUDA vector addition | `SM120_RTX5090` | ✅ Active | Multi-block baseline validation |
| `integration_test.cc` | General integration tests | `SM120_RTX5090` | ✅ Active | Multi-block validation |
| `mbarrier_test.cc` (integration) | Memory barrier tests | `SM120_RTX5090_REDUCED` | ✅ Active | Functionality validation |

## Excluded/Broken Tests

The following tests are currently excluded from test runs due to unimplemented features. These exclusions are hard-coded in `test/run_tests.sh` (line 328).

| Test Name | Test Suite | Issue | Status | Reason |
|-----------|------------|-------|--------|--------|
| `TMA.CPAsyncMethod` | `cuda_tma_test.cc` | Unimplemented `cp.async` instruction | Excluded | Uses `cp.async` PTX instruction not yet implemented in GPGPU-Sim |
| `TMA.PerformanceComparison` | `cuda_tma_test.cc` | Depends on CPAsyncMethod | Excluded | Internally calls CPAsyncMethod, which is excluded |

**Exclusion mechanism:**
```bash
# In test/run_tests.sh:328
local EXCLUDED_TESTS="-*CPAsyncMethod*:*PerformanceComparison*"
```

## Configuration Reference

### SM120_RTX5090 (Default)

**Use cases:**
- Full validation runs
- Multi-block test suites
- Performance testing
- Final verification before release

**Architecture:**
- 170 Streaming Multiprocessors (SMs)
- 16 memory controllers
- Full L2 cache hierarchy
- Complete interconnect network

**Resource usage:**
- Memory: High (simulates full GPU)
- Time: Slow (comprehensive simulation)

**Command:**
```bash
./test/run_tests.sh test
# or explicitly:
./test/run_tests.sh -c SM120_RTX5090 test
```

### SM120_RTX5090_REDUCED

**Use cases:**
- Quick smoke tests
- Development iterations
- Single-block functionality tests (MMA, TMA, mbarrier)
- CI/CD pipelines

**Architecture:**
- 1 Streaming Multiprocessor (SM)
- 1 L2 cache bank
- 1 DDR memory controller
- Minimal interconnect

**Resource usage:**
- Memory: Low (single SM simulation)
- Time: Fast (4-8x speedup vs. full config)

**Command:**
```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED test
```

## Test Selection Guidelines

### When to Use SM120_RTX5090_REDUCED

Use the reduced configuration for:
- All MMA test suites (F16, BF16, TF32, S8/U8, tensor MMA)
- ldmatrix/stmatrix tests
- TMA tests (except excluded ones)
- mbarrier tests
- Any single-block functionality test
- **CI/CD pipelines** (used by GitHub Actions workflow)

**Rationale:** These tests validate instruction-level functionality within a single thread block. Full multi-SM simulation is unnecessary and wastes resources.

**Example:**
```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED test "*MMA*"
```

**CI Usage:** The automated CI workflow (`.github/workflows/pr-tests.yml`) uses this configuration via `test/ci/run_ci_tests.sh` for resource-efficient testing on PR approval.

### When to Use SM120_RTX5090 (Full Config)

Use the full configuration for:
- Multi-block tests (vector add, integration tests)
- Performance benchmarking
- Full validation runs before release
- Tests requiring inter-SM communication

**Example:**
```bash
./test/run_tests.sh test CudaVectorAdd
```

## Usage Examples

### Run All MMA Tests (Fast)
```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED test "*MMA*"
```

### Run Specific Test Suite
```bash
./test/run_tests.sh -c SM120_RTX5090_REDUCED test MMAS8M16N8K16
```

### Run Full Validation
```bash
./test/run_tests.sh test
```

### List All Available Tests
```bash
./test/run_tests.sh list
```

### List Available Configurations
```bash
./test/run_tests.sh list-configs
```

## Maintenance Notes

### Keeping This Matrix Up to Date

This matrix is manually maintained. When adding new tests or configurations:

1. **Adding a new test file:**
   - Add entry to "Active Test Suites" table
   - Specify recommended config (use REDUCED for single-block tests)
   - Update doc-guard tests in `test/test_multiconfig.sh` if needed

2. **Excluding a test:**
   - Update `test/run_tests.sh` EXCLUDED_TESTS list
   - Add entry to "Excluded/Broken Tests" table with reasoning

3. **Adding a new config:**
   - Add entry to "Configuration Reference" section
   - Update usage guidelines as needed

### Doc-Guard Tests

The `test/test_multiconfig.sh` script includes automated checks to prevent this matrix from drifting:

- Verifies this file exists
- Confirms both config names are documented
- Ensures excluded tests are mentioned

These checks run as part of the test suite and will fail if the matrix is incomplete.

## Related Documentation

- [Testing Instructions](testing-instructions.md) - How to run tests
- [Test Framework README](../test/README.md) - Test framework overview
- [CLAUDE.md](../CLAUDE.md) - Quick reference for Claude Code

---

**Last updated:** 2026-01-05 (Issue #28 implementation)
