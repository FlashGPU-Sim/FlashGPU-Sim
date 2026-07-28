# Test Configuration Matrix

This document provides a comprehensive reference for all test suites in GPGPU-Sim, their recommended GPU configurations, current status, and usage notes.

## Overview

The test framework supports multiple GPU configurations to balance validation thoroughness with resource usage. This matrix helps developers select the appropriate configuration for each test type.

## Active Test Suites

| Test File | Description | Recommended Config | Status | Notes |
|-----------|-------------|-------------------|--------|-------|
| `cuda_mma_f16_test.cc` | F16/BF16 MMA instruction tests | `SM120_RTX5090` | ✅ Active | Single-block functionality tests |
| `cuda_mma_bf16_test.cc` | BF16 MMA instruction tests | `SM120_RTX5090` | ✅ Active | Single-block functionality tests |
| `cuda_mma_tf32_test.cc` | TF32 MMA instruction tests | `SM120_RTX5090` | ✅ Active | Single-block functionality tests |
| `cuda_mma_s8_test.cc` | S8/U8 MMA instruction tests (M16N8K16/K32, M8N8K16) | `SM120_RTX5090` | ✅ Active | Single-block functionality tests |
| `cuda_tensor_mma_test.cc` | General tensor MMA tests | `SM120_RTX5090` | ✅ Active | Single-block functionality tests |
| `cuda_tma_test.cc` | Tensor Memory Accelerator (TMA) tests | `SM120_RTX5090` | ⚠️ Partial | One source-disabled wrapper and one default-excluded test (see below) |
| `cuda_ld_st_matrix_test.cc` | ldmatrix/stmatrix instruction tests | `SM120_RTX5090` | ✅ Active | Matrix load/store operations |
| `cuda_vector_add_test.cc` | Basic CUDA vector addition | `SM120_RTX5090` | ✅ Active | Multi-block baseline validation |
| `integration_test.cc` | General integration tests | `SM120_RTX5090` | ✅ Active | Multi-block validation |
| `mbarrier_test.cc` (integration) | Memory barrier tests | `SM120_RTX5090` | ⚠️ Known failure | `MBarrierThreadLevelTest.DifferentThreadsArriveSameBarrier` reaches the liveness timeout |
| Hopper instruction suites | SM90 instruction semantics | `SM90_H100` | ✅ Active | Included in PR CI |
| FA2 smoke suites | Four FP16 forward smoke shapes | `SM90_H100` | ✅ Active | Included in PR CI; exercises ordinary `cp.async` |
| `Fa3PrefillFp16SmokeTest` | FA3 FP16 forward smoke shapes | `SM90_H100` | ✅ Active | Included in PR CI |
| `Fa3FwdHdim128Fp16IntegrationTest.FixedForwardCase` | Fixed FA3 forward integration case | `SM90_H100` | ✅ Active | Included in PR CI |

## Inactive/Default-Excluded Tests

The following legacy TMA coverage is not part of the default simulator run.
Ordinary `cp.async` itself is implemented and exercised by the SM90 FA2 smoke
tests; these entries describe the state of the older TMA comparison wrapper.

| Test Name | Test Suite | Issue | Status | Reason |
|-----------|------------|-------|--------|--------|
| `TMA.CPAsyncMethod` | `cuda_tma_test.cc` | Wrapper is commented out | Source-disabled | Needs re-enabling and validation against the current ordinary `cp.async` model |
| `TMA.PerformanceComparison` | `cuda_tma_test.cc` | Legacy default exclusion | Excluded | Its CP_ASYNC comparison entry is currently commented out; audit separately before restoring this multi-iteration performance test |

**Exclusion mechanism:**
```bash
# In the gtest-multi executor in test/run_tests.sh
local EXCLUDED_TESTS="-*CPAsyncMethod*:*PerformanceComparison*:MBarrierSanityTest.*"
```

### Known SM120 gating failure

`MBarrierThreadLevelTest.DifferentThreadsArriveSameBarrier` reaches the
100,000-cycle liveness timeout in detailed performance simulation with
`SM120_RTX5090`. The test remains active and is not added to the default
exclusion list; CI should continue to expose this performance-simulator
correctness issue until it is fixed.

### SM90 backward smoke coverage

The `fa3-smoke` registry group continues to include
`Fa3PrefillFp16BackwardSmokeTest.*`. PR CI gates the backward smoke cases with
`SM90_H100` alongside the FA3 forward smoke suite and fixed-forward
integration case.

## Pull Request CI Scope

Pull requests targeting `flash` run `test/ci/run_ci_tests.sh` in the CI
container. The gate checks:

- PTX scheduler SETP def/use classification;
- SM120 unit and integration suites with `SM120_RTX5090`;
- SM90 instruction and FA2 smoke suites with `SM90_H100`;
- FA3 forward smoke shapes, the fixed-forward integration case, and FA3
  backward smoke shapes; and
- the existing SM120 GPT-2 trace smoke tests.

The workflow runs three independent CI-container shards: `sm120`, `sm90-fa2`,
and `sm90-fa3`. Each shard invokes the same `test/ci/run_ci_tests.sh` entrypoint
and runs with a 7 GiB container memory limit. Simulator builds use two jobs;
FA2 kernel compilation uses one because each NVCC translation unit is
memory-heavy. FA3 standard kernel specializations compile as serial translation
units so no individual NVCC process exceeds the runner memory budget.

Each build/run group writes a text log under `test/logs/ci/logs/`; gtest groups
also write XML under `test/logs/ci/xml/`. The workflow uploads the entire
`test/logs/ci` tree even when a gate fails. Artifact names include the shard and
workflow attempt.

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
./test/run_tests.sh -c SM120_RTX5090 run test --target sm120 --group integration
```

### SM90_H100

**Use cases:**
- Hopper instruction validation
- FA2 and FA3 forward correctness smoke tests
- Full H100 validation and performance-oriented experiments

**Architecture:**
- 132 Streaming Multiprocessors (SMs)
- 80 HBM memory controllers
- 160 L2/HBM subpartitions

**Command:**
```bash
./test/run_tests.sh -c SM90_H100 run test --target sm90 --group fa2-smoke
```

## Test Selection Guidelines

Use `SM120_RTX5090` for SM120 suites and `SM90_H100` for Hopper suites.
These are also the defaults used by the pull-request CI entrypoint. Select a
different architecture-specific configuration only when the experiment calls
for it explicitly.

## Usage Examples

### Run All MMA Tests
```bash
./test/run_tests.sh -c SM120_RTX5090 run test --target sm120 --group integration MMA
```

### Run Specific Test Suite
```bash
./test/run_tests.sh -c SM120_RTX5090 run test --target sm120 --group integration MMAS8M16N8K16
```

### Run Full Validation
```bash
./test/run_tests.sh -c SM120_RTX5090 run test --target sm120 --group unit
./test/run_tests.sh -c SM120_RTX5090 run test --target sm120 --group integration
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
   - Specify the matching architecture configuration
   - Update doc-guard tests in `test/test_multiconfig.sh` if needed

2. **Excluding a test:**
   - Update `test/run_tests.sh` EXCLUDED_TESTS list
   - Add an entry to "Inactive/Default-Excluded Tests" with reasoning

3. **Adding a new config:**
   - Add entry to "Configuration Reference" section
   - Update usage guidelines as needed

### Doc-Guard Tests

The `test/test_multiconfig.sh` script includes automated checks to prevent this matrix from drifting:

- Verifies this file exists
- Confirms the default SM120 config is documented
- Ensures inactive/default-excluded tests are mentioned

Run them with `bash test/test_multiconfig.sh`. The scheduler regression also
runs directly in PR CI.

## Related Documentation

- [Testing Instructions](testing-instructions.md) - How to run tests
- [Test Framework README](../test/README.md) - Test framework overview
- [CLAUDE.md](../CLAUDE.md) - Quick reference for Claude Code

---

**Last updated:** 2026-07-28
