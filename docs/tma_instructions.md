# TMA (Tensor Memory Accelerator) Testing Instructions

## Background

This project currently implements basic functionality for `cp.async.bulk` and `cp.async.bulk.tensor` instructions, which control the TMA (Tensor Memory Accelerator) hardware unit. Related instructions include:
- `tensormap.replace`
- `tensormap.cp_fence`
- `cp.async.bulk.commit_group` (stubbed as NOP)
- `cp.async.bulk.wait_group` (stubbed as NOP)

**Target GPU**: RTX 5090
**CUDA Version**: 12.8
**PTX ISA Reference**: `docs/ptx_isa_9.1.pdf` section 9.7.9.25.4

## Testing Requirements

### 1. Multi-Dimensional Coverage (1D, 3D-5D)

The current implementation supports 1D-5D tensor operations, but only 2D is currently tested via `test/triton_trace/example_tensor_add.py`. This test plan adds coverage for:
- **1D**: Linear tensors with remainder handling
- **3D**: Multi-dimensional coordinate mapping
- **4D**: Degenerate cases (dimensions = 1)
- **5D**: High-dimensional indexing with non-uniform sizes

### 2. Stubbed Instructions Policy

**PTX Inspection Allowlist**: The following instructions are parsed but implemented as NOPs (no functional simulation). They are **explicitly allowed** to appear in PTX extracted by triton_tracker:

- `cp.async.bulk.commit_group` - NOP with debug logging (see FLASH.md:297)
- `cp.async.bulk.wait_group` - NOP with debug logging (see FLASH.md:297)
- `tensormap.cp_fence` - Parsed but not functionally simulated

**Rationale**: These instructions are documented as stub implementations in `FLASH.md`. PTX inspection should verify **no other unsupported instructions** appear, but these three are expected and allowed.

### 3. Device-Side Tensormap Creation

**Requirement**: All tensormap descriptors **MUST** be created on the device side using Triton's `tl.make_tensor_descriptor` within the kernel, **NOT** passed as host-side parameters.

**Example** (from `test/triton_trace/example_tensor_add.py`):
```python
# Inside @triton.jit kernel - CORRECT
desc = tl.make_tensor_descriptor(
    base=ptr,
    shape=[M, N],
    strides=[N, 1],
    box=[BLOCK_M, BLOCK_N],
    elem_ty=tl.float32
)

# Host-side creation - INCORRECT for these tests
# (Do not create CUtensorMap on host and pass as parameter)
```

**Rationale**: This validates device-side descriptor creation behavior and aligns with Triton's JIT compilation model.

### 4. Corner Case Coverage

Tests should cover edge cases within reasonable memory/runtime limits:
- Tensor dimensions not divisible by tile size (remainder handling)
- Minimal dimensions (e.g., some dims = 1)
- Non-uniform dimension sizes
- **NO stress testing required** (stay within practical memory limits)

## Test Workflow

### Step 1: Setup Environment

```bash
# Activate GPGPU-Sim environment
source setup_environment

# Activate Triton virtual environment (for kernel tracking)
source test/triton_trace/.venv/bin/activate
```

### Step 2: Create/Modify Test Kernels

Edit `test/triton_trace/example_tensor_add.py` to add multi-dimensional test variants:

```python
# Add parameterized kernel paths for 1D, 3D, 4D, 5D
# Keep tl.make_tensor_descriptor inside kernel
# Cover corner cases (remainder tiles, minimal dims)
```

### Step 3: Extract PTX via triton_tracker

```bash
python test/triton_trace/triton_kernel_tracking/tracker.py example_tensor_add.py
```

This generates launcher artifacts in:
```
test/triton_trace/triton_kernel_tracking/example_tensor_add/launchers/
```

### Step 4: PTX Inspection

**Verify PTX contains only supported instructions**, allowing the three stubbed instructions listed in Section 2.

```bash
# Check extracted PTX (in launcher artifacts directory)
grep -E "cp\.async\.bulk\.(commit_group|wait_group)|tensormap\.cp_fence" *.ptx
# These are allowed (stubbed)

grep -v -E "cp\.async|tensormap|mbarrier" *.ptx | grep "unsupported_instruction"
# Should find NO other unsupported instructions
```

### Step 5: Build Launchers

```bash
make -f ./test/triton_trace/triton_kernel_tracking/example_tensor_add/launchers/ kernel_add_1d_launch2_Makefile
```

### Step 6: Execute Under GPGPU-Sim

**CRITICAL**: Tests MUST be run with GPGPU-Sim environment sourced. Without `setup_environment`, tests will run on real GPU instead of simulation!

```bash
# REQUIRED: Source GPGPU-Sim environment (DO NOT SKIP!)
source /path/to/gpgpu-sim/setup.sh
source /path/to/gpgpu-sim/setup_environment

# Method 1: Using bash -c with environment (RECOMMENDED)
bash -c "source /path/to/gpgpu-sim/setup.sh && \
         source /path/to/gpgpu-sim/setup_environment && \
         cd /path/to/launchers && ./kernel_add_1d_launch2"

# Method 2: If already in launcher directory with environment sourced
cd /path/to/gpgpu-sim/test/triton_trace/triton_kernel_tracking/example_tensor_add/launchers
./kernel_add_1d_launch2
```

**Notes**:
- Launchers require config files (gpgpusim.config, etc.) in the same directory
- Without `setup_environment`, launcher will use real CUDA runtime and produce misleading results
- Verify GPGPU-Sim is active by checking for "GPGPU-Sim version" message at startup

### Step 7: Validate Output

Check that tensor addition produces correct results for all dimensions (1D, 2D, 3D, 4D, 5D).

## Known Issues and Fixes

### Issue #31: 4D/5D TMA Parser and Memory System Support

**Problem**: GPGPU-Sim failed to run `kernel_add_4d_launch4` and `kernel_add_5d_launch5` due to:
1. Missing parser support for 5-element coordinate vectors
2. TMA memory requests lacking proper byte/sector masks
3. TMA response handler not handling late responses from multi-mem_fetch transactions

**Root Causes**:
1. PTX parser (`src/cuda-sim/ptx.y`) lacked grammar for 5-element vectors needed for 5D coordinates
2. TMA AGU created mem_fetch without byte_mask/sector_mask, causing L2 cache assertion: `assert(0 && "no mf sent")`
3. TMA transactions issue multiple mem_fetch requests via AGU, but completion logic erased transaction while responses were still in flight, causing crash when late responses arrived

**Fixes Applied** (commits e20b0ffa + current):
- **Parser**: Added 5-vector operand grammar in `src/cuda-sim/ptx.y:725`
- **Parser**: Implemented `add_5vector_operand()` in `src/cuda-sim/ptx_parser.cc`
- **Parser**: Added declaration in `src/cuda-sim/ptx_parser.h`
- **Parser**: Mirrored changes in `cuobjdump_to_ptxplus/` for consistency
- **TMA Memory**: Compute byte_mask and sector_mask for requests (`src/gpgpu-sim/flash/tma.cc:687-698`)
- **TMA Response**: Handle both top-level and sector-subdivided responses (`src/gpgpu-sim/flash/tma.cc:604`)
- **TMA Completion**: Gracefully handle late responses after transaction completion (`src/gpgpu-sim/flash/tma.cc:607-622`)

**Test Status** (verified in GPGPU-Sim with `setup_environment` sourced):
- ✅ **1D TMA**: PASSED (8,192 elements validated, tolerance 1e-05)
- ✅ **3D TMA**: PASSED (262,144 elements validated, tolerance 1e-05)
- ✅ **4D TMA**: PASSED (1,048,576 elements validated, tolerance 1e-05, Grid 16x4x4 = 256 blocks)
- ✅ **5D TMA**: PASSED (1,048,576 elements validated, tolerance 1e-05, Grid 64x4x4 = 1024 blocks)

**All multi-dimensional TMA tests now pass successfully!**

## Success Criteria

- [x] PTX inspection passes (only stubbed instructions appear, no new unsupported ops)
- [x] 1D launcher runs and validates output
- [x] 3D launcher runs and validates output
- [x] 4D launcher runs and validates output ✓ Fixed in issue #31
- [x] 5D launcher runs and validates output ✓ Fixed in issue #31
- [ ] 2D regression test still passes (not re-verified)

## References

- **PTX ISA 9.1**: `docs/ptx_isa_9.1.pdf` section 9.7.9.25.4 (cp.async.bulk.tensor)
- **Known Limitations**: `FLASH.md` lines 295-299 (TMA subsystem limitations)
- **Triton Descriptor API**: [triton.language.make_tensor_descriptor](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html)
- **Test Environment Setup**: `docs/testing-instructions.md`
