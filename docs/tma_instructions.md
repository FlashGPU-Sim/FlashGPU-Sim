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
cd test/triton_trace
source .venv/bin/activate
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
cd test/triton_trace
python triton_kernel_tracking/tracker.py example_tensor_add.py
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
cd test/triton_trace/triton_kernel_tracking/example_tensor_add/launchers
make
```

### Step 6: Execute Under GPGPU-Sim

```bash
# From launchers directory, with setup_environment sourced
./kernel_add_launch1  # (or other generated launcher executable)
```

### Step 7: Validate Output

Check that tensor addition produces correct results for all dimensions (1D, 2D, 3D, 4D, 5D).

## Success Criteria

- [ ] PTX inspection passes (only stubbed instructions appear, no new unsupported ops)
- [ ] 1D launcher runs and validates output
- [ ] 3D launcher runs and validates output
- [ ] 4D launcher runs and validates output (including degenerate cases)
- [ ] 5D launcher runs and validates output
- [ ] 2D regression test still passes

## References

- **PTX ISA 9.1**: `docs/ptx_isa_9.1.pdf` section 9.7.9.25.4 (cp.async.bulk.tensor)
- **Known Limitations**: `FLASH.md` lines 295-299 (TMA subsystem limitations)
- **Triton Descriptor API**: [triton.language.make_tensor_descriptor](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html)
- **Test Environment Setup**: `docs/testing-instructions.md`
