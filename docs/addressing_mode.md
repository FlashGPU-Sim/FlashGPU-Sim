# Generic Addressing Mode in FlashGPU-Sim

## Overview

FlashGPU-Sim implements PTX generic addressing mode, which provides a unified virtual address space that encompasses local, shared, and global memory regions. This document describes the addressing layout, critical limitations, and fixes implemented to support modern GPU architectures.

## Generic Address Space Layout

The generic address space is partitioned into distinct regions:

```
+---------------------------+ 0x0000000000000000
|                           |
|   Static Allocations      | .ptx global variables
|   (globals from .ptx)     | allocated at load time
|                           |
+---------------------------+ STATIC_ALLOC_LIMIT
|                           |
|   [Unused Gap]            |
|                           |
+---------------------------+ LOCAL_GENERIC_START
|                           |
|   Per-Thread Local Memory | MAX_THREAD_PER_SM * LOCAL_MEM_SIZE_MAX
|   (SM 0)                  | per SM (170 SMs × 2048 threads × 16KB)
|                           |
+---------------------------+
|   Per-Thread Local Memory |
|   (SM 1)                  |
+---------------------------+
|   ...                     |
+---------------------------+
|   Per-Thread Local Memory |
|   (SM N-1)                |
+---------------------------+ SHARED_GENERIC_START
|                           |
|   Per-SM Shared Memory    | SHARED_MEM_SIZE_MAX per SM
|   (SM 0)                  | (170 SMs × 1 MB = 170 MB)
|                           |
+---------------------------+
|   Per-SM Shared Memory    |
|   (SM 1)                  |
+---------------------------+
|   ...                     |
+---------------------------+
|   Per-SM Shared Memory    |
|   (SM N-1)                |
+---------------------------+ GLOBAL_HEAP_START = 0xC00000000
|                           |
|   Dynamic Heap            | malloc/cudaMalloc allocations
|   (Global Memory)         | grow upward from GLOBAL_HEAP_START
|                           |
+---------------------------+
```

### Address Space Constants

Defined in `src/abstract_hardware_model.h`:

```cpp
const unsigned long long GLOBAL_HEAP_START = 0xC00000000;  // 48 GB
const unsigned long long SHARED_MEM_SIZE_MAX = 1024 * (1 << 10);  // 1 MB
const unsigned long long LOCAL_MEM_SIZE_MAX = 1 << 14;  // 16 KB
const unsigned MAX_STREAMING_MULTIPROCESSORS = 1024;
const unsigned MAX_THREAD_PER_SM = 1 << 11;  // 2048
```

Derived constants:

```cpp
const unsigned long long TOTAL_LOCAL_MEM_PER_SM =
    MAX_THREAD_PER_SM * LOCAL_MEM_SIZE_MAX;  // 32 MB per SM

const unsigned long long TOTAL_SHARED_MEM =
    MAX_STREAMING_MULTIPROCESSORS * SHARED_MEM_SIZE_MAX;  // 1 GB

const unsigned long long TOTAL_LOCAL_MEM =
    MAX_STREAMING_MULTIPROCESSORS * TOTAL_LOCAL_MEM_PER_SM;  // 32 GB

const unsigned long long SHARED_GENERIC_START =
    GLOBAL_HEAP_START - TOTAL_SHARED_MEM;  // 0xBFFFFFC00

const unsigned long long LOCAL_GENERIC_START =
    SHARED_GENERIC_START - TOTAL_LOCAL_MEM;  // 0xBFFFFF400

const unsigned long long STATIC_ALLOC_LIMIT =
    GLOBAL_HEAP_START - (TOTAL_LOCAL_MEM + TOTAL_SHARED_MEM);  // 0xBFFFFF400
```

## Address Translation Functions

Located in `src/cuda-sim/cuda-sim.cc`:

### Shared Memory

```cpp
// Convert local shared memory address to generic address
addr_t shared_to_generic(unsigned smid, addr_t addr) {
  assert(addr < SHARED_MEM_SIZE_MAX);
  return SHARED_GENERIC_START + smid * SHARED_MEM_SIZE_MAX + addr;
}

// Convert generic address to local shared memory address
addr_t generic_to_shared(unsigned smid, addr_t addr) {
  return addr - (SHARED_GENERIC_START + smid * SHARED_MEM_SIZE_MAX);
}

// Check if address is in shared memory range for given SM
bool isspace_shared(unsigned smid, addr_t addr) {
  addr_t start = SHARED_GENERIC_START + smid * SHARED_MEM_SIZE_MAX;
  addr_t end = SHARED_GENERIC_START + (smid + 1) * SHARED_MEM_SIZE_MAX;
  return (addr >= start) && (addr < end);
}
```

### Local Memory

```cpp
// Convert local memory address to generic address
addr_t local_to_generic(unsigned smid, unsigned hwtid, addr_t addr) {
  assert(addr < LOCAL_MEM_SIZE_MAX);
  return LOCAL_GENERIC_START + (smid * MAX_THREAD_PER_SM) * LOCAL_MEM_SIZE_MAX +
         (LOCAL_MEM_SIZE_MAX * hwtid) + addr;
}

// Convert generic address to local memory address
addr_t generic_to_local(unsigned smid, unsigned hwtid, addr_t addr) {
  return addr - (LOCAL_GENERIC_START +
                 (smid * MAX_THREAD_PER_SM + hwtid) * LOCAL_MEM_SIZE_MAX);
}
```

### Global Memory

```cpp
bool isspace_global(addr_t addr) {
  return (addr >= GLOBAL_HEAP_START) || (addr < STATIC_ALLOC_LIMIT);
}
```

## Critical Bug Fixes

### Issue 1: Insufficient SM Support (80 → 1024 SMs)

**Problem**: Original `MAX_STREAMING_MULTIPROCESSORS = 80` was designed for Volta Titan V, but modern GPUs like RTX 5090 have 170 SMs.

**Impact**:
- Generic address calculation would overflow/wrap around for SMs >= 80
- Shared memory accesses from high-numbered SMs would access incorrect memory regions
- Catastrophic data corruption in multi-SM simulations

**Fix**: Increased `MAX_STREAMING_MULTIPROCESSORS` to 1024 to support current and future architectures.

### Issue 2: Insufficient Shared Memory (96 KB → 1 MB)

**Problem**: Original `SHARED_MEM_SIZE_MAX = 96 KB` was based on Volta specification, but Hopper/Blackwell support up to 228 KB per SM.

**Impact**:
- Kernels requiring more than 96 KB shared memory would corrupt adjacent SM's shared memory region
- Generic addressing calculation would be incorrect for large shared memory allocations

**Fix**: Increased `SHARED_MEM_SIZE_MAX` to 1 MB to accommodate future architectures.

### Issue 3: Global Heap Start Address Overflow (32-bit → 64-bit)

**Problem**: Original `GLOBAL_HEAP_START = 0xC0000000` (3.0 GB) was sufficient when total local+shared memory was small.

**Calculation with new limits**:
```
TOTAL_LOCAL_MEM  = 1024 SMs × 2048 threads × 16 KB = 32 GB
TOTAL_SHARED_MEM = 1024 SMs × 1 MB              = 1 GB
TOTAL            = 33 GB > 3 GB (old GLOBAL_HEAP_START)
```

**Impact**:
- `SHARED_GENERIC_START` would be negative (0xC0000000 - 1 GB < 0)
- Overlap between shared memory and global heap regions
- Catastrophic memory corruption

**Fix**: Increased `GLOBAL_HEAP_START` from `0xC0000000` (32-bit, 3 GB) to `0xC00000000` (64-bit, 48 GB).

### Issue 4: 32-bit Address Truncation in ld/st/mma Instructions

**Problem**: Memory instructions extracted addresses using `.u32` instead of `.u64`:

```cpp
// WRONG (before fix)
addr_t addr = src1_data.u32;  // Truncates to 32-bit

// CORRECT (after fix)
addr_t addr = src1_data.u64;  // Preserves full 64-bit address
```

**Impact**:
- Any address >= 0x100000000 (4 GB) would be truncated
- With `GLOBAL_HEAP_START = 0xC00000000` (48 GB), all heap allocations would be truncated
- Load/store operations would access completely wrong memory locations
- Silent data corruption

**Affected Instructions**:
- `ld_exec()` in `src/cuda-sim/instructions.cc:3497`
- `st_impl()` in `src/cuda-sim/instructions.cc:5934`
- `mma_ld_impl()` in `src/cuda-sim/instructions.cc:3707`
- `mma_st_impl()` in `src/cuda-sim/instructions.cc:3584`

**Fix**: Changed all address extraction from `.u32` to `.u64`.

## PTX Requirements

### Address Size Directive

All PTX files MUST include:

```ptx
.address_size 64
```

32-bit address mode (`.address_size 32`) is **no longer supported** due to the expanded address space requirements. A panic check enforces this requirement (see Implementation section).

### Example PTX Header

```ptx
.version 8.0
.target sm_120
.address_size 64  // REQUIRED

.visible .entry my_kernel() {
    // ...
}
```

## Validation and Safety Checks

Three panic checks are implemented to catch configuration errors early:

### 1. PTX Address Size Check

**Location**: `src/cuda-sim/ptx_loader.cc` (PTX parsing)

**Check**: Ensures `.address_size 64` is present in PTX files

**Error Message**:
```
GPGPU-Sim PTX: ERROR - 32-bit address size is not supported.
Modern GPU configurations require .address_size 64 in PTX files.
Found: .address_size 32
File: <ptx_file_path>
```

### 2. Config SM Count Check

**Location**: `src/gpgpu-sim/gpu-sim.cc` (config parsing)

**Check**: Ensures `num_shader` (SM count) ≤ `MAX_STREAMING_MULTIPROCESSORS`

**Error Message**:
```
GPGPU-Sim Config ERROR:
  Configured number of SMs (170) exceeds MAX_STREAMING_MULTIPROCESSORS (1024)
  defined in src/abstract_hardware_model.h

  This will cause address space overflow and memory corruption.

  To fix:
  1. If you need more than 1024 SMs, increase MAX_STREAMING_MULTIPROCESSORS
     in src/abstract_hardware_model.h and adjust GLOBAL_HEAP_START accordingly
  2. Reduce -gpgpu_n_clusters and -gpgpu_n_cores_per_cluster in your config
```

### 3. Kernel Shared Memory Check

**Location**: `src/gpgpu-sim/shader.cc` (kernel launch)

**Check**: Ensures kernel shared memory ≤ `SHARED_MEM_SIZE_MAX`

**Error Message**:
```
GPGPU-Sim Kernel Launch ERROR:
  Kernel 'my_kernel' requires 228 KB shared memory
  Exceeds MAX supported: 1024 KB (SHARED_MEM_SIZE_MAX)

  This will cause address space corruption.

  To fix:
  1. Increase SHARED_MEM_SIZE_MAX in src/abstract_hardware_model.h
  2. Adjust GLOBAL_HEAP_START to accommodate larger shared memory region
  3. Reduce kernel shared memory usage
```

## Example Calculation: RTX 5090

For RTX 5090 with 170 SMs:

```
LOCAL_GENERIC_START   = 0xC00000000 - 1 GB - 32 GB
                      = 0xBFFFFFC00 (not shown, simplify)

SHARED_GENERIC_START  = 0xC00000000 - 1 GB
                      = 0xBFFFFFC00

GLOBAL_HEAP_START     = 0xC00000000

Shared memory for SM 42:
  Start: 0xBFFFFFC00 + 42 * 1 MB = 0xC00002A00
  End:   0xC00002A00 + 1 MB      = 0xC00003A00

Local memory for SM 42, thread 100:
  Address = LOCAL_GENERIC_START + (42 * 2048 + 100) * 16 KB
          = 0xBFFFFFC00 + (86116) * 16384
          = 0xBFFFFFC00 + 0x54A4000
          = 0xC00054A4000
```

**Key Point**: All these addresses exceed 32-bit range (> 0xFFFFFFFF), demonstrating why 64-bit addressing is mandatory.

## Design Rationale

### Why Increase MAX_STREAMING_MULTIPROCESSORS to 1024?

- RTX 5090: 170 SMs
- Blackwell GB202: 192 SMs
- Safety margin for future architectures
- Minimal memory overhead (only affects address space layout, not actual memory usage)

### Why Increase SHARED_MEM_SIZE_MAX to 1 MB?

- Hopper GH100: 228 KB per SM
- Blackwell: potentially larger
- 1 MB provides comfortable headroom
- Per-SM allocation is virtual, not physical

### Why GLOBAL_HEAP_START = 0xC00000000?

- Must satisfy: `GLOBAL_HEAP_START > SHARED_GENERIC_START`
- With 1024 SMs: `TOTAL_LOCAL_MEM + TOTAL_SHARED_MEM ≈ 33 GB`
- 48 GB (0xC00000000) provides sufficient margin
- Still fits comfortably in 64-bit address space

## Memory Overhead Analysis

These changes increase **virtual address space** but **not physical memory**:

- **Virtual**: Generic address space expanded from ~3 GB to ~48 GB
- **Physical**: Memory is allocated on-demand per memory access
- **Simulator Memory**: Controlled by hash map implementation in `memory_space_impl`

## Performance Implications

- **No performance impact**: Address translation is simple arithmetic
- **No memory overhead**: Physical memory allocated lazily
- **Correctness**: Essential for simulating modern GPUs accurately

## Testing Recommendations

When developing with generic addressing:

1. **Enable PTX instruction tracing**: Set `PTX_INST_EXEC` trace flag
2. **Check address ranges**: Verify addresses in expected regions
3. **Test edge cases**:
   - Kernels with maximum shared memory
   - Configurations with many SMs
   - Large global memory allocations
4. **Use reduced configs for iteration**: `SM120_RTX5090_REDUCED` for faster debugging

## References

- PTX ISA Specification: Generic Addressing (Section 3.1.4)
- CUDA Programming Guide: Memory Hierarchy (Chapter 4)
- FlashGPU-Sim: `CLAUDE.md` (build and test instructions)
- Source Files:
  - `src/abstract_hardware_model.h` (address space constants)
  - `src/cuda-sim/cuda-sim.cc` (address translation functions)
  - `src/cuda-sim/instructions.cc` (memory instruction implementations)

## Summary

The generic addressing fixes ensure FlashGPU-Sim can accurately simulate modern GPUs:

1. **64-bit addressing mandatory**: All addresses extracted as `.u64`
2. **Expanded limits**: 1024 SMs, 1 MB shared memory per SM
3. **Correct heap placement**: `GLOBAL_HEAP_START = 0xC00000000`
4. **Validation checks**: Three panic checks prevent configuration errors

These changes are **backward compatible** for configurations within old limits (≤ 80 SMs, ≤ 96 KB shared memory) but **mandatory** for modern GPUs.
