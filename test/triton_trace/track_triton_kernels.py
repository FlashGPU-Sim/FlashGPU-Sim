#!/usr/bin/env python3
"""
Triton Kernel Tracker and Binary Extractor

This script demonstrates how to track Triton kernel compilation and invocation,
and extract standalone binaries for simulation in GPGPU-Sim or other tools.

Usage:
    python track_triton_kernels.py [--output-dir OUTPUT_DIR] [--save-binaries]

Features:
    - Tracks all kernel compilations
    - Captures kernel binaries (CUBIN/PTX)
    - Records launch parameters (grid, block, args)
    - Generates standalone launcher code
    - Exports metadata for replay

Author: Generated for Triton runtime analysis
"""

import argparse
import json
import shutil
import sys
import os
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

import triton
import triton.language as tl

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARNING] PyTorch not available, tensor argument capture disabled")


@dataclass
class ArgumentInfo:
    """Information about a kernel argument"""
    index: int
    name: str
    arg_type: str  # 'tensor', 'scalar', 'pointer', 'constexpr'
    dtype: Optional[str] = None
    shape: Optional[Tuple[int, ...]] = None
    value: Optional[Any] = None  # For scalars
    data_file: Optional[str] = None  # For input tensors
    output_file: Optional[str] = None  # For output tensors (captured after kernel execution)
    size_bytes: int = 0


@dataclass
class KernelLaunchInfo:
    """Information about a single kernel launch"""
    timestamp: str
    kernel_name: str
    kernel_hash: str
    grid: tuple
    block: tuple
    shared_memory: int
    num_warps: int
    num_ctas: int
    args_info: List[ArgumentInfo]
    launch_id: int = 0
    stream: Optional[int] = None
    global_scratch_size: int = 0
    global_scratch_align: int = 1
    profile_scratch_size: int = 0
    profile_scratch_align: int = 1


@dataclass
class KernelBinaryInfo:
    """Information about compiled kernel binary"""
    kernel_name: str
    kernel_hash: str
    binary_path: str
    ptx_path: Optional[str]
    metadata_path: str
    metadata: Dict[str, Any]
    source_hash: str


class TritonKernelTracker:
    """Tracks Triton kernel compilation and invocation"""
    
    def __init__(self, output_dir: Path, save_binaries: bool = True, capture_args: bool = True, enabled: bool = True):
        self.output_dir = output_dir
        self.save_binaries = save_binaries
        self.capture_args = capture_args
        self.enabled = enabled  # Master switch for tracking
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for tracked data
        self.compiled_kernels: Dict[str, KernelBinaryInfo] = {}
        self.kernel_launches: List[KernelLaunchInfo] = []
        self.launch_counter = 0
        self.pending_args: Optional[tuple] = None  # Store args from JIT wrapper
        self.pending_grid: Optional[tuple] = None  # Store grid from JIT wrapper
        self.pending_args_snapshots: Optional[List[torch.Tensor]] = None  # Store pre-launch tensor snapshots
        self.function_to_hash: Dict[int, str] = {}  # Map function pointers to kernel hashes
        
        # Create subdirectories
        self.binaries_dir = output_dir / "binaries"
        self.metadata_dir = output_dir / "metadata"
        self.launchers_dir = output_dir / "launchers"
        self.data_dir = output_dir / "data"  # For argument data
        
        if save_binaries:
            self.binaries_dir.mkdir(exist_ok=True)
            self.metadata_dir.mkdir(exist_ok=True)
            self.launchers_dir.mkdir(exist_ok=True)
            self.data_dir.mkdir(exist_ok=True)
        
        # Install hooks
        self._install_hooks()
        
        print(f"[TritonKernelTracker] Initialized")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Save binaries: {self.save_binaries}")
        print(f"  Capture arguments: {self.capture_args}")
        print(f"  Tracking enabled: {self.enabled}")
    
    def enable(self):
        """Enable tracking"""
        self.enabled = True
        print("[TritonKernelTracker] Tracking ENABLED")
    
    def disable(self):
        """Disable tracking"""
        self.enabled = False
        print("[TritonKernelTracker] Tracking DISABLED")
    
    def is_enabled(self) -> bool:
        """Check if tracking is enabled"""
        return self.enabled
    
    def _install_hooks(self):
        """Install Triton runtime hooks"""
        # Hook for kernel loading (captures binaries)
        triton.knobs.runtime.kernel_load_end_hook.add(self._on_kernel_load)

        # Hook for kernel launch (captures invocations)
        triton.knobs.runtime.launch_enter_hook.add(self._on_launch_enter)

        # Hook for kernel exit (captures outputs after execution)
        triton.knobs.runtime.launch_exit_hook.add(self._on_launch_exit)

        # We use a custom pre_run_hook approach by monkey-patching
        self._patch_jit_run()

        print("[TritonKernelTracker] Hooks installed")

    @staticmethod
    def _normalize_grid(grid) -> tuple:
        """Normalize grid to a 3-tuple (x, y, z)"""
        if isinstance(grid, int):
            return (grid, 1, 1)
        elif isinstance(grid, (list, tuple)):
            if len(grid) == 1:
                return (grid[0], 1, 1)
            elif len(grid) == 2:
                return (grid[0], grid[1], 1)
            else:
                return tuple(grid[:3])
        else:
            return (grid, 1, 1)
    
    def _patch_jit_run(self):
        """Patch JITFunction.run to capture arguments and grid"""
        # Store original run method
        from triton.runtime.jit import JITFunction
        original_run = JITFunction.run
        tracker_self = self
        
        def patched_run(jit_self, *args, **kwargs):
            # Capture arguments and grid before running
            warmup = kwargs.get('warmup', False)
            if tracker_self.enabled and tracker_self.capture_args and not warmup:
                # Store args for the launch hook
                tracker_self.pending_args = args
                
                # Capture and evaluate grid like Triton does (see jit.py line 693-696)
                grid = kwargs.get('grid')
                if grid is not None:
                    # Build bound_args from the function signature
                    # This is a simplified version - Triton does more complex binding
                    if callable(grid):
                        # Evaluate grid function with metadata-like dict
                        # For grids like lambda meta: (cdiv(N, meta['BLOCK_SIZE']),)
                        # we need to pass constexprs from kwargs.
                        meta_dict = {k: v for k, v in kwargs.items() if k.isupper() or k == 'grid'}
                        meta_dict.pop('grid', None)  # Remove grid itself
                        try:
                            grid = grid(meta_dict)
                        except Exception as e:
                            raise RuntimeError(
                                f"Failed to evaluate grid function: {e}\n"
                                f"Grid function: {grid}\n"
                                f"Available meta keys: {list(meta_dict.keys())}\n"
                                f"This may indicate missing constexpr values in kwargs."
                            ) from e
                    
                    # Normalize to 3-tuple
                    tracker_self.pending_grid = TritonKernelTracker._normalize_grid(grid)
            
            # Call original run - this will trigger compilation and launch
            result = original_run(jit_self, *args, **kwargs)

            # Don't clear pending data here - launch_exit_hook needs it!
            # It will be cleared in _on_launch_exit
            # Note: If no launch happens (warmup), we should clear it
            if warmup or not tracker_self.enabled or not tracker_self.capture_args:
                tracker_self.pending_args = None
                tracker_self.pending_grid = None
                tracker_self.pending_args_snapshots = None

            return result
        
        # Replace the run method
        JITFunction.run = patched_run
        print("[TritonKernelTracker] Patched JITFunction.run to capture arguments and grid")
    
    def _on_kernel_load(self, module, function, name, metadata_group, hash_val):
        """Called when a kernel binary is loaded"""

        # We always track the kernel compilation.
        
        if hash_val in self.compiled_kernels:
            return  # Already tracked
        
        print(f"\n[TritonKernelTracker] Kernel loaded: {name} (hash: {hash_val[:8]}...)")
        
        # Extract binary and metadata files from cache
        binary_path = None
        ptx_path = None
        metadata_path = None
        
        for category, path in metadata_group.items():
            if category.endswith('.cubin'):
                binary_path = path
            elif category.endswith('.ptx'):
                ptx_path = path
            elif category.endswith('.json'):
                metadata_path = path
        
        if not binary_path and not ptx_path:
            print(f"  [WARNING] No binary found for {name}")
            return
        
        # Read metadata
        metadata = {}
        if metadata_path:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Save binary if requested
        saved_binary_path = None
        saved_ptx_path = None
        saved_metadata_path = None
        
        if self.save_binaries:
            kernel_dir = self.binaries_dir / f"{name}_{hash_val[:8]}"
            kernel_dir.mkdir(exist_ok=True)
            
            if binary_path:
                saved_binary_path = kernel_dir / f"{name}.cubin"
                shutil.copy(binary_path, saved_binary_path)
                print(f"  Saved CUBIN: {saved_binary_path}")
            
            if ptx_path:
                saved_ptx_path = kernel_dir / f"{name}.ptx"
                shutil.copy(ptx_path, saved_ptx_path)
                print(f"  Saved PTX: {saved_ptx_path}")
            
            if metadata_path:
                saved_metadata_path = kernel_dir / f"{name}_metadata.json"
                shutil.copy(metadata_path, saved_metadata_path)
                print(f"  Saved metadata: {saved_metadata_path}")
        
        # Create kernel info
        kernel_info = KernelBinaryInfo(
            kernel_name=name,
            kernel_hash=hash_val,
            binary_path=str(saved_binary_path or binary_path),
            ptx_path=str(saved_ptx_path or ptx_path) if ptx_path else None,
            metadata_path=str(saved_metadata_path or metadata_path),
            metadata=metadata,
            source_hash=hashlib.sha256(str(metadata_group).encode()).hexdigest()[:16]
        )
        
        self.compiled_kernels[hash_val] = kernel_info
        
        # Store function pointer to hash mapping for launch-time lookup
        # The function parameter is already a CUDA function pointer (CUfunction, an integer)
        # Store this integer directly as the key
        self.function_to_hash[function] = hash_val
        
        # Generate standalone launcher
        if self.save_binaries:
            self._generate_launcher(kernel_info)
    
    def _serialize_tensor(self, tensor, launch_id: int, arg_idx: int, kernel_name: str) -> Tuple[str, Dict[str, Any]]:
        """Serialize a PyTorch tensor to a binary file with metadata"""
        if not HAS_TORCH or not isinstance(tensor, torch.Tensor):
            return None, {}
        
        # Move tensor to CPU for serialization
        cpu_tensor = tensor.detach().cpu()
        np_array = cpu_tensor.numpy()
        
        # Create filename
        tensor_filename = f"{kernel_name}_launch{launch_id}_arg{arg_idx}.bin"
        tensor_path = self.data_dir / tensor_filename
        
        # Save as raw binary
        np_array.tofile(str(tensor_path))
        
        # Create metadata
        metadata = {
            'shape': list(np_array.shape),
            'dtype': str(np_array.dtype),
            'size_bytes': np_array.nbytes,
            'device': str(tensor.device),
            'requires_grad': tensor.requires_grad
        }
        
        print(f"    Saved tensor arg[{arg_idx}]: shape={metadata['shape']}, dtype={metadata['dtype']}, size={metadata['size_bytes']} bytes")
        
        return str(tensor_path), metadata
    
    def _capture_arguments(self, args: tuple, kernel_name: str, launch_id: int) -> List[ArgumentInfo]:
        """Capture and serialize kernel arguments
        
        NOTE: This captures arguments at the Python call level. Triton may optimize away
        some arguments during compilation (e.g., redundant strides for contiguous tensors).
        The actual PTX kernel may have fewer parameters than captured here.
        Use the PTX signature to determine the actual kernel parameters for harness generation.
        """
        arg_infos = []
        
        # Also store snapshots of tensor arguments for later comparison
        if HAS_TORCH:
            self.pending_args_snapshots = []
        
        for idx, arg in enumerate(args):
            arg_info = ArgumentInfo(index=idx, name=f"arg{idx}", arg_type='unknown')
            
            if HAS_TORCH and isinstance(arg, torch.Tensor):
                # PyTorch tensor
                data_file, metadata = self._serialize_tensor(arg, launch_id, idx, kernel_name)
                arg_info.arg_type = 'tensor'
                arg_info.dtype = metadata.get('dtype', 'unknown')
                arg_info.shape = tuple(metadata.get('shape', []))
                arg_info.data_file = data_file
                arg_info.size_bytes = metadata.get('size_bytes', 0)
                
                # Store a snapshot for comparison after execution
                # Clone to capture current state before kernel modifies it
                snapshot = arg.detach().clone()
                self.pending_args_snapshots.append(snapshot)
                
            elif isinstance(arg, (int, float, bool)):
                # Scalar value
                arg_info.arg_type = 'scalar'
                arg_info.value = arg
                arg_info.dtype = type(arg).__name__
                arg_info.size_bytes = sys.getsizeof(arg)
                print(f"    Captured scalar arg[{idx}]: {arg_info.dtype} = {arg}")
                
            elif isinstance(arg, (list, tuple)):
                # Could be a constexpr or shape
                arg_info.arg_type = 'constexpr'
                arg_info.value = arg
                arg_info.dtype = 'list/tuple'
                print(f"    Captured constexpr arg[{idx}]: {arg}")
                
            else:
                # Unknown type
                arg_info.arg_type = 'unknown'
                arg_info.dtype = type(arg).__name__
                print(f"    [WARNING] Unknown arg[{idx}] type: {type(arg)}")
            
            arg_infos.append(arg_info)
        
        return arg_infos
    
    def _on_launch_exit(self, launch_metadata):
        """Called after kernel launch completes"""
        if not self.enabled:
            return  # Tracking disabled
        
        # Extract metadata
        metadata = launch_metadata.get() if hasattr(launch_metadata, 'get') else launch_metadata
        
        if not isinstance(metadata, dict):
            return
        
        kernel_name = metadata.get('name', 'unknown')
        
        # Find the corresponding launch info from launch_enter
        if not self.kernel_launches:
            return
        
        # Get the most recent launch (should be the one that just finished)
        launch_info = self.kernel_launches[-1]
        
        if launch_info.kernel_name != kernel_name:
            print(f"[WARNING] Kernel name mismatch in launch_exit: expected {launch_info.kernel_name}, got {kernel_name}")
            return
        
        # Capture output tensors by comparing with pre-launch snapshots
        if self.capture_args and self.pending_args is not None and self.pending_args_snapshots is not None and HAS_TORCH:
            print(f"\n[TritonKernelTracker] Detecting and capturing outputs for launch #{launch_info.launch_id}")
            
            # Ensure GPU operations are complete before checking
            torch.cuda.synchronize()
            
            snapshot_idx = 0
            for idx, arg in enumerate(self.pending_args):
                if isinstance(arg, torch.Tensor):
                    if snapshot_idx >= len(self.pending_args_snapshots):
                        print(f"    [WARNING] Snapshot index out of bounds for arg[{idx}]")
                        continue
                    
                    arg_info = launch_info.args_info[idx] if idx < len(launch_info.args_info) else None
                    if not arg_info or arg_info.arg_type != 'tensor':
                        snapshot_idx += 1
                        continue
                    
                    # Compare current tensor with pre-launch snapshot
                    pre_snapshot = self.pending_args_snapshots[snapshot_idx]
                    snapshot_idx += 1
                    
                    # Check if tensor was modified (is an output)
                    is_output = False
                    try:
                        # Use torch.equal for exact comparison, or allclose for floating point
                        if arg.dtype.is_floating_point:
                            is_output = not torch.allclose(arg, pre_snapshot, rtol=1e-7, atol=1e-7)
                        else:
                            is_output = not torch.equal(arg, pre_snapshot)
                    except Exception as e:
                        print(f"    [WARNING] Could not compare arg[{idx}]: {e}")
                        # Assume it's an output if we can't compare
                        is_output = True
                    
                    if is_output:
                        # Save the output tensor state
                        output_filename = f"{kernel_name}_launch{launch_info.launch_id}_arg{idx}_output.bin"
                        output_path = self.data_dir / output_filename
                        
                        # Save output tensor
                        cpu_tensor = arg.detach().cpu()
                        np_array = cpu_tensor.numpy()
                        np_array.tofile(str(output_path))
                        
                        # Update arg_info with output file
                        arg_info.output_file = str(output_path)
                        
                        print(f"    Saved OUTPUT arg[{idx}]: shape={list(np_array.shape)}, dtype={np_array.dtype}, size={np_array.nbytes} bytes")
                    else:
                        print(f"    Skipped arg[{idx}]: unchanged (input-only)")
            
            print(f"  Output capture complete for launch #{launch_info.launch_id}")
            
            # Clear snapshots
            self.pending_args_snapshots = None
            
            # Regenerate harness now that we have output files
            if self.save_binaries:
                # Use the kernel_hash from launch_info (set reliably via function pointer in launch_enter)
                matching_kernel = self.compiled_kernels.get(launch_info.kernel_hash)
                if matching_kernel:
                    print(f"  Generating harness with validation code...")
                    self._generate_launch_specific_harness(kernel_info=matching_kernel,
                                                           launch_info=launch_info)

        # Clear pending args after processing
        self.pending_args = None
        self.pending_grid = None
    
    def _on_launch_enter(self, launch_metadata):
        """Called before kernel launch"""
        if not self.enabled:
            return  # Tracking disabled
        
        self.launch_counter += 1
        
        # Extract metadata (LazyDict)
        metadata = launch_metadata.get() if hasattr(launch_metadata, 'get') else launch_metadata
        
        if not isinstance(metadata, dict):
            return
        
        kernel_name = metadata.get('name', 'unknown')
        function = metadata.get('function', None)

        # Use function pointer to hash mapping (the only reliable method)
        # The function is a CUDA function pointer (integer) stored during kernel_load
        if function is None:
            raise RuntimeError(
                f"Cannot match kernel '{kernel_name}': function pointer is None. "
                f"This indicates a Triton runtime issue - launch_metadata should contain 'function'."
            )

        if function not in self.function_to_hash:
            raise RuntimeError(
                f"Cannot match kernel '{kernel_name}': function pointer {function} not in mapping. "
                f"This indicates kernel_load hook was not called before launch. "
                f"Available mappings: {list(self.function_to_hash.keys())}"
            )

        kernel_hash = self.function_to_hash[function]
        if kernel_hash not in self.compiled_kernels:
            raise RuntimeError(
                f"Cannot match kernel '{kernel_name}': hash {kernel_hash} not in compiled_kernels. "
                f"This should not happen - kernel_load should have registered it."
            )

        matching_kernel = self.compiled_kernels[kernel_hash]
        print(f"[TritonKernelTracker] Launch #{self.launch_counter}: {kernel_name} (hash: {kernel_hash[:8]}...)")
        
        # Extract launch parameters
        # Get grid from pending_grid (already evaluated)
        # Metadata does NOT contain grid size by default, so we must capture it ourselves
        grid = self.pending_grid
        if grid is None:
            raise RuntimeError(
                f"Grid size not captured for kernel {kernel_name}! "
                f"This should not happen - the grid should have been captured in JITFunction.run patch. "
                f"Available metadata keys: {list(metadata.keys())}"
            )
        
        # Normalize grid to 3-tuple (should already be done, but just in case)
        grid = self._normalize_grid(grid)

        meta_dict = matching_kernel.metadata
        num_warps = meta_dict.get('num_warps', 1)
        num_ctas = meta_dict.get('num_ctas', 1)
        shared_memory = meta_dict.get('shared', 0)
        
        # Calculate block dimensions
        warp_size = 32  # Assume CUDA
        block_x = warp_size * num_warps
        block = (block_x, 1, 1)
        
        # Capture arguments if available
        args_info = []
        if self.capture_args and self.pending_args is not None:
            print(f"  Capturing {len(self.pending_args)} arguments...")
            args_info = self._capture_arguments(self.pending_args, kernel_name, self.launch_counter)
        
        # Extract scratch memory metadata (per-CTA sizes)
        per_cta_global_scratch = meta_dict.get('global_scratch_size', 0)
        global_scratch_align = meta_dict.get('global_scratch_align', 1)
        per_cta_profile_scratch = meta_dict.get('profile_scratch_size', 0)
        profile_scratch_align = meta_dict.get('profile_scratch_align', 1)
        
        # Calculate total scratch sizes as Triton does:
        # At runtime CudaLauncher.__call__ multiplies the per-CTA size by gridX*gridY*gridZ * num_ctas
        total_ctas = grid[0] * grid[1] * grid[2] * num_ctas
        global_scratch_size = per_cta_global_scratch * total_ctas
        profile_scratch_size = per_cta_profile_scratch * total_ctas
        
        # Create launch info
        launch_info = KernelLaunchInfo(
            timestamp=datetime.now().isoformat(),
            kernel_name=kernel_name,
            kernel_hash=matching_kernel.kernel_hash,
            grid=grid,
            block=block,
            shared_memory=shared_memory,
            num_warps=num_warps,
            num_ctas=num_ctas,
            args_info=args_info,
            launch_id=self.launch_counter,
            stream=None,
            global_scratch_size=global_scratch_size,
            global_scratch_align=global_scratch_align,
            profile_scratch_size=profile_scratch_size,
            profile_scratch_align=profile_scratch_align
        )
        
        self.kernel_launches.append(launch_info)
        
        print(f"\n[TritonKernelTracker] Launch #{self.launch_counter}: {kernel_name}")
        print(f"  Grid: {grid}, Block: {block}")
        print(f"  Shared memory: {shared_memory} bytes")
        print(f"  Num warps: {num_warps}, Num CTAs: {num_ctas}")
    
    def _generate_helper_functions(self):
        """Generate reusable helper functions for data loading and validation"""
        return """
// Helper function to load tensor from binary file
void* load_tensor_arg(const char* exe_path, const char* rel_path, size_t size, int arg_idx,
                      const char* dtype, const char* shape) {
    // Handle zero-size tensors (e.g., dummy bias when HAS_BIAS=False)
    if (size == 0) {
        printf("  Loaded arg[%d]: tensor shape=%s, dtype=%s, size=0 bytes (zero-size, using NULL)\\n", arg_idx, shape, dtype);
        return NULL;
    }

    void* d_ptr;
    cudaMalloc(&d_ptr, size);

    char data_path[2048];
    snprintf(data_path, sizeof(data_path), "%s/%s", exe_path, rel_path);
    FILE* fp = fopen(data_path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s\\n", data_path);
        return NULL;
    }

    void* h_ptr = malloc(size);
    fread(h_ptr, 1, size, fp);
    fclose(fp);
    cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
    free(h_ptr);

    printf("  Loaded arg[%d]: tensor shape=%s, dtype=%s, size=%zu bytes\\n", arg_idx, shape, dtype, size);
    return d_ptr;
}

// Helper to convert FP16 to float for comparison
float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    
    if (exponent == 0) {
        if (mantissa == 0) {
            return sign ? -0.0f : 0.0f;
        } else {
            float val = mantissa / 1024.0f / 16384.0f;
            return sign ? -val : val;
        }
    } else if (exponent == 31) {
        if (mantissa == 0) {
            return sign ? -INFINITY : INFINITY;
        } else {
            return NAN;
        }
    } else {
        // Cast exponent to int before subtraction to avoid unsigned underflow
        float val = (1.0f + mantissa / 1024.0f) * powf(2.0f, (float)((int)exponent - 15));
        return sign ? -val : val;
    }
}

// Helper to convert BF16 to float for comparison
float bf16_to_fp32(uint16_t h) {
    uint32_t f32_bits = ((uint32_t)h) << 16;
    float result;
    memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

// Helper function to validate output tensor with dtype support
int validate_tensor_output(void* d_actual, const char* exe_path, const char* rel_expected_path, 
                           size_t size, int arg_idx, const char* dtype) {
    char expected_path[2048];
    snprintf(expected_path, sizeof(expected_path), "%s/%s", exe_path, rel_expected_path);
    FILE* fp = fopen(expected_path, "rb");
    if (!fp) {
        printf("  No expected output file for arg[%d], skipping validation\\n", arg_idx);
        return 0;
    }
    
    void* h_expected = malloc(size);
    void* h_actual = malloc(size);
    fread(h_expected, 1, size, fp);
    fclose(fp);
    cudaMemcpy(h_actual, d_actual, size, cudaMemcpyDeviceToHost);
    
    // Determine data type
    int is_fp16 = (strstr(dtype, "float16") != NULL || strstr(dtype, "fp16") != NULL);
    int is_bf16 = (strstr(dtype, "bfloat16") != NULL || strstr(dtype, "bf16") != NULL);
    int is_fp32 = (strstr(dtype, "float32") != NULL || strstr(dtype, "fp32") != NULL);
    int is_fp64 = (strstr(dtype, "float64") != NULL || strstr(dtype, "fp64") != NULL);
    int is_int8 = (strstr(dtype, "int8") != NULL);
    int is_int16 = (strstr(dtype, "int16") != NULL);
    int is_int32 = (strstr(dtype, "int32") != NULL);
    int is_int64 = (strstr(dtype, "int64") != NULL);
    int is_uint8 = (strstr(dtype, "uint8") != NULL);
    int is_uint16 = (strstr(dtype, "uint16") != NULL);
    int is_uint32 = (strstr(dtype, "uint32") != NULL);
    int is_uint64 = (strstr(dtype, "uint64") != NULL);
    
    // Determine element size
    size_t elem_size = 4;  // default
    if (is_fp16 || is_bf16 || is_int16 || is_uint16) elem_size = 2;
    else if (is_fp64 || is_int64 || is_uint64) elem_size = 8;
    else if (is_int8 || is_uint8) elem_size = 1;
    else if (is_fp32 || is_int32 || is_uint32) elem_size = 4;
    
    size_t num_elements = size / elem_size;
    int mismatches = 0;
    
    printf("  Validating arg[%d]: dtype=%s, elem_size=%zu, num_elements=%zu\\n", 
           arg_idx, dtype, elem_size, num_elements);
    
    // Floating-point types
    if (is_fp16) {
        uint16_t* expected_data = (uint16_t*)h_expected;
        uint16_t* actual_data = (uint16_t*)h_actual;
        float rel_tolerance = 1e-3f; // relative tolerance: 0.1%
        float abs_tolerance = 1e-2f; // absolute tolerance: 0.01

        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            float exp_f = fp16_to_fp32(expected_data[i]);
            float act_f = fp16_to_fp32(actual_data[i]);
            float diff = fabsf(exp_f - act_f);
            float max_abs = fmaxf(fabsf(exp_f), fabsf(act_f));
            float rel_err = (max_abs > 0) ? (diff / max_abs) : 0;
            // Pass if relative error <= 0.1% OR absolute error <= 0.01
            int pass = (rel_err <= rel_tolerance) || (diff <= abs_tolerance);

            if (!pass) {
                if (mismatches == 0) {
                    printf("\\n  Validation FAILED for arg[%d]:\\n", arg_idx);
                }
                printf("    Element %zu: expected=%.6f, actual=%.6f, diff=%.6e, rel_err=%.6e\\n",
                       i, exp_f, act_f, diff, rel_err);
                mismatches++;
            }
        }
    } else if (is_bf16) {
        uint16_t* expected_data = (uint16_t*)h_expected;
        uint16_t* actual_data = (uint16_t*)h_actual;
        float rel_tolerance = 1e-3f; // relative tolerance: 0.1%
        float abs_tolerance = 1e-2f; // absolute tolerance: 0.01

        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            float exp_f = bf16_to_fp32(expected_data[i]);
            float act_f = bf16_to_fp32(actual_data[i]);
            float diff = fabsf(exp_f - act_f);
            float max_abs = fmaxf(fabsf(exp_f), fabsf(act_f));
            float rel_err = (max_abs > 0) ? (diff / max_abs) : 0;
            // Pass if relative error <= 0.1% OR absolute error <= 0.01
            int pass = (rel_err <= rel_tolerance) || (diff <= abs_tolerance);

            if (!pass) {
                if (mismatches == 0) {
                    printf("\\n  Validation FAILED for arg[%d]:\\n", arg_idx);
                }
                printf("    Element %zu: expected=%.6f, actual=%.6f, diff=%.6e, rel_err=%.6e\\n",
                       i, exp_f, act_f, diff, rel_err);
                mismatches++;
            }
        }
    } else if (is_fp32) {
        float* expected_data = (float*)h_expected;
        float* actual_data = (float*)h_actual;
        float rel_tolerance = 1e-3f; // relative tolerance: 0.1%
        float abs_tolerance = 1e-2f; // absolute tolerance: 0.01

        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            float exp_f = expected_data[i];
            float act_f = actual_data[i];
            float diff = fabsf(exp_f - act_f);
            float max_abs = fmaxf(fabsf(exp_f), fabsf(act_f));
            float rel_err = (max_abs > 0) ? (diff / max_abs) : 0;
            // Pass if relative error <= 0.1% OR absolute error <= 0.01
            int pass = (rel_err <= rel_tolerance) || (diff <= abs_tolerance);

            if (!pass) {
                if (mismatches == 0) {
                    printf("\\n  Validation FAILED for arg[%d]:\\n", arg_idx);
                }
                printf("    Element %zu: expected=%.6f, actual=%.6f, diff=%.6e, rel_err=%.6e\\n",
                       i, exp_f, act_f, diff, rel_err);
                mismatches++;
            }
        }
    } else if (is_fp64) {
        double* expected_data = (double*)h_expected;
        double* actual_data = (double*)h_actual;
        double rel_tolerance = 1e-6; // relative tolerance: 0.0001%
        double abs_tolerance = 1e-12; // absolute tolerance for values near zero

        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            double exp_d = expected_data[i];
            double act_d = actual_data[i];
            double diff = fabs(exp_d - act_d);
            double max_abs = fmax(fabs(exp_d), fabs(act_d));
            int mismatch = (max_abs > abs_tolerance) ? (diff / max_abs > rel_tolerance) : (diff > abs_tolerance);

            if (mismatch) {
                if (mismatches == 0) {
                    printf("\\n  Validation FAILED for arg[%d]:\\n", arg_idx);
                }
                double rel_err = (max_abs > 0) ? (diff / max_abs) : 0;
                printf("    Element %zu: expected=%.15f, actual=%.15f, diff=%.6e, rel_err=%.6e\\n",
                       i, exp_d, act_d, diff, rel_err);
                mismatches++;
            }
        }
    } 
    // Integer types - exact comparison
    else if (is_int8) {
        int8_t* expected_data = (int8_t*)h_expected;
        int8_t* actual_data = (int8_t*)h_actual;
        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            if (expected_data[i] != actual_data[i]) {
                if (mismatches == 0) {
                    printf("\\n  Validation FAILED for arg[%d]:\\n", arg_idx);
                }
                printf("    Element %zu: expected=%d, actual=%d\\n", 
                       i, expected_data[i], actual_data[i]);
                mismatches++;
            }
        }
    } else if (is_uint8) {
        uint8_t* expected_data = (uint8_t*)h_expected;
        uint8_t* actual_data = (uint8_t*)h_actual;
        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            if (expected_data[i] != actual_data[i]) {
                if (mismatches == 0) {
                    printf("\\n  Validation FAILED for arg[%d]:\\n", arg_idx);
                }
                printf("    Element %zu: expected=%u, actual=%u\\n", 
                       i, expected_data[i], actual_data[i]);
                mismatches++;
            }
        }
    } else if (is_int32) {
        int32_t* expected_data = (int32_t*)h_expected;
        int32_t* actual_data = (int32_t*)h_actual;
        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            if (expected_data[i] != actual_data[i]) {
                if (mismatches == 0) {
                    printf("\\n  Validation FAILED for arg[%d]:\\n", arg_idx);
                }
                printf("    Element %zu: expected=%d, actual=%d\\n", 
                       i, expected_data[i], actual_data[i]);
                mismatches++;
            }
        }
    } else if (is_int16) {
        int16_t* expected_data = (int16_t*)h_expected;
        int16_t* actual_data = (int16_t*)h_actual;
        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            if (expected_data[i] != actual_data[i]) {
                if (mismatches == 0) {
                    printf("\\n  Validation FAILED for arg[%d]:\\n", arg_idx);
                }
                printf("    Element %zu: expected=%d, actual=%d\\n",
                       i, expected_data[i], actual_data[i]);
                mismatches++;
            }
        }
    } else if (is_uint16) {
        uint16_t* expected_data = (uint16_t*)h_expected;
        uint16_t* actual_data = (uint16_t*)h_actual;
        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            if (expected_data[i] != actual_data[i]) {
                if (mismatches == 0) {
                    printf("\\n  Validation FAILED for arg[%d]:\\n", arg_idx);
                }
                printf("    Element %zu: expected=%u, actual=%u\\n",
                       i, expected_data[i], actual_data[i]);
                mismatches++;
            }
        }
    } else if (is_int64) {
        int64_t* expected_data = (int64_t*)h_expected;
        int64_t* actual_data = (int64_t*)h_actual;
        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            if (expected_data[i] != actual_data[i]) {
                if (mismatches == 0) {
                    printf("\\n  Validation FAILED for arg[%d]:\\n", arg_idx);
                }
                printf("    Element %zu: expected=%lld, actual=%lld\\n",
                       i, (long long)expected_data[i], (long long)actual_data[i]);
                mismatches++;
            }
        }
    } else if (is_uint32) {
        uint32_t* expected_data = (uint32_t*)h_expected;
        uint32_t* actual_data = (uint32_t*)h_actual;
        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            if (expected_data[i] != actual_data[i]) {
                if (mismatches == 0) {
                    printf("\\n  Validation FAILED for arg[%d]:\\n", arg_idx);
                }
                printf("    Element %zu: expected=%u, actual=%u\\n",
                       i, expected_data[i], actual_data[i]);
                mismatches++;
            }
        }
    } else if (is_uint64) {
        uint64_t* expected_data = (uint64_t*)h_expected;
        uint64_t* actual_data = (uint64_t*)h_actual;
        for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
            if (expected_data[i] != actual_data[i]) {
                if (mismatches == 0) {
                    printf("\\n  Validation FAILED for arg[%d]:\\n", arg_idx);
                }
                printf("    Element %zu: expected=%llu, actual=%llu\\n",
                       i, (unsigned long long)expected_data[i], (unsigned long long)actual_data[i]);
                mismatches++;
            }
        }
    } else {
        // Unsupported dtype - report error
        printf("\\n  ERROR: Unsupported dtype '%s' for validation of arg[%d]\\n", dtype, arg_idx);
        printf("  Supported types: float16, bfloat16, float32, float64, int8/16/32/64, uint8/16/32/64\\n");
        free(h_expected);
        free(h_actual);
        return -1;  // Return error code
    }
    
    if (mismatches == 0) {
        printf("  Validation PASSED for arg[%d]: all %zu elements match\\n", 
               arg_idx, num_elements);
    } else {
        printf("  Total mismatches for arg[%d]: %d (showing first 10)\\n", arg_idx, mismatches);
    }
    
    free(h_expected);
    free(h_actual);
    return (mismatches == 0) ? 0 : 1;
}
"""


    def _parse_ptx_signature(self, ptx_path: Path, kernel_name: str) -> int:
        """Parse PTX file to count actual kernel parameters
        
        Returns: Number of user parameters (excluding Triton runtime scratch pointers)
        """
        if not ptx_path.exists():
            return -1
        
        try:
            with open(ptx_path, 'r') as f:
                content = f.read()
            
            # Find the kernel entry
            import re
            pattern = rf'\.visible\s+\.entry\s+{re.escape(kernel_name)}\s*\('
            match = re.search(pattern, content)
            if not match:
                return -1
            
            # Count parameters (lines with .param until we hit closing paren)
            start_pos = match.end()
            closing_paren = content.find(')', start_pos)
            if closing_paren == -1:
                return -1
            
            params_section = content[start_pos:closing_paren]
            param_count = params_section.count('.param')
            
            # Triton always adds 2 scratch pointers at the end, so user params = total - 2
            user_param_count = max(0, param_count - 2)
            
            return user_param_count
        except Exception as e:
            print(f"    [WARNING] Could not parse PTX signature: {e}")
            return -1
    
    def _check_dynamic_shared_memory(self, ptx_path: Path) -> bool:
        """Check if PTX uses dynamic shared memory
        
        Returns: True if the PTX declares external shared memory (dynamic allocation)
        """
        if not ptx_path.exists():
            return False
        
        try:
            with open(ptx_path, 'r') as f:
                content = f.read()
            
            # Look for .extern .shared declaration (indicates dynamic shared memory)
            import re
            # Pattern matches: .extern .shared .align N .b8 name[];
            pattern = r'\.extern\s+\.shared\s+.*\[\s*\]'
            return bool(re.search(pattern, content))
        except Exception as e:
            print(f"    [WARNING] Could not check for dynamic shared memory: {e}")
            return False

    def _parse_ptx_target(self, ptx_path: Path) -> str:
        if not ptx_path.exists():
            return "sm_120a"
        try:
            import re
            content = ptx_path.read_text()
            match = re.search(r'^\s*\.target\s+([A-Za-z0-9_]+)', content, re.MULTILINE)
            if match:
                return match.group(1)
        except Exception:
            pass
        return "sm_120a"

    def _parse_ptx_register_fallback(self, ptx_path: Path) -> int:
        """Estimate register count from PTX if cuobjdump resource usage is unavailable."""
        if not ptx_path.exists():
            return 0
        try:
            import re
            content = ptx_path.read_text()
            total = 0
            for reg_type, count in re.findall(r'\.reg\s+\.([a-z0-9]+)\s+%[A-Za-z_][A-Za-z0-9_]*<(\d+)>', content):
                count_i = int(count)
                if reg_type == "b64":
                    total += count_i * 2
                elif reg_type != "pred":
                    total += count_i
            return total
        except Exception:
            return 0

    def _generate_ptxinfo_sidecar(self, cubin_path: Path, ptx_path: Path,
                                  output_path: Path, kernel_name: str) -> None:
        """Generate a ptxas-style resource sidecar from Triton's CUBIN.

        GPGPU-Sim normally shells out to ptxas -v to get register/shared-memory
        usage. New Triton TMA PTX can be newer than the system ptxas, while the
        CUBIN already contains the resource usage. This sidecar lets the
        simulator consume the same ptxinfo format without reassembling PTX.
        """
        import re
        import subprocess

        arch = self._parse_ptx_target(ptx_path)
        regs = self._parse_ptx_register_fallback(ptx_path)
        stack = 0
        shared = 0
        local = 0
        cmem0 = 0
        gmem = 0

        try:
            result = subprocess.run(
                ["cuobjdump", "--dump-resource-usage", str(cubin_path)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            output = result.stdout
            global_match = re.search(r'\bGLOBAL:(\d+)', output)
            if global_match:
                gmem = int(global_match.group(1))

            func_match = re.search(
                rf'Function\s+{re.escape(kernel_name)}:\s*\n\s*'
                r'REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)'
                r'\s+CONSTANT\[0\]:(\d+)',
                output,
            )
            if func_match:
                regs = int(func_match.group(1))
                stack = int(func_match.group(2))
                shared = int(func_match.group(3))
                local = int(func_match.group(4))
                cmem0 = int(func_match.group(5))
        except Exception as e:
            print(f"  Warning: cuobjdump resource usage failed, using PTX register estimate: {e}")

        output_path.write_text(
            f"ptxas info    : {gmem} bytes gmem\n"
            f"ptxas info    : Compiling entry function '{kernel_name}' for '{arch}'\n"
            f"ptxas info    : Function properties for {kernel_name}\n"
            f"    {stack} bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
            f"ptxas info    : Used {regs} registers, {shared} bytes smem, "
            f"{cmem0} bytes cmem[0], {local} bytes lmem\n"
        )
        print(f"  Generated ptxinfo sidecar: {output_path}")
    
    def _generate_scratch_allocation_code(self, launch_info: KernelLaunchInfo) -> str:
        """Generate scratch buffer allocation code based on metadata

        Only generates allocation code if size > 0, avoiding unnecessary conditionals in C code.
        Note: Sizes are already calculated as total = per_cta_size * gridX*gridY*gridZ * num_ctas
        """
        code_lines = []
        total_ctas = launch_info.grid[0] * launch_info.grid[1] * launch_info.grid[2] * launch_info.num_ctas

        if launch_info.global_scratch_size > 0:
            code_lines.append(f"""
    // Total size = per_cta_size * grid_size * num_ctas = per_cta * {total_ctas}
    cudaMalloc(&global_scratch, {launch_info.global_scratch_size});
    printf("  Allocated global_scratch: %zu bytes (alignment: %zu)\\n",
           (size_t){launch_info.global_scratch_size}, (size_t){launch_info.global_scratch_align});""")

        if launch_info.profile_scratch_size > 0:
            code_lines.append(f"""
    // Total size = per_cta_size * grid_size * num_ctas = per_cta * {total_ctas}
    cudaMalloc(&profile_scratch, {launch_info.profile_scratch_size});
    printf("  Allocated profile_scratch: %zu bytes (alignment: %zu)\\n",
           (size_t){launch_info.profile_scratch_size}, (size_t){launch_info.profile_scratch_align});""")

        return ''.join(code_lines)
    
    def _generate_scratch_cleanup_code(self, launch_info: KernelLaunchInfo) -> str:
        """Generate scratch buffer cleanup code based on metadata
        
        Only generates cleanup code if size > 0, avoiding unnecessary conditionals in C code.
        """
        code_lines = []
        
        if launch_info.global_scratch_size > 0:
            code_lines.append("""    
    cudaFree(global_scratch);
    printf("  Freed global_scratch\\n");""")
        
        if launch_info.profile_scratch_size > 0:
            code_lines.append("""    
    cudaFree(profile_scratch);
    printf("  Freed profile_scratch\\n");""")
        
        return ''.join(code_lines)
    
    def _generate_launch_specific_harness(self, kernel_info: KernelBinaryInfo, launch_info: KernelLaunchInfo):
        """Generate a harness for a specific launch with actual argument data"""
        kernel_name = kernel_info.kernel_name
        launch_id = launch_info.launch_id
        
        # Read PTX to check if we can generate the harness. PTX remains useful
        # for signature parsing even when the executable fatbin is built from
        # Triton's CUBIN.
        binary_path = Path(kernel_info.binary_path)
        ptx_path = binary_path.parent / f"{kernel_name}.ptx"
        cubin_path = binary_path if binary_path.suffix == ".cubin" else None
        
        ptx_string = None
        if ptx_path.exists():
            try:
                with open(ptx_path, 'r') as f:
                    ptx_content = f.read()
                ptx_string = "exists"  # Just need to know it exists
            except Exception as e:
                print(f"  Warning: Could not read PTX: {e}")
        
        # Check for argument count mismatch
        ptx_param_count = self._parse_ptx_signature(ptx_path, kernel_name)
        captured_arg_count = len(launch_info.args_info)
        
        if ptx_param_count >= 0 and ptx_param_count != captured_arg_count:
            error_msg = (
                f"\n"
                f"Argument count mismatch for kernel '{kernel_name}':\n"
                f"  Captured from Python: {captured_arg_count} arguments\n"
                f"  PTX kernel signature: {ptx_param_count} user parameters (+ 2 runtime scratch pointers)\n"
                f"  Mismatch: {captured_arg_count - ptx_param_count} arguments were optimized away by Triton\n"
                f"\n"
                f"  Triton optimizes away redundant parameters during compilation.\n"
                f"  Cannot automatically generate correct harness without knowing which parameters were kept.\n"
                f"  Please inspect the PTX file manually: {ptx_path}\n\n"
                f"\n"
            )
            raise RuntimeError(error_msg)
        
        # Make binary path relative to launchers directory
        launchers_dir = self.launchers_dir.resolve()
        
        # Build argument loading code using helper functions
        arg_declarations = []
        arg_loading_calls = []
        arg_pointers = []
        validation_calls = []
        cleanup_code = []
        
        for arg_info in launch_info.args_info:
            idx = arg_info.index
            
            if arg_info.arg_type == 'tensor' and arg_info.data_file:
                # Make data file path relative to launchers directory
                data_file_path = Path(arg_info.data_file)
                if not data_file_path.is_absolute():
                    data_file_path = (Path.cwd() / data_file_path).resolve()
                try:
                    relative_data_path = os.path.relpath(data_file_path, launchers_dir)
                except Exception as e:
                    relative_data_path = arg_info.data_file
                
                # Generate simplified code using helper function
                arg_declarations.append(f"    void* d_arg{idx};")
                arg_declarations.append(f"    size_t arg{idx}_size = {arg_info.size_bytes};")
                
                shape_str = str(list(arg_info.shape))
                arg_loading_calls.append(
                    f'    d_arg{idx} = load_tensor_arg(exe_path, "{relative_data_path}", '
                    f'arg{idx}_size, {idx}, "{arg_info.dtype}", "{shape_str}");'
                )
                # Zero-size tensors legitimately return NULL — only fail for non-zero sizes
                if arg_info.size_bytes > 0:
                    arg_loading_calls.append(f'    if (!d_arg{idx}) return 1;')
                
                arg_pointers.append(f"&d_arg{idx}")
                cleanup_code.append(f"    cudaFree(d_arg{idx});")
                
                # Add validation call if we have expected output
                if arg_info.output_file:
                    output_file_path = Path(arg_info.output_file)
                    if not output_file_path.is_absolute():
                        output_file_path = (Path.cwd() / output_file_path).resolve()
                    try:
                        relative_output_path = os.path.relpath(output_file_path, launchers_dir)
                    except:
                        relative_output_path = arg_info.output_file
                    
                    validation_calls.append(
                        f'    validate_tensor_output(d_arg{idx}, exe_path, "{relative_output_path}", '
                        f'arg{idx}_size, {idx}, "{arg_info.dtype}");'
                    )
                
            elif arg_info.arg_type == 'scalar':
                # Generate code for scalar argument
                dtype_map = {
                    'int': 'int32_t',
                    'float': 'float',
                    'bool': 'uint8_t'
                }
                c_type = dtype_map.get(arg_info.dtype, 'int32_t')
                value = arg_info.value
                
                arg_declarations.append(f"    {c_type} arg{idx} = {value};")
                arg_loading_calls.append(f'    printf("  Set arg[{idx}]: {arg_info.dtype} = {value}\\n");')
                arg_pointers.append(f"&arg{idx}")
        
        # Generate harness with helper functions
        if ptx_string:
            fatbin_filename = f"{kernel_name}_launch{launch_id}_kernel.fatbin"
            helper_functions = self._generate_helper_functions()
            
            # Check if kernel uses dynamic shared memory
            uses_dynamic_smem = self._check_dynamic_shared_memory(ptx_path)
            
            # Generate shared memory configuration code if needed
            smem_config_code = ""
            if uses_dynamic_smem and launch_info.shared_memory > 0:
                smem_config_code = f"""
    // Configure shared memory if needed ({launch_info.shared_memory} bytes requires opt-in on some GPUs)
    int shared_mem_bytes = {launch_info.shared_memory};
    CUresult attr_result = cuFuncSetAttribute(
        kernel_func, 
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 
        shared_mem_bytes);
    if (attr_result != CUDA_SUCCESS) {{{{
      const char* errStr;
      cuGetErrorString(attr_result, &errStr);
      fprintf(stderr, "Warning: Failed to set shared memory size: %s\\n", errStr);
      fprintf(stderr, "Attempting to continue anyway...\\n");
    }}}} else {{{{
      printf("Configured dynamic shared memory: %d bytes\\n", shared_mem_bytes);
    }}}}
    printf("\\n");
"""
            else:
                smem_config_code = "    printf(\"\\n\");\n"

            cpp_code = f"""
// Standalone harness for Triton kernel: {kernel_name}
// Launch ID: {launch_id}
// Generated by TritonKernelTracker with captured arguments
// Kernel hash: {kernel_info.kernel_hash}

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>

{helper_functions}

int main(int argc, char** argv) {{{{
    printf("=== Standalone Harness for Triton Kernel ===\\n");
    printf("Kernel: {kernel_name}\\n");
    printf("Launch ID: {launch_id}\\n");
    printf("Grid: {launch_info.grid}\\n");
    printf("Block: {launch_info.block}\\n");
    printf("Shared memory: {launch_info.shared_memory} bytes\\n");
    printf("\\n");

    // Initialize CUDA
    cuInit(0);
    CUdevice device;
    CUcontext context;
    cuDeviceGet(&device, 0);
#if CUDA_VERSION >= 12050
    cuCtxCreate(&context, NULL, 0, device);
#else
    cuCtxCreate(&context, 0, device);
#endif

    // Get executable directory
    char exe_path[1024];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len == -1) {{{{
        fprintf(stderr, "Failed to get executable path\\n");
        return 1;
    }}}}
    exe_path[len] = '\\0';
    char* last_slash = strrchr(exe_path, '/');
    if (last_slash) *last_slash = '\\0';

    // Load module
    char fatbin_path[2048];
    snprintf(fatbin_path, sizeof(fatbin_path), "%s/{fatbin_filename}", exe_path);
    CUmodule module;
    CUresult load_result = cuModuleLoad(&module, fatbin_path);
    if (load_result != CUDA_SUCCESS) {{{{
        const char* errStr;
        cuGetErrorString(load_result, &errStr);
        fprintf(stderr, "Failed to load fatbin: %s\\n", errStr);
        return 1;
    }}}}
    printf("Loaded module from: %s\\n", fatbin_path);

    // Get kernel function
    CUfunction kernel_func;
    CUresult func_result = cuModuleGetFunction(&kernel_func, module, "{kernel_name}");
    if (func_result != CUDA_SUCCESS) {{{{
        const char* errStr;
        cuGetErrorString(func_result, &errStr);
        fprintf(stderr, "Failed to get function: %s\\n", errStr);
        return 1;
    }}}}
    printf("Got kernel function: {kernel_name}\\n");
{smem_config_code}
    // Initialize arguments
    printf("Initializing arguments:\\n");
{chr(10).join(arg_declarations)}

{chr(10).join(arg_loading_calls)}

    // Allocate Triton runtime scratch buffers based on metadata
    // Note: These pointers may differ from Triton's original allocation, but are functionally equivalent
    void* global_scratch = NULL;
    void* profile_scratch = NULL;
{self._generate_scratch_allocation_code(launch_info)}
    // Setup kernel arguments (user args + triton runtime args)
    void* args[] = {{ {', '.join(arg_pointers)}, &global_scratch, &profile_scratch }};

    // Launch kernel
    printf("\\nLaunching kernel...\\n");
    CUresult launch_result = cuLaunchKernel(
        kernel_func,
        {launch_info.grid[0]}, {launch_info.grid[1]}, {launch_info.grid[2]},
        {launch_info.block[0]}, {launch_info.block[1]}, {launch_info.block[2]},
        {launch_info.shared_memory}, 0, args, 0
    );
    if (launch_result != CUDA_SUCCESS) {{{{
        const char* errStr;
        cuGetErrorString(launch_result, &errStr);
        fprintf(stderr, "Kernel launch failed: %s\\n", errStr);
        return 1;
    }}}}

    cuCtxSynchronize();
    printf("Kernel execution completed successfully\\n");

    // Validate outputs
    printf("\\nValidating outputs...\\n");
{chr(10).join(validation_calls)}

    // Cleanup
    printf("\\nCleaning up...\\n");
{chr(10).join(cleanup_code)}
{self._generate_scratch_cleanup_code(launch_info)}
    cuModuleUnload(module);
    cuCtxDestroy(context);

    printf("Done!\\n");
    return 0;
}}}}
"""
        
        else:
            # Fallback: warn that PTX couldn't be embedded
            cpp_code = f"""
// WARNING: PTX embedding failed - this is a placeholder
#warning "PTX could not be embedded"
int main() {{
    printf("ERROR: PTX file could not be read for embedding\\n");
    return 1;
}}
"""
        
        # Save harness
        harness_filename = f"{kernel_name}_launch{launch_id}_harness.cu"
        harness_path = self.launchers_dir / harness_filename
        harness_path.write_text(cpp_code)
        print(f"  Generated harness with arguments: {harness_path}")
        
        # Save PTX/CUBIN files for linking with nvcc/fatbinary
        if ptx_string:
            ptx_filename = f"{kernel_name}_launch{launch_id}_kernel.ptx"
            ptx_file_path = self.launchers_dir / ptx_filename
            # Copy the original PTX content
            ptx_path_src = Path(kernel_info.binary_path).parent / f"{kernel_name}.ptx"
            if ptx_path_src.exists():
                import shutil
                shutil.copy(ptx_path_src, ptx_file_path)
                print(f"  Copied PTX for inspection: {ptx_file_path}")

        cubin_file_path = None
        if cubin_path and cubin_path.exists():
            cubin_filename = f"{kernel_name}_launch{launch_id}_kernel.cubin"
            cubin_file_path = self.launchers_dir / cubin_filename
            import shutil
            shutil.copy(cubin_path, cubin_file_path)
            print(f"  Copied CUBIN for linking: {cubin_file_path}")
            ptxinfo_path = self.launchers_dir / f"{kernel_name}_launch{launch_id}_kernel.ptxinfo"
            self._generate_ptxinfo_sidecar(
                cubin_file_path, ptx_path, ptxinfo_path, kernel_name)

        # Generate Makefile. If PTX was found we'll produce steps to build a fatbin.
        makefile_path = self.launchers_dir / f"{kernel_name}_launch{launch_id}_Makefile"
        target_name = f"{kernel_name}_launch{launch_id}"

        if ptx_string:
            ptx_filename = f"{kernel_name}_launch{launch_id}_kernel.ptx"
            fatbin_filename = f"{kernel_name}_launch{launch_id}_kernel.fatbin"
            cubin_filename = f"{kernel_name}_launch{launch_id}_kernel.cubin"
            has_cubin = bool(cubin_path and cubin_path.exists())
            fatbin_deps = "$(CUBIN_FILE) $(PTX_FILE)" if has_cubin else "$(PTX_FILE)"
            fatbin_images = (
                "--image=profile=$(ARCH),file=$(CUBIN_FILE) "
                "--image=profile=$(PTX_PROFILE),file=$(PTX_FILE)"
                if has_cubin else
                "--image=profile=$(PTX_PROFILE),file=$(PTX_FILE)"
            )
            fatbin_images_image3 = (
                "--image3=kind=elf,sm=$(ARCH_NUM),file=$(CUBIN_FILE) "
                "--image3=kind=ptx,sm=$(ARCH_NUM),file=$(PTX_FILE)"
                if has_cubin else
                "--image3=kind=ptx,sm=$(ARCH_NUM),file=$(PTX_FILE)"
            )

            makefile_content = f"""
# Makefile for {kernel_name} launch {launch_id}
# Packages Triton's generated binary into a fatbinary loaded at runtime.
# PTX is kept alongside it for inspection and trace tooling.

NVCC = nvcc
FATBINARY = fatbinary
NVCC_BIN := $(shell command -v $(NVCC))
CUDA_LIB_DIR := $(abspath $(dir $(NVCC_BIN))/../lib)
CUDART_FLAG := $(shell if [ -f "$(CUDA_LIB_DIR)/libcudart.so.13" ]; then echo "-l:libcudart.so.13"; else echo "-lcudart"; fi)
CUDA_FLAGS = -L$(CUDA_LIB_DIR) -Xlinker -rpath -Xlinker $(CUDA_LIB_DIR) $(CUDART_FLAG) -lcuda
TARGET = {target_name}
PTX_FILE = {ptx_filename}
CUBIN_FILE = {cubin_filename}
FATBIN_FILE = {fatbin_filename}
FATBIN_DEPS = {fatbin_deps}
FATBINARY_IMAGES_LEGACY = {fatbin_images}
FATBINARY_IMAGES_IMAGE3 = {fatbin_images_image3}

# Auto-detect architecture from PTX .target directive. CUBIN comes from the
# same Triton compile, so this profile also matches the CUBIN payload.
ARCH := $(shell grep '^\\.target' $(PTX_FILE) | head -1 | awk '{{print $$2}}')
ARCH_NUM := $(patsubst sm_%,%,$(ARCH))
PTX_PROFILE := $(patsubst sm_%,compute_%,$(ARCH))

all: $(TARGET) $(FATBIN_FILE)

# Package Triton's CUBIN when available. Falling back to PTX-only avoids ptxas,
# but may require a driver that understands the PTX version emitted by Triton.
$(FATBIN_FILE): $(FATBIN_DEPS)
\t@echo "Detected architecture: $(ARCH)"
\t@if $(FATBINARY) --help 2>&1 | grep -q -- '--image3'; then \\
\t\t$(FATBINARY) --create=$(FATBIN_FILE) $(FATBINARY_IMAGES_IMAGE3); \\
\telse \\
\t\t$(FATBINARY) --create=$(FATBIN_FILE) $(FATBINARY_IMAGES_LEGACY); \\
\tfi

# Compile harness (fatbin is loaded from file at runtime)
$(TARGET): {harness_filename}
\t$(NVCC) -arch=$(ARCH) -o $(TARGET) {harness_filename} $(CUDA_FLAGS)

clean:
\trm -f $(TARGET) $(FATBIN_FILE)

.PHONY: all clean
"""
        else:
            makefile_content = f"""
# Makefile for {kernel_name} launch {launch_id}

NVCC = nvcc
CUDA_FLAGS = -lcuda -lcudart
TARGET = {target_name}

all: $(TARGET)

$(TARGET): {harness_filename}
\t$(NVCC) -o $(TARGET) {harness_filename} $(CUDA_FLAGS)

clean:
\trm -f $(TARGET)

.PHONY: all clean
"""

        makefile_path.write_text(makefile_content)
        print(f"  Generated Makefile: {makefile_path}")
    
    def _generate_launcher(self, kernel_info: KernelBinaryInfo):
        """Generate standalone C++ launcher for the kernel"""
        kernel_name = kernel_info.kernel_name
        metadata = kernel_info.metadata
        
        # Extract kernel properties
        num_warps = metadata.get('num_warps', 1)
        shared_memory = metadata.get('shared', 0)
        
        # This generates a template launcher without specific arguments
        # The launch-specific harness will be generated when we have actual arguments
        
        # Generate C++ code
        cpp_code = f"""
// Standalone launcher for Triton kernel: {kernel_name}
// Generated by TritonKernelTracker
// Kernel hash: {kernel_info.kernel_hash}
// NOTE: This is a template. Use the launch-specific harness for actual argument data.

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call) \\
    do {{ \\
        CUresult err = call; \\
        if (err != CUDA_SUCCESS) {{ \\
            const char* errStr; \\
            cuGetErrorString(err, &errStr); \\
            fprintf(stderr, "CUDA error at %s:%d - %s\\n", \\
                    __FILE__, __LINE__, errStr); \\
            exit(1); \\
        }} \\
    }} while (0)

// Kernel metadata
#define KERNEL_NAME "{kernel_name}"
#define NUM_WARPS {num_warps}
#define WARP_SIZE 32
#define BLOCK_SIZE_X (NUM_WARPS * WARP_SIZE)
#define SHARED_MEMORY {shared_memory}

int main(int argc, char** argv) {{
    printf("Standalone launcher for Triton kernel: {kernel_name}\\n");
    printf("NOTE: This is a template launcher.\\n");
    printf("For actual argument data, use the launch-specific harness.\\n\\n");
    
    // Initialize CUDA
    CUDA_CHECK(cuInit(0));
    
    CUdevice device;
    CUcontext context;
    CUDA_CHECK(cuDeviceGet(&device, 0));
#if CUDA_VERSION >= 12050
    CUDA_CHECK(cuCtxCreate(&context, NULL, 0, device));
#else
    CUDA_CHECK(cuCtxCreate(&context, 0, device));
#endif
    
    // Load kernel binary
    const char* binary_path = "{kernel_info.binary_path}";
    CUmodule module;
    CUDA_CHECK(cuModuleLoad(&module, binary_path));
    printf("Loaded kernel from: %s\\n", binary_path);
    
    // Get kernel function
    CUfunction kernel;
    CUDA_CHECK(cuModuleGetFunction(&kernel, module, KERNEL_NAME));
    printf("Got kernel function: %s\\n\\n", KERNEL_NAME);
    printf("NOTE: This is a template. No arguments configured.\\n");
    printf("Check for launch-specific harness files for actual replays.\\n");
    
    cuModuleUnload(module);
    cuCtxDestroy(context);
    
    return 0;
}}
"""
        
        # Save launcher code
        launcher_path = self.launchers_dir / f"{kernel_name}_{kernel_info.kernel_hash[:8]}_template.cu"
        launcher_path.write_text(cpp_code)
        print(f"  Generated template launcher: {launcher_path}")
    
    def save_summary(self):
        """Save summary of all tracked kernels and launches"""
        
        # Convert launches to dict with proper serialization
        launches_dict = []
        for launch in self.kernel_launches:
            launch_dict = asdict(launch)
            # Convert ArgumentInfo objects to dicts
            launch_dict['args_info'] = [asdict(arg) for arg in launch.args_info]
            launches_dict.append(launch_dict)
        
        summary = {
            'tracking_session': {
                'timestamp': datetime.now().isoformat(),
                'output_dir': str(self.output_dir),
                'total_kernels_compiled': len(self.compiled_kernels),
                'total_launches': len(self.kernel_launches),
                'argument_capture_enabled': self.capture_args
            },
            'compiled_kernels': {
                hash_val: asdict(info) for hash_val, info in self.compiled_kernels.items()
            },
            'kernel_launches': launches_dict
        }
        
        summary_path = self.output_dir / "tracking_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[TritonKernelTracker] Summary saved to: {summary_path}")
        print(f"  Total kernels compiled: {len(self.compiled_kernels)}")
        print(f"  Total kernel launches: {len(self.kernel_launches)}")
        
        # Also save a human-readable report
        report_path = self.output_dir / "tracking_report.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Triton Kernel Tracking Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(f"Total kernels compiled: {len(self.compiled_kernels)}\n")
            f.write(f"Total kernel launches: {len(self.kernel_launches)}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("Compiled Kernels\n")
            f.write("=" * 80 + "\n\n")
            for hash_val, info in self.compiled_kernels.items():
                f.write(f"Kernel: {info.kernel_name}\n")
                f.write(f"  Hash: {hash_val}\n")
                f.write(f"  Binary: {info.binary_path}\n")
                if info.ptx_path:
                    f.write(f"  PTX: {info.ptx_path}\n")
                f.write(f"  Metadata: {info.metadata_path}\n")
                f.write(f"  Num warps: {info.metadata.get('num_warps', 'N/A')}\n")
                f.write(f"  Shared memory: {info.metadata.get('shared', 'N/A')} bytes\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("Kernel Launches\n")
            f.write("=" * 80 + "\n\n")
            for i, launch in enumerate(self.kernel_launches, 1):
                f.write(f"Launch #{i}: {launch.kernel_name}\n")
                f.write(f"  Timestamp: {launch.timestamp}\n")
                f.write(f"  Grid: {launch.grid}\n")
                f.write(f"  Block: {launch.block}\n")
                f.write(f"  Shared memory: {launch.shared_memory} bytes\n")
                f.write(f"  Num warps: {launch.num_warps}\n")
                f.write(f"  Global scratch: {launch.global_scratch_size} bytes (align: {launch.global_scratch_align})\n")
                f.write(f"  Profile scratch: {launch.profile_scratch_size} bytes (align: {launch.profile_scratch_align})\n")
                f.write(f"  Arguments: {len(launch.args_info)}\n")
                for arg_info in launch.args_info:
                    f.write(f"    [{arg_info.index}] {arg_info.arg_type}")
                    if arg_info.arg_type == 'tensor':
                        f.write(f" shape={arg_info.shape}, dtype={arg_info.dtype}, size={arg_info.size_bytes} bytes\n")
                        if arg_info.data_file:
                            f.write(f"        input: {arg_info.data_file}\n")
                        if arg_info.output_file:
                            f.write(f"        output: {arg_info.output_file}\n")
                    elif arg_info.arg_type == 'scalar':
                        f.write(f" {arg_info.dtype} = {arg_info.value}\n")
                    else:
                        f.write(f" {arg_info.dtype}\n")
                f.write("\n")
        
        print(f"[TritonKernelTracker] Report saved to: {report_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Track Triton kernel compilation and invocation",
        epilog="For a complete example, see example_vector_add.py"
    )
    parser.add_argument('--output-dir', type=str, default='./triton_kernel_tracking',
                        help='Directory to save tracking data')
    parser.add_argument('--save-binaries', action='store_true', default=True,
                        help='Save kernel binaries')
    
    args = parser.parse_args()
    
    # Initialize tracker
    output_dir = Path(args.output_dir)
    tracker = TritonKernelTracker(output_dir, save_binaries=args.save_binaries)
    
    print("\nTritonKernelTracker initialized.")
    print("The tracker will automatically capture kernel compilation and launch data.")
    print("\nUsage:")
    print("  1. Import your Triton kernels")
    print("  2. Run them normally")
    print("  3. Call tracker.save_summary() to export tracking data")
    print("\nFor a complete example, run: python example_vector_add.py")
