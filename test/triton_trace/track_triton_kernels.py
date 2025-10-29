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
    
    def __init__(self, output_dir: Path, save_binaries: bool = True, capture_args: bool = True):
        self.output_dir = output_dir
        self.save_binaries = save_binaries
        self.capture_args = capture_args
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for tracked data
        self.compiled_kernels: Dict[str, KernelBinaryInfo] = {}
        self.kernel_launches: List[KernelLaunchInfo] = []
        self.launch_counter = 0
        self.pending_args: Optional[tuple] = None  # Store args from JIT wrapper
        self.pending_grid: Optional[tuple] = None  # Store grid from JIT wrapper
        self.pending_args_snapshots: Optional[List[torch.Tensor]] = None  # Store pre-launch tensor snapshots
        
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
    
    def _install_hooks(self):
        """Install Triton runtime hooks"""
        # Hook for kernel loading (captures binaries)
        triton.knobs.runtime.kernel_load_end_hook.add(self._on_kernel_load)
        
        # Hook for kernel launch (captures invocations)
        triton.knobs.runtime.launch_enter_hook.add(self._on_launch_enter)
        
        # Hook for kernel exit (captures outputs after execution)
        triton.knobs.runtime.launch_exit_hook.add(self._on_launch_exit)
        
        # Hook for post-run to capture arguments
        # This gets called after JIT compilation but before actual launch
        def jit_hook(key, repr_str, fn, compile_info, is_manual_warmup, already_compiled):
            # Store compile info for argument capture
            pass
        
        # We'll use a custom pre_run_hook approach by monkey-patching
        self._patch_jit_run()
        
        print("[TritonKernelTracker] Hooks installed")
    
    def _patch_jit_run(self):
        """Patch JITFunction.run to capture arguments and grid"""
        # Store original run method
        from triton.runtime.jit import JITFunction
        original_run = JITFunction.run
        tracker_self = self
        
        def patched_run(jit_self, *args, **kwargs):
            # Capture arguments and grid before running
            warmup = kwargs.get('warmup', False)
            if tracker_self.capture_args and not warmup:
                # Store args for the launch hook
                tracker_self.pending_args = args
                
                # Capture and evaluate grid like Triton does (see jit.py line 693-696)
                grid = kwargs.get('grid')
                if grid is not None:
                    # Build bound_args from the function signature
                    # This is a simplified version - Triton does more complex binding
                    if callable(grid):
                        # Try to evaluate with metadata-like dict
                        # For simple grids like lambda meta: (cdiv(N, meta['BLOCK_SIZE']),)
                        # we need to pass constexprs. Try with kwargs that might have them.
                        try:
                            # Extract constexpr values from kwargs (like BLOCK_SIZE=256)
                            meta_dict = {k: v for k, v in kwargs.items() if k.isupper() or k == 'grid'}
                            meta_dict.pop('grid', None)  # Remove grid itself
                            grid_val = grid(meta_dict)
                        except:
                            # Fallback: just call it with empty dict
                            try:
                                grid_val = grid({})
                            except:
                                grid_val = (1, 1, 1)
                        grid = grid_val
                    
                    # Normalize to 3-tuple
                    if isinstance(grid, int):
                        tracker_self.pending_grid = (grid, 1, 1)
                    elif isinstance(grid, (list, tuple)):
                        if len(grid) == 1:
                            tracker_self.pending_grid = (grid[0], 1, 1)
                        elif len(grid) == 2:
                            tracker_self.pending_grid = (grid[0], grid[1], 1)
                        else:
                            tracker_self.pending_grid = tuple(grid[:3])
                    else:
                        tracker_self.pending_grid = (grid, 1, 1)
            
            # Call original run
            result = original_run(jit_self, *args, **kwargs)
            
            # Don't clear pending data here - launch_exit_hook needs it!
            # It will be cleared in _on_launch_exit
            # Note: If no launch happens (warmup), we should clear it
            if warmup or not tracker_self.capture_args:
                tracker_self.pending_args = None
                tracker_self.pending_grid = None
                tracker_self.pending_args_snapshots = None
            
            return result
        
        # Replace the run method
        JITFunction.run = patched_run
        print("[TritonKernelTracker] Patched JITFunction.run to capture arguments and grid")
    
    def _on_kernel_load(self, module, function, name, metadata_group, hash_val):
        """Called when a kernel binary is loaded"""
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
        """Capture and serialize kernel arguments"""
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
                # Find the matching kernel info
                matching_kernel = None
                for kernel_info in self.compiled_kernels.values():
                    if kernel_info.kernel_name == kernel_name:
                        matching_kernel = kernel_info
                        break
                
                if matching_kernel:
                    print(f"  Regenerating harness with validation code...")
                    self._generate_launch_specific_harness(kernel_info=matching_kernel, 
                                                           launch_info=launch_info)
        
        # Clear pending args after processing
        self.pending_args = None
        self.pending_grid = None
    
    def _on_launch_enter(self, launch_metadata):
        """Called before kernel launch"""
        self.launch_counter += 1
        
        # Extract metadata (LazyDict)
        metadata = launch_metadata.get() if hasattr(launch_metadata, 'get') else launch_metadata
        
        if not isinstance(metadata, dict):
            return
        
        kernel_name = metadata.get('name', 'unknown')
        function = metadata.get('function', None)
        
        # Try to find matching kernel
        matching_kernel = None
        for kernel_info in self.compiled_kernels.values():
            if kernel_info.kernel_name == kernel_name:
                matching_kernel = kernel_info
                break
        
        if not matching_kernel:
            print(f"[TritonKernelTracker] Launch #{self.launch_counter}: {kernel_name} (not yet tracked)")
            return
        
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
        if not isinstance(grid, tuple):
            grid = (grid, 1, 1)
        elif len(grid) == 1:
            grid = (grid[0], 1, 1)
        elif len(grid) == 2:
            grid = (grid[0], grid[1], 1)
        
        meta_dict = matching_kernel.metadata
        num_warps = meta_dict.get('num_warps', 1)
        num_ctas = meta_dict.get('num_ctas', 1)
        cluster_dims = meta_dict.get('cluster_dims', (1, 1, 1))
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
            stream=None
        )
        
        self.kernel_launches.append(launch_info)
        
        print(f"\n[TritonKernelTracker] Launch #{self.launch_counter}: {kernel_name}")
        print(f"  Grid: {grid}, Block: {block}")
        print(f"  Shared memory: {shared_memory} bytes")
        print(f"  Num warps: {num_warps}, Num CTAs: {num_ctas}")
        
        # Generate launcher with actual arguments if captured
        # Note: This will be regenerated after launch_exit with validation code
        if self.save_binaries and args_info:
            self._generate_launch_specific_harness(kernel_info=matching_kernel, 
                                                     launch_info=launch_info)
    
    def _generate_launch_specific_harness(self, kernel_info: KernelBinaryInfo, launch_info: KernelLaunchInfo):
        """Generate a harness for a specific launch with actual argument data"""
        kernel_name = kernel_info.kernel_name
        launch_id = launch_info.launch_id
        
        # Read PTX to embed as C string
        binary_path = Path(kernel_info.binary_path)
        ptx_path = binary_path.parent / f"{kernel_name}.ptx"
        
        ptx_string = None
        if ptx_path.exists():
            try:
                with open(ptx_path, 'r') as f:
                    ptx_content = f.read()
                # Escape for C string literal - split into multiple lines for readability
                lines = ptx_content.split('\n')
                escaped_lines = ['    "' + line.replace('\\', '\\\\').replace('"', '\\"') + '\\n"' for line in lines]
                ptx_string = '\n'.join(escaped_lines)
            except Exception as e:
                print(f"  Warning: Could not read PTX: {e}")
        
        # Make binary path relative to launchers directory (fallback)
        if not binary_path.is_absolute():
            binary_path = (Path.cwd() / binary_path).resolve()
        launchers_dir = self.launchers_dir.resolve()
        try:
            relative_binary_path = os.path.relpath(binary_path, launchers_dir)
        except:
            relative_binary_path = str(binary_path)
        
        # Build argument loading code
        arg_alloc_code = []
        arg_init_code = []
        arg_cleanup_code = []
        arg_pointers = []
        validation_code = []
        
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
                    
                # Generate code to load tensor using simple CUDA Runtime API
                arg_alloc_code.append(f"""
    // Argument {idx}: Tensor ({arg_info.shape}, {arg_info.dtype})
    void* d_arg{idx};
    size_t arg{idx}_size = {arg_info.size_bytes};
    cudaMalloc(&d_arg{idx}, arg{idx}_size);
    
    // Load data from file
    {{
        char data_path[2048];
        snprintf(data_path, sizeof(data_path), "%s/{relative_data_path}", exe_path);
        FILE* fp{idx} = fopen(data_path, "rb");
        if (!fp{idx}) {{
            fprintf(stderr, "Error: Cannot open %s\\n", data_path);
            return 1;
        }}
        void* h_arg{idx} = malloc(arg{idx}_size);
        fread(h_arg{idx}, 1, arg{idx}_size, fp{idx});
        fclose(fp{idx});
        cudaMemcpy(d_arg{idx}, h_arg{idx}, arg{idx}_size, cudaMemcpyHostToDevice);
        free(h_arg{idx});
        printf("  Loaded arg[{idx}]: tensor shape={list(arg_info.shape)}, dtype={arg_info.dtype}, size=%zu bytes\\n", arg{idx}_size);
    }}
""")
                arg_pointers.append(f"&d_arg{idx}")
                arg_cleanup_code.append(f"    cudaFree(d_arg{idx});")
                
                # Add validation code if we have expected output
                if arg_info.output_file:
                    output_file_path = Path(arg_info.output_file)
                    if not output_file_path.is_absolute():
                        output_file_path = (Path.cwd() / output_file_path).resolve()
                    try:
                        relative_output_path = os.path.relpath(output_file_path, launchers_dir)
                    except:
                        relative_output_path = arg_info.output_file
                    
                    validation_code.append(f"""
    // Validate output for arg[{idx}]
    {{
        char expected_output_path[2048];
        snprintf(expected_output_path, sizeof(expected_output_path), "%s/{relative_output_path}", exe_path);
        FILE* fp_expected{idx} = fopen(expected_output_path, "rb");
        if (fp_expected{idx}) {{
            void* h_expected{idx} = malloc(arg{idx}_size);
            void* h_actual{idx} = malloc(arg{idx}_size);
            
            fread(h_expected{idx}, 1, arg{idx}_size, fp_expected{idx});
            fclose(fp_expected{idx});
            
            cudaMemcpy(h_actual{idx}, d_arg{idx}, arg{idx}_size, cudaMemcpyDeviceToHost);
            
            // Compare outputs (byte-by-byte for exact match, or element-wise with tolerance)
            int mismatches = 0;
            size_t num_elements = arg{idx}_size / sizeof(float);  // Assuming float, adjust as needed
            float* expected_data = (float*)h_expected{idx};
            float* actual_data = (float*)h_actual{idx};
            float tolerance = 1e-5f;
            
            for (size_t i = 0; i < num_elements && mismatches < 10; i++) {{
                float diff = fabsf(expected_data[i] - actual_data[i]);
                if (diff > tolerance) {{
                    if (mismatches == 0) {{
                        printf("\\n  Validation FAILED for arg[{idx}]:\\n");
                    }}
                    printf("    Element %zu: expected=%.6f, actual=%.6f, diff=%.6e\\n", 
                           i, expected_data[i], actual_data[i], diff);
                    mismatches++;
                }}
            }}
            
            if (mismatches == 0) {{
                printf("  Validation PASSED for arg[{idx}]: all %zu elements match within tolerance %.2e\\n", 
                       num_elements, tolerance);
            }} else {{
                printf("  Total mismatches for arg[{idx}]: %d (showing first 10)\\n", mismatches);
            }}
            
            free(h_expected{idx});
            free(h_actual{idx});
        }} else {{
            printf("  No expected output file found for arg[{idx}], skipping validation\\n");
        }}
    }}
""")
                
            elif arg_info.arg_type == 'scalar':
                # Generate code for scalar argument
                dtype_map = {
                    'int': 'int32_t',
                    'float': 'float',
                    'bool': 'uint8_t'
                }
                c_type = dtype_map.get(arg_info.dtype, 'int32_t')
                value = arg_info.value
                
                arg_alloc_code.append(f"""
    // Argument {idx}: Scalar ({arg_info.dtype} = {value})
    {c_type} arg{idx} = {value};
    printf("  Set arg[{idx}]: {arg_info.dtype} = {value}\\n");
""")
                arg_pointers.append(f"&arg{idx}")
        
        # Generate harness that loads fatbin from file (simpler than embedding)
        if ptx_string:
            fatbin_filename = f"{kernel_name}_launch{launch_id}_kernel.fatbin"

            cpp_code = f"""
// Standalone harness for Triton kernel: {kernel_name}
// Launch ID: {launch_id}
// Generated by TritonKernelTracker with captured arguments
// Kernel hash: {kernel_info.kernel_hash}
//
// Loads fatbin from file using cuModuleLoad.
// Compatible with cuobjdump (can extract PTX from the fatbin file).

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>

int main(int argc, char** argv) {{
    printf("=== Standalone Harness for Triton Kernel ===\\n");
    printf("Kernel: {kernel_name}\\n");
    printf("Launch ID: {launch_id}\\n");
    printf("Grid: {launch_info.grid}\\n");
    printf("Block: {launch_info.block}\\n");
    printf("Shared memory: {launch_info.shared_memory} bytes\\n");
    printf("\\n");

    // Initialize CUDA Driver API
    cuInit(0);

    CUdevice device;
    CUcontext context;
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);

    // Resolve fatbin path relative to executable location
    char exe_path[1024];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len == -1) {{
        fprintf(stderr, "Failed to get executable path\\n");
        return 1;
    }}
    exe_path[len] = '\\0';
    
    // Get directory of executable
    char* last_slash = strrchr(exe_path, '/');
    if (last_slash) {{
        *last_slash = '\\0';
    }}
    
    // Build full path to fatbin
    char fatbin_path[2048];
    snprintf(fatbin_path, sizeof(fatbin_path), "%s/{fatbin_filename}", exe_path);
    
    // Load module directly from fatbin file
    CUmodule module;
    CUresult load_result = cuModuleLoad(&module, fatbin_path);
    if (load_result != CUDA_SUCCESS) {{
        const char* errStr;
        cuGetErrorString(load_result, &errStr);
        fprintf(stderr, "Failed to load fatbin: %s\\n", errStr);
        return 1;
    }}
    printf("Loaded module from: %s\\n", fatbin_path);

    // Get kernel function
    CUfunction kernel_func;
    CUresult func_result = cuModuleGetFunction(&kernel_func, module, "{kernel_name}");
    if (func_result != CUDA_SUCCESS) {{
        const char* errStr;
        cuGetErrorString(func_result, &errStr);
        fprintf(stderr, "Failed to get function: %s\\n", errStr);
        return 1;
    }}
    printf("Got kernel function: {kernel_name}\\n\\n");

    printf("Initializing arguments:\\n");
{''.join(arg_alloc_code)}

    // Triton adds global_scratch and profile_scratch arguments
    void* global_scratch = NULL;
    void* profile_scratch = NULL;

    // Setup kernel arguments (user args + global_scratch + profile_scratch)
    void* args[] = {{ {', '.join(arg_pointers)}, &global_scratch, &profile_scratch }};

    // Launch kernel
    printf("\\nLaunching kernel...\\n");
    CUresult launch_result = cuLaunchKernel(
        kernel_func,
        {launch_info.grid[0]}, {launch_info.grid[1]}, {launch_info.grid[2]},  // grid
        {launch_info.block[0]}, {launch_info.block[1]}, {launch_info.block[2]},  // block
        {launch_info.shared_memory},  // shared memory
        0,  // stream
        args,
        0   // extra
    );

    if (launch_result != CUDA_SUCCESS) {{
        const char* errStr;
        cuGetErrorString(launch_result, &errStr);
        fprintf(stderr, "Kernel launch failed: %s\\n", errStr);
        return 1;
    }}

    // Synchronize
    cuCtxSynchronize();
    printf("Kernel execution completed successfully\\n");

    // Validate outputs
    printf("\\nValidating outputs...\\n");
{''.join(validation_code)}

    // Cleanup
    printf("\\nCleaning up...\\n");
{''.join(arg_cleanup_code)}    cuModuleUnload(module);
    cuCtxDestroy(context);

    printf("Done!\\n");
    return 0;
}}
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
        
        # Save PTX file for linking with nvcc
        if ptx_string:
            ptx_filename = f"{kernel_name}_launch{launch_id}_kernel.ptx"
            ptx_file_path = self.launchers_dir / ptx_filename
            # Write the original PTX content (not escaped)
            ptx_path_src = Path(kernel_info.binary_path).parent / f"{kernel_name}.ptx"
            if ptx_path_src.exists():
                import shutil
                # Read and fix PTX: replace sm_XXXa with sm_XXX for fatbin compatibility
                with open(ptx_path_src, 'r') as f:
                    ptx_content = f.read()
                # Replace sm_120a -> sm_120 (and any other 'a' variants)
                import re
                ptx_fixed = re.sub(r'\.target (sm_\d+)a\b', r'.target \1', ptx_content)
                with open(ptx_file_path, 'w') as f:
                    f.write(ptx_fixed)
                print(f"  Copied PTX for linking: {ptx_file_path}")
                if 'sm_' in ptx_content and ptx_content != ptx_fixed:
                    print(f"    Fixed architecture variants (e.g., sm_120a -> sm_120) for fatbin compatibility")
        
        # Generate Makefile. If PTX was found we'll produce steps to build a fatbin
        makefile_path = self.launchers_dir / f"{kernel_name}_launch{launch_id}_Makefile"
        target_name = f"{kernel_name}_launch{launch_id}"

        if ptx_string:
            ptx_filename = f"{kernel_name}_launch{launch_id}_kernel.ptx"
            fatbin_filename = f"{kernel_name}_launch{launch_id}_kernel.fatbin"

            makefile_content = f"""
# Makefile for {kernel_name} launch {launch_id}
# Compiles PTX into fatbinary (loaded at runtime).
# The fatbin file can be examined with cuobjdump to extract PTX.

NVCC = nvcc
CUDA_FLAGS = -lcudart -lcuda
TARGET = {target_name}
PTX_FILE = {ptx_filename}
FATBIN_FILE = {fatbin_filename}

# Auto-detect architecture from PTX .target directive
ARCH := $(shell grep '^\\.target' $(PTX_FILE) | head -1 | awk '{{print $$2}}')

all: $(TARGET) $(FATBIN_FILE)

# Create fatbin from PTX (use detected architecture)
$(FATBIN_FILE): $(PTX_FILE)
\t@echo "Detected architecture: $(ARCH)"
\t$(NVCC) -fatbin -arch=$(ARCH) -o $(FATBIN_FILE) $(PTX_FILE)

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
    CUDA_CHECK(cuCtxCreate(&context, 0, device));
    
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

