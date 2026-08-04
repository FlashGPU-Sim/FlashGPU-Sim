from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False
    print("[WARNING] PyTorch not available, tensor argument capture disabled")


@dataclass
class ArgumentInfo:
    """Information about a kernel argument."""

    index: int
    name: str
    arg_type: str
    dtype: Optional[str] = None
    shape: Optional[Tuple[int, ...]] = None
    value: Optional[Any] = None
    data_file: Optional[str] = None
    output_file: Optional[str] = None
    size_bytes: int = 0


@dataclass
class KernelLaunchInfo:
    """Information about a single kernel launch."""

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
    """Information about one compiled kernel binary."""

    kernel_name: str
    kernel_hash: str
    binary_path: str
    ptx_path: Optional[str]
    metadata_path: str
    metadata: Dict[str, Any]
    source_hash: str


class TrackingSession:
    """Own the records and files produced by one tracking run."""

    def __init__(
        self,
        output_dir,
        *,
        save_binaries: bool,
        capture_args: bool,
        mode: str,
        target_name: Optional[str],
    ):
        self.output_dir = Path(output_dir)
        self.save_binaries = save_binaries
        self.capture_args = capture_args
        self.mode = mode
        self.target_name = target_name

        self.compiled_kernels: Dict[str, KernelBinaryInfo] = {}
        self.kernel_launches: List[KernelLaunchInfo] = []
        self.launch_counter = 0

        self.binaries_dir = self.output_dir / "binaries"
        self.launchers_dir = self.output_dir / "launchers"
        self.data_dir = self.output_dir / "data"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_binaries:
            self.binaries_dir.mkdir(exist_ok=True)
            self.launchers_dir.mkdir(exist_ok=True)
            self.data_dir.mkdir(exist_ok=True)

    @staticmethod
    def normalize_grid(grid) -> tuple:
        """Normalize a launch grid to an (x, y, z) tuple."""
        if isinstance(grid, int):
            return (grid, 1, 1)
        if isinstance(grid, (list, tuple)):
            if len(grid) == 1:
                return (grid[0], 1, 1)
            if len(grid) == 2:
                return (grid[0], grid[1], 1)
            return tuple(grid[:3])
        return (grid, 1, 1)

    def record_kernel(self, name, metadata_group, hash_val, *, event="compiled"):
        """Copy compiler artifacts and register one compiled kernel."""
        if hash_val in self.compiled_kernels:
            return self.compiled_kernels[hash_val], False

        print(f"\n[TritonTracker] Kernel {event}: {name} (hash: {hash_val[:8]}...)")

        binary_path = None
        ptx_path = None
        metadata_path = None
        for category, path in metadata_group.items():
            if category.endswith(".cubin"):
                binary_path = path
            elif category.endswith(".ptx"):
                ptx_path = path
            elif category.endswith(".json"):
                metadata_path = path

        if not binary_path and not ptx_path:
            print(f"  [WARNING] No binary found for {name}")
            return None, False

        metadata = {}
        if metadata_path:
            with open(metadata_path) as metadata_file:
                metadata = json.load(metadata_file)

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

        kernel_info = KernelBinaryInfo(
            kernel_name=name,
            kernel_hash=hash_val,
            binary_path=str(saved_binary_path or binary_path),
            ptx_path=str(saved_ptx_path or ptx_path) if ptx_path else None,
            metadata_path=str(saved_metadata_path or metadata_path),
            metadata=metadata,
            source_hash=hashlib.sha256(str(metadata_group).encode()).hexdigest()[:16],
        )
        self.compiled_kernels[hash_val] = kernel_info
        return kernel_info, True

    def _serialize_tensor(self, tensor, launch_id: int, arg_idx: int, kernel_name: str):
        if not HAS_TORCH or not isinstance(tensor, torch.Tensor):
            return None, {}

        cpu_tensor = tensor.detach().cpu()
        np_array = cpu_tensor.numpy()
        tensor_path = self.data_dir / f"{kernel_name}_launch{launch_id}_arg{arg_idx}.bin"
        np_array.tofile(str(tensor_path))

        metadata = {
            "shape": list(np_array.shape),
            "dtype": str(np_array.dtype),
            "size_bytes": np_array.nbytes,
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad,
        }
        print(
            f"    Saved tensor arg[{arg_idx}]: shape={metadata['shape']}, "
            f"dtype={metadata['dtype']}, size={metadata['size_bytes']} bytes"
        )
        return str(tensor_path), metadata

    def capture_arguments(
        self, args: tuple, kernel_name: str, launch_id: int, *, snapshot_tensors: bool
    ):
        """Serialize runtime arguments and optionally snapshot tensors."""
        arg_infos = []
        snapshots = [] if HAS_TORCH and snapshot_tensors else None

        for idx, arg in enumerate(args):
            arg_info = ArgumentInfo(index=idx, name=f"arg{idx}", arg_type="unknown")

            if HAS_TORCH and isinstance(arg, torch.Tensor):
                data_file, metadata = self._serialize_tensor(arg, launch_id, idx, kernel_name)
                arg_info.arg_type = "tensor"
                arg_info.dtype = metadata.get("dtype", "unknown")
                arg_info.shape = tuple(metadata.get("shape", []))
                arg_info.data_file = data_file
                arg_info.size_bytes = metadata.get("size_bytes", 0)
                if snapshots is not None:
                    snapshots.append(arg.detach().clone())
            elif isinstance(arg, (int, float, bool)):
                arg_info.arg_type = "scalar"
                arg_info.value = arg
                arg_info.dtype = type(arg).__name__
                arg_info.size_bytes = sys.getsizeof(arg)
                print(f"    Captured scalar arg[{idx}]: {arg_info.dtype} = {arg}")
            elif isinstance(arg, (list, tuple)):
                arg_info.arg_type = "constexpr"
                arg_info.value = arg
                arg_info.dtype = "list/tuple"
                print(f"    Captured constexpr arg[{idx}]: {arg}")
            else:
                arg_info.dtype = type(arg).__name__
                print(f"    [WARNING] Unknown arg[{idx}] type: {type(arg)}")

            arg_infos.append(arg_info)

        return arg_infos, snapshots

    def record_launch(
        self,
        kernel_info: KernelBinaryInfo,
        grid,
        args: tuple,
        *,
        snapshot_tensors: bool,
    ):
        """Register one kernel launch and return its optional tensor snapshots."""
        self.launch_counter += 1
        grid = self.normalize_grid(grid)
        metadata = kernel_info.metadata
        num_warps = metadata.get("num_warps", 1)
        num_ctas = metadata.get("num_ctas", 1)
        shared_memory = metadata.get("shared", 0)
        block = (32 * num_warps, 1, 1)

        args_info = []
        snapshots = None
        if self.capture_args:
            print(f"  Capturing {len(args)} arguments...")
            args_info, snapshots = self.capture_arguments(
                args,
                kernel_info.kernel_name,
                self.launch_counter,
                snapshot_tensors=snapshot_tensors,
            )

        total_ctas = grid[0] * grid[1] * grid[2] * num_ctas
        launch_info = KernelLaunchInfo(
            timestamp=datetime.now().isoformat(),
            kernel_name=kernel_info.kernel_name,
            kernel_hash=kernel_info.kernel_hash,
            grid=grid,
            block=block,
            shared_memory=shared_memory,
            num_warps=num_warps,
            num_ctas=num_ctas,
            args_info=args_info,
            launch_id=self.launch_counter,
            stream=None,
            global_scratch_size=metadata.get("global_scratch_size", 0) * total_ctas,
            global_scratch_align=metadata.get("global_scratch_align", 1),
            profile_scratch_size=metadata.get("profile_scratch_size", 0) * total_ctas,
            profile_scratch_align=metadata.get("profile_scratch_align", 1),
        )
        self.kernel_launches.append(launch_info)

        print(f"\n[TritonTracker] Launch #{self.launch_counter}: {kernel_info.kernel_name}")
        print(f"  Grid: {grid}, Block: {block}")
        print(f"  Shared memory: {shared_memory} bytes")
        print(f"  Num warps: {num_warps}, Num CTAs: {num_ctas}")
        return launch_info, snapshots

    def capture_outputs(self, launch_info: KernelLaunchInfo, args: tuple, snapshots):
        """Compare post-launch tensors with snapshots and save modified outputs."""
        if not self.capture_args or snapshots is None or not HAS_TORCH:
            return

        print(
            f"\n[TritonTracker] Detecting and capturing outputs for launch "
            f"#{launch_info.launch_id}"
        )
        torch.cuda.synchronize()

        snapshot_idx = 0
        for idx, arg in enumerate(args):
            if not isinstance(arg, torch.Tensor):
                continue
            if snapshot_idx >= len(snapshots):
                print(f"    [WARNING] Snapshot index out of bounds for arg[{idx}]")
                continue

            arg_info = launch_info.args_info[idx] if idx < len(launch_info.args_info) else None
            pre_snapshot = snapshots[snapshot_idx]
            snapshot_idx += 1
            if not arg_info or arg_info.arg_type != "tensor":
                continue

            try:
                if arg.dtype.is_floating_point:
                    is_output = not torch.allclose(arg, pre_snapshot, rtol=1e-7, atol=1e-7)
                else:
                    is_output = not torch.equal(arg, pre_snapshot)
            except Exception as error:
                print(f"    [WARNING] Could not compare arg[{idx}]: {error}")
                is_output = True

            if is_output:
                output_path = self.data_dir / (
                    f"{launch_info.kernel_name}_launch{launch_info.launch_id}_"
                    f"arg{idx}_output.bin"
                )
                np_array = arg.detach().cpu().numpy()
                np_array.tofile(str(output_path))
                arg_info.output_file = str(output_path)
                print(
                    f"    Saved OUTPUT arg[{idx}]: shape={list(np_array.shape)}, "
                    f"dtype={np_array.dtype}, size={np_array.nbytes} bytes"
                )
            else:
                print(f"    Skipped arg[{idx}]: unchanged (input-only)")

        print(f"  Output capture complete for launch #{launch_info.launch_id}")

    def save(self):
        """Write the machine-readable summary and human-readable report."""
        launches = []
        for launch in self.kernel_launches:
            launch_dict = asdict(launch)
            launch_dict["args_info"] = [asdict(arg) for arg in launch.args_info]
            launches.append(launch_dict)

        summary = {
            "tracking_session": {
                "timestamp": datetime.now().isoformat(),
                "output_dir": str(self.output_dir),
                "mode": self.mode,
                "target": self.target_name,
                "total_kernels_compiled": len(self.compiled_kernels),
                "total_launches": len(self.kernel_launches),
                "argument_capture_enabled": self.capture_args,
            },
            "compiled_kernels": {
                hash_val: asdict(info) for hash_val, info in self.compiled_kernels.items()
            },
            "kernel_launches": launches,
        }

        summary_path = self.output_dir / "tracking_summary.json"
        with open(summary_path, "w") as summary_file:
            json.dump(summary, summary_file, indent=2)

        print(f"\n[TritonTracker] Summary saved to: {summary_path}")
        print(f"  Total kernels compiled: {len(self.compiled_kernels)}")
        print(f"  Total kernel launches: {len(self.kernel_launches)}")

        report_path = self.output_dir / "tracking_report.txt"
        with open(report_path, "w") as report:
            report.write("=" * 80 + "\n")
            report.write("Triton Kernel Tracking Report\n")
            report.write("=" * 80 + "\n\n")
            report.write(f"Timestamp: {datetime.now().isoformat()}\n")
            report.write(f"Output directory: {self.output_dir}\n")
            report.write(f"Mode: {self.mode}\n")
            if self.target_name is not None:
                report.write(f"Target: {self.target_name}\n")
            report.write(f"Total kernels compiled: {len(self.compiled_kernels)}\n")
            report.write(f"Total kernel launches: {len(self.kernel_launches)}\n\n")

            report.write("=" * 80 + "\nCompiled Kernels\n" + "=" * 80 + "\n\n")
            for hash_val, info in self.compiled_kernels.items():
                report.write(f"Kernel: {info.kernel_name}\n")
                report.write(f"  Hash: {hash_val}\n")
                report.write(f"  Binary: {info.binary_path}\n")
                if info.ptx_path:
                    report.write(f"  PTX: {info.ptx_path}\n")
                report.write(f"  Metadata: {info.metadata_path}\n")
                report.write(f"  Num warps: {info.metadata.get('num_warps', 'N/A')}\n")
                report.write(
                    f"  Shared memory: {info.metadata.get('shared', 'N/A')} bytes\n\n"
                )

            report.write("=" * 80 + "\nKernel Launches\n" + "=" * 80 + "\n\n")
            for index, launch in enumerate(self.kernel_launches, 1):
                report.write(f"Launch #{index}: {launch.kernel_name}\n")
                report.write(f"  Timestamp: {launch.timestamp}\n")
                report.write(f"  Grid: {launch.grid}\n")
                report.write(f"  Block: {launch.block}\n")
                report.write(f"  Shared memory: {launch.shared_memory} bytes\n")
                report.write(f"  Num warps: {launch.num_warps}\n")
                report.write(
                    f"  Global scratch: {launch.global_scratch_size} bytes "
                    f"(align: {launch.global_scratch_align})\n"
                )
                report.write(
                    f"  Profile scratch: {launch.profile_scratch_size} bytes "
                    f"(align: {launch.profile_scratch_align})\n"
                )
                report.write(f"  Arguments: {len(launch.args_info)}\n")
                for arg_info in launch.args_info:
                    report.write(f"    [{arg_info.index}] {arg_info.arg_type}")
                    if arg_info.dtype:
                        report.write(f" ({arg_info.dtype})")
                    if arg_info.shape:
                        report.write(f" shape={arg_info.shape}")
                    if arg_info.value is not None:
                        report.write(f" value={arg_info.value}")
                    report.write("\n")
                report.write("\n")

        print(f"[TritonTracker] Report saved to: {report_path}")
