"""Independent CuTe workload used to check that capture is not FA4-specific."""
import argparse
from pathlib import Path

import cutlass.cute as cute
import torch
from CutedslTrace import Tracker


@cute.kernel
def add_kernel(a: cute.Tensor, b: cute.Tensor, output: cute.Tensor):
    index, _, _ = cute.arch.thread_idx()
    output[index] = a[index] + b[index]


@cute.jit
def add(a: cute.Tensor, b: cute.Tensor, output: cute.Tensor, stream):
    add_kernel(a, b, output).launch(grid=(1, 1, 1), block=(128, 1, 1), stream=stream)


def export_smoke(directory, target="sm_120a", sm_count=170):
    tracker = Tracker(directory, target=target, sm_count=sm_count)
    tracker.enable()
    try:
        a = torch.arange(128, device="cuda", dtype=torch.float32)
        b = torch.ones_like(a)
        output = torch.empty_like(a)
        tensors = [cute.runtime.from_dlpack(t, enable_tvm_ffi=True) for t in (a, b, output)]
        function = cute.compile(add, *tensors,
                                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                                options="--enable-tvm-ffi")
        function(a, b, output)
        function(output, b, output)
        tracker.check(output, expected=torch.arange(128, dtype=torch.float32) + 2)
        tracker.export()
    finally:
        tracker.disable()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "exports/vector_add"))
    parser.add_argument("--target", default="sm_120a")
    parser.add_argument("--sm-count", type=int, default=170)
    args = parser.parse_args()
    export_smoke(args.output_dir, args.target, args.sm_count)
