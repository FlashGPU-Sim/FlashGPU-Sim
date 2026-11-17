# CutedslTrace offline exporter

`CutedslTrace.Tracker` records normal CuTe DSL compilation and function calls,
without executing GPU kernels or knowing anything about the workload. Install
with `pip install -e tools`; the export environment also needs PyTorch, CuTe DSL
and TVM FFI (tested with 2.9.0, 4.5.2 and 0.1.13.post2 respectively).

```python
import torch
import CutedslTrace

tracker = CutedslTrace.Tracker("output", target="sm_100a", sm_count=148)
tracker.enable()
try:
    x = torch.ones(128, device="cuda")
    out = my_workload(x)  # Normal CuTe DSL compile/call path.
    tracker.check(out, expected=2)  # Optional replay check.
    tracker.export()
finally:
    tracker.disable()  # Restore hooks even if the workload raises.
```

CUDA allocations while capture is enabled use CPU storage and CUDA-shaped metadata.
CuTe compilation keeps the workload's TVM FFI options; calls are recorded and
skipped. Tensor storage, views, scalar arguments and call order are serialized.
There is no GPU output during export. Host computations on captured buffers and
host control flow depending on CUDA values are rejected.

`enable()` starts or resumes capture; `disable()` restores hooks/environment and
retains records. `export()` automatically disables capture, generates a single
harness for all captured calls, and removes temporary files after success.
Repeated enable/disable calls are harmless; only one tracker may be enabled.
After a successful export, use a new tracker for another capture. On an error,
call `disable()` (as above); no output is automatically published. Temporary
files from an abandoned, disabled tracker are reclaimed when it is collected.
There is no required `with` block or public `close()` method.

Outputs: `modules/*.o`, `modules/*.ptx`, `modules/*.cubin`, `data/*.bin`,
`harness.cc`, `Makefile`, `manifest.json`. TVM FFI uses its standard C interface,
so no workload-specific C header or handwritten harness is needed.

```bash
cd output
make
./replay  # Native GPU in a shell without the simulator environment.
# Or, with the same executable and a prepared simulator working directory:
source /path/to/flashgpu-sim/setup_environment
./replay
```

The caller prepares the simulator configuration in the working directory.
The simulator's `ptxas` must support the exported PTX version. If the simulator
uses an older CUDA Toolkit, set `PTXAS_CUDA_INSTALL_PATH` to a compatible Toolkit
(e.g. CUDA 13.2 for PTX 9.1) after sourcing its environment.
The executable finds PTX/data beside itself, replays calls sequentially, writes
`data/storageN.result.bin`, and optionally performs the requested exact checks.
The exporter does not generate simulator configs, SASS guides, or a `make run`.

Build dependencies are recorded as overridable Make variables: `CUDA_HOME`,
`TVM_FFI_ROOT`, `CUTEDSL_LIB`. Build uses the normal CUDA Toolkit's shared
`libcudart` and requires no simulator build. CUDA/TVM library RUNPATHs provide
native defaults; `LD_LIBRARY_PATH` from simulator setup takes precedence.
CuTe's static host glue uses lazy CUDA symbol binding so unused CUDA APIs do
not prevent startup with the simulator's partial CUDA implementation.
Replay needs no Python interpreter, but retains the shared TVM FFI dependency.
The exported GPU architecture must match the hardware used for native replay
(e.g. `sm_100a` FA4 artifacts need B200-class hardware, not RTX 5090).

Initial scope: TVM FFI ABI, positional tensor/bool/int/float/None arguments,
one device, sequential calls and tensor views. Plain CuTe C ABI, nested argument
structures, explicit streams, host mutations between calls, and concurrent
enabled trackers are rejected/not supported. Use a fresh Python process for an
export; compiled functions cached by a workload belong to that tracker.
CUDA API/device-description coverage is intentionally limited to the queries
implemented by the offline layer; unsupported queries must not be interpreted
as properties of the target GPU.
