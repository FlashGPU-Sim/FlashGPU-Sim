# FlashGPU-Sim Development Container

The development image provides CUDA 12.8, the simulator build dependencies,
and the Python stack used to develop TritonTrace:

- Python 3.12.3
- PyTorch 2.9.0
- Triton 3.5.0
- NumPy 2.4.0

The repository is mounted at `/workspace/flashgpu-sim`. The container user
uses the host user's UID and GID, so files created in the mounted repository
remain owned by the host user.

## Prerequisites

Install Docker with the Compose plugin. GPU-enabled containers additionally
require an NVIDIA driver and the NVIDIA Container Toolkit on the host.

## Build the Image

Run all commands from the repository root:

```bash
./docker.sh build
```

## Open a Development Shell

Start a disposable shell without GPU access:

```bash
./docker.sh shell
```

The shell starts in the native CUDA environment. Configure FlashGPU-Sim only
when building or running the simulator:

```bash
source setup_environment
make -j "$(nproc)"
```

For example, run the CUDA VectorAdd tutorial:

```bash
./tutorials/vectorAdd/run.sh
```

## Use a GPU

Expose all available GPUs:

```bash
./docker.sh shell --gpu
```

Select one host GPU for CUDA applications:

```bash
./docker.sh shell --gpu 1
```

The latter still attaches the GPU resources requested by Compose and sets
`CUDA_VISIBLE_DEVICES=1` inside the container.

For example, capture the Triton GEMM tutorial from a GPU-enabled shell:

```bash
cd tutorials/triton-gemm
./capture.sh
```

## Persistent Data

The source tree is a host bind mount. Python, Triton, and PyTorch caches use
the `flashgpusim-cache` Docker volume and persist across disposable shells.
Containers themselves are removed when their shells exit.

## Nsight Compute

Nsight Compute is not installed in the image. Install a version compatible
with the host GPU and driver on the host, then use
[`tutorials/profile_ncu.sh`](../tutorials/profile_ncu.sh) outside the
container.

## CI Image

[`Dockerfile.ci`](Dockerfile.ci) is a separate minimal image used by the
GitHub Actions workflow. The development image and helper do not replace or
modify the CI environment.
