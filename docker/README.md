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

[`Dockerfile.ci`](Dockerfile.ci) is the reusable image used by GitHub Actions.
It starts from NVIDIA's CUDA 12.8 `base` image and installs pinned leaf
packages rather than the full CUDA development metapackage: `cuda-nvcc`,
`cuda-nvvm`, `cuda-crt`, and `cuda-cccl` provide compilation support;
`cuda-cudart`, `cuda-cudart-dev`, and `cuda-driver-dev` provide the runtime,
headers, and driver stub; and `cuda-cuobjdump` plus `cuda-nvdisasm` provide the
binary-inspection tools. CUDA math-library development packages and Nsight
Compute are not installed.
The image also contains the simulator build dependencies, GoogleTest 1.12.1
under `/opt/googletest`, and the same Python 3.12.3, PyTorch 2.9.0, Triton 3.5.0,
and NumPy 2.4.0 versions as the development image. CI uses PyTorch's CPU-only
wheel because Triton capture is offline. Repository source and test binaries
are not included; CI mounts and builds the selected commit at runtime. The
in-tree `TritonTrace` package is exposed from the mounted checkout through
`PYTHONPATH=/gpgpu-sim/tools`.

The [PR workflow](../.github/workflows/pr-tests.yml) pins the prebuilt image by
immutable digest. The workflow is the source of truth for the image selected
by CI.

Build and smoke-test the equivalent environment locally:

```bash
docker build -f docker/Dockerfile.ci \
  -t flashgpu-sim-ci:local .
docker run --rm \
  -v "$PWD:/gpgpu-sim:ro" \
  flashgpu-sim-ci:local \
  bash -c 'python3 -c "import torch, triton, TritonTrace" && nvcc --version'
```

The development image and helper do not replace or modify the CI environment.
