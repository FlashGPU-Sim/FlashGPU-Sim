# FA2/FA3 Reproduction Notes

This branch packages the June 17 FA2/H100 simulator state for remote
reproduction. The official FlashAttention changes are not committed as
submodule contents; `make prepare-fa3-flash-attention` clones the pinned
upstream commit and applies the local FA2/FA3 patch before building FA2
sensitivity or FA3 profile variants.

## Checkout

```bash
git fetch origin codex/fa2-h100-repro-20260617
git switch codex/fa2-h100-repro-20260617
cd test
make prepare-fa3-flash-attention
cd ..
```

If the generated FlashAttention checkout already contains these local hooks,
the prepare script may report that the patch is already applied.

## Build Simulator

CUDA 12.8 is the validated setup for these runs. Adjust `CUDA_INSTALL_PATH`
only if the remote CUDA 12.8 install lives elsewhere.

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda-12.8
export CUDA_HOME=${CUDA_INSTALL_PATH}
export CUDA_PATH=${CUDA_INSTALL_PATH}
export CUDA_VERSION_NUMBER=12080
export GPGPUSIM_CONFIG=gcc-13.3.0/cuda-12080/release
source setup_environment
make -j$(nproc)
```

The FA2 simulator config to use is:

```text
SM90_H100_1500MHZ_HBM80_L2S160_MSHR512_L2NOC1700_FA2_REGALLOC
```

It enables PTX register allocation and PTX reorder. Its calibrated classic
`mma.sync` tensor[0] parameters are latency `22` and initiation `6`.

The formal FA3 baseline config is:

```text
SM90_H100_1500MHZ_HBM80_L2S160_MSHR512_L2NOC1700
```

## Build FA2/FA3 Binaries

From the repo root:

```bash
cd test
CUDA_INSTALL_PATH=/usr/local/cuda-12.8 ./run_tests.sh build hopper-fa2-small
CUDA_INSTALL_PATH=/usr/local/cuda-12.8 ./run_tests.sh build hopper-fa2-medium
CUDA_INSTALL_PATH=/usr/local/cuda-12.8 ./run_tests.sh build hopper-fa2-large-h16d128-full
CUDA_INSTALL_PATH=/usr/local/cuda-12.8 ./run_tests.sh build hopper-fa2-large-h32d64-full
CUDA_INSTALL_PATH=/usr/local/cuda-12.8 ./run_tests.sh build hopper-fa2-sensitivity-h1d128
CUDA_INSTALL_PATH=/usr/local/cuda-12.8 ./run_tests.sh build hopper-fa3-extended
CUDA_INSTALL_PATH=/usr/local/cuda-12.8 ./run_tests.sh build hopper-fa3-sensitivity-extended
cd ..
```

Use narrower targets when possible. For example, `hopper-fa2-small` builds only
the four small split FA2 binaries instead of every FA2 case.

## Run FA2 Small Simulator Cases

The four-case queue used for the saved result table is committed at
`test/jobs/fa2_small4.tsv`.

```bash
python3 test/scripts/run_sim_queue.py \
  --root . \
  --run-root test/run/FA2_SMALL4_REPRO_$(date +%Y%m%d_%H%M%S) \
  --jobs test/jobs/fa2_small4.tsv \
  --config SM90_H100_1500MHZ_HBM80_L2S160_MSHR512_L2NOC1700_FA2_REGALLOC \
  --max-parallel 4 \
  --cpu-sets 0,2,4,6 8,10,12,14 16-19 20-23 \
  --threads-per-job 4 \
  --timeout 0 \
  --cuda-path /usr/local/cuda-12.8 \
  --cuda-version-number 12080 \
  --gpgpusim-config gcc-13.3.0/cuda-12080/release
```

The queue writes `status/summary.tsv`; `gpu_tot_sim_cycle` is the cycle column
used for comparison.

## Saved FA2 Small Result

The latest local result is recorded in:

```text
docs/results/fa2_small_tensor6_lat22_20260617.tsv
docs/results/fa2_small_tensor6_lat22_20260617_logs/
```

Summary:

```text
H32D64FullB32S256      sim 154427 cycles, 102.951 us; H100 NCU 140113.9 cycles, 99.168 us
H16D128FullB32S256     sim 126540 cycles, 84.360 us; H100 NCU 120424.2 cycles, 82.976 us
H32D64CausalB32S256    sim 117846 cycles, 78.564 us; H100 NCU 123712.4 cycles, 86.016 us
H16D128CausalB32S256   sim 111523 cycles, 74.349 us; H100 NCU 112769.7 cycles, 80.352 us
```

The H100 reference source on this machine was:

```text
test/run/H100_FA2_FULL_NCU_CUDA128_LOCAL_COPY_20260615_090305/fa2_vs_fa3_ext_ncu_cycles_time.csv
```

## H100 NCU Collection

For FA2 full NCU collection, build a CUDA 12.8 prebuilt bundle locally and move
the tarball to the H100 machine:

```bash
cd test
CUDA_INSTALL_PATH=/usr/local/cuda-12.8 JOBS=8 ./scripts/prepare_fa2_full_ncu_prebuilt.sh
cd ..
```

On H100:

```bash
tar -xzf fa2_full_ncu_prebuilt_cuda128.tar.gz
cd prebuilt
CUDA_INSTALL_PATH=/usr/local/cuda-12.8 NCU_SET=full ./run_remote.sh
```

To collect only selected cases:

```bash
SELECT_CASES="H32D64FullB32S256 H16D128FullB32S256" ./run_remote.sh
```

For FA3 extended H1D128 profiling on H100:

```bash
cd test
CUDA_INSTALL_PATH=/usr/local/cuda-12.8 \
FA3_CASES="H16D128FullB64S512 H16D128FullB32S1024 H16D128FullB16S2048 H16D128FullB8S4096" \
NCU_SET=full \
./scripts/run_fa3_extended_n176_h100.sh
```

For the FA3 WGMMA extended script:

```bash
cd test
CUDA_INSTALL_PATH=/usr/local/cuda-12.8 \
FA3_CASES="H1D128B1S4096" \
NCU_SET=full \
./scripts/run_fa3_extended_wgmma_h100.sh
```

## Notes

- `GPGPUSIM_PTX_DEBUG=1` enables verbose PTX extraction/debug prints.
- `-gpgpu_ptx_reorder 1` dumps reordered PTX files under `sass_ptxline/`.
  With `-gpgpu_ptx_reorder_sass_guided 1`, the simulator also auto-extracts
  full SASS PTX-line info into the same directory and uses it as the guide.
- Keep generated `test/run/*`, microbenchmark binaries, and temporary config
  sweep directories out of commits.
