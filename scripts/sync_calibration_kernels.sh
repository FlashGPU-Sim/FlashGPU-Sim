#!/usr/bin/env bash
# Populate the git-ignored calibration/ tree. Safe to re-run; does not touch
# git-tracked files.
#
# Always fetches public dsm_bw / tma_bw snapshots from GitHub at pinned
# commits. Optionally copies sibling checkouts if present:
#   H200_PROFILING_DIR   default: <repo>/../H200_profiling
#   HOPPER_BENCH_DIR     default: <repo>/../NVIDIA-Hopper-Benchmark
# Missing siblings are skipped. Nothing here hard-codes a home directory.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '2,9p' "$0" | sed 's/^# //'
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/calibration"
H200_SRC="${H200_PROFILING_DIR:-${ROOT}/../H200_profiling}"
HOPPER_SRC="${HOPPER_BENCH_DIR:-${ROOT}/../NVIDIA-Hopper-Benchmark}"
H200_PROFILING_REV="${H200_PROFILING_REV:-c880781eb1bb5a2864308f8ca33d0dfd8673fab4}"
DSM_BW_COMMIT="${DSM_BW_COMMIT:-4e8c4f91dd7b00584efcb3ac4b602b33ce2631cd}"
TMA_BW_COMMIT="${TMA_BW_COMMIT:-4e8c4f91dd7b00584efcb3ac4b602b33ce2631cd}"
DSM_BW_URL="${DSM_BW_URL:-https://github.com/seanzw/random.git}"

mkdir -p "${DEST}/kernels" "${DEST}/results" "${DEST}/runs"

copy_if_present() {
  local src="$1"
  local dst="$2"
  shift 2
  if [[ ! -d "${src}" ]]; then
    echo "WARN: skip missing ${src}"
    return 0
  fi
  mkdir -p "${dst}"
  rsync -a --delete "$@" "${src}/" "${dst}/"
  echo "copied ${src} -> ${dst}"
}

if [[ -d "${H200_SRC}/.git" ]]; then
  actual_rev="$(git -C "${H200_SRC}" rev-parse HEAD)"
  [[ "${actual_rev}" == "${H200_PROFILING_REV}" ]] || {
    echo "ERROR: H200_profiling is ${actual_rev}; expected ${H200_PROFILING_REV}. Set H200_PROFILING_REV deliberately to accept a new snapshot." >&2
    exit 1
  }
  git -C "${H200_SRC}" diff --quiet "${H200_PROFILING_REV}" -- || {
    echo "ERROR: tracked H200_profiling files differ from ${H200_PROFILING_REV}" >&2
    exit 1
  }
fi

copy_if_present "${H200_SRC}" "${DEST}/kernels/h200_probes" \
  --exclude 'output-*' \
  --exclude 'error-*' \
  --exclude 'python/.venv' \
  --exclude 'python/__pycache__' \
  --exclude '__pycache__' \
  --exclude 'nvprof' \
  --exclude 'h200_latency_suite' \
  --exclude '*.o' \
  --exclude '.git'

# Keep simulator-only single-case compilation guards reproducible. Without
# FLASHGPU_SIM_REPRESENTATIVE these remain exact vendor behavior.
if [[ -d "${DEST}/kernels/h200_probes/vendor/microbench/flashgpu_sim/wgmma" ]]; then
  for src in wgmma_rf_bandwidth_bench.cc wgmma_softmax_mix_bench.cc; do
    rsync -a "${ROOT}/test/src/microbench/wgmma/${src}" \
      "${DEST}/kernels/h200_probes/vendor/microbench/flashgpu_sim/wgmma/${src}"
  done
fi

if [[ -d "${HOPPER_SRC}/NewFeatures" ]]; then
  mkdir -p "${DEST}/kernels/hopper_paper"
  rsync -a --delete \
    "${HOPPER_SRC}/NewFeatures/DSM/" "${DEST}/kernels/hopper_paper/DSM/"
  rsync -a --delete \
    "${HOPPER_SRC}/NewFeatures/TMA/" "${DEST}/kernels/hopper_paper/TMA/"
  echo "copied Hopper-paper DSM/TMA kernels"
else
  echo "WARN: skip missing ${HOPPER_SRC}/NewFeatures"
fi

# seanzw/random: copy both vendor kernel sets verbatim at fixed revisions.
if command -v git >/dev/null 2>&1; then
  tmp="$(mktemp -d)"
  git clone --filter=blob:none --sparse "${DSM_BW_URL}" "${tmp}/random" >/dev/null
  git -C "${tmp}/random" sparse-checkout set dsm_bw tma_bw

  git -C "${tmp}/random" fetch --depth 1 origin "${DSM_BW_COMMIT}" "${TMA_BW_COMMIT}" >/dev/null
  git -C "${tmp}/random" checkout "${DSM_BW_COMMIT}" -- dsm_bw
  rm -rf "${DEST}/kernels/dsm_bw"
  mkdir -p "${DEST}/kernels/dsm_bw"
  rsync -a "${tmp}/random/dsm_bw/" "${DEST}/kernels/dsm_bw/"
  echo "copied dsm_bw @ ${DSM_BW_COMMIT} (verbatim; do not rewrite)"

  git -C "${tmp}/random" checkout "${TMA_BW_COMMIT}" -- tma_bw
  rm -rf "${DEST}/kernels/tma_bw"
  mkdir -p "${DEST}/kernels/tma_bw"
  rsync -a "${tmp}/random/tma_bw/" "${DEST}/kernels/tma_bw/"
  echo "copied tma_bw @ ${TMA_BW_COMMIT} (verbatim; GMEM TMA/L2/HBM, not DSM)"

  rm -rf "${tmp}"
else
  echo "WARN: git not available; cannot fetch seanzw/random"
fi

cat > "${DEST}/kernels/h200_probes/SOURCE_MANIFEST.json" <<EOF
{
  "h200_profiling_revision": "${H200_PROFILING_REV}",
  "dsm_bw_revision": "${DSM_BW_COMMIT}",
  "tma_bw_revision": "${TMA_BW_COMMIT}"
}
EOF

cat > "${DEST}/README.md" <<'EOF'
# Local calibration tree (git-ignored)

This directory is **not** part of the FlashGPU-Sim git tree. It holds optional
kernel snapshots and local simulator runs for H200 CLUSTER132 calibration.

Status: docs/cluster_noc/calibration.md

Refresh copies:

    bash scripts/sync_calibration_kernels.sh

Optional sibling trees (override with H200_PROFILING_DIR / HOPPER_BENCH_DIR):

    ../H200_profiling
    ../NVIDIA-Hopper-Benchmark

Public dsm_bw / tma_bw snapshots are fetched from GitHub. Do not commit
anything under calibration/.
EOF

echo "calibration tree ready under ${DEST}"
find "${DEST}/kernels" -maxdepth 2 -type d | sort
