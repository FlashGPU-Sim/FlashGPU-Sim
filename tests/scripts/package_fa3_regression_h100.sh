#!/usr/bin/env bash
# Build only the four forward specializations. No clock64/sensitivity variants.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ $# -eq 1 ]] || { echo "Usage: $0 NEW_PACKAGE_DIRECTORY" >&2; exit 2; }
OUT="$(realpath -m "$1")"
[[ ! -e "$OUT" ]] || { echo "Output already exists: $OUT" >&2; exit 2; }
CUDA_INSTALL_PATH="${CUDA_INSTALL_PATH:-/usr/local/cuda-13.3}"
export CUDA_INSTALL_PATH
export PATH="$CUDA_INSTALL_PATH/bin:$PATH"
mkdir -p "$OUT/bin" "$OUT/lib"
# Separate dependency/build directories avoid modifying an experimental checkout.
make -C "$ROOT/tests" -j"${JOBS:-4}" fa3-forward \
  BUILD_DIR="$OUT/build" FLASH_ATTENTION_DIR="$OUT/dependency" \
  NVCC="$CUDA_INSTALL_PATH/bin/nvcc"
cp "$OUT/build/bin/sm90/fa3/forward_tests" "$OUT/bin/standard_tests"
cp -L "$CUDA_INSTALL_PATH/lib64/libcudart.so.13" "$OUT/lib/"
cp "$ROOT/tests/scripts/run_fa3_standard_prebuilt_h100.sh" "$OUT/collect.sh"
cp "$ROOT/tests/scripts/summarize_fa3_standard_ncu.py" "$OUT/"
git -C "$ROOT" rev-parse HEAD > "$OUT/source-commit.txt"
git -C "$ROOT" diff --binary HEAD > "$OUT/source-changes.patch"
"$CUDA_INSTALL_PATH/bin/nvcc" --version > "$OUT/compiler.txt"
(cd "$OUT" && sha256sum bin/standard_tests lib/libcudart.so.13 \
  collect.sh summarize_fa3_standard_ncu.py source-commit.txt source-changes.patch compiler.txt > checksums.sha256)
echo "Package: $OUT (transfer bin/, lib/, collect.sh, summarize_fa3_standard_ncu.py and provenance files)"
