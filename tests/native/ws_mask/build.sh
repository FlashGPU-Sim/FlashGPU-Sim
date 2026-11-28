#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
: "${OUT:?Set OUT to a new package directory}"
: "${CUTLASS_ROOT:?Set CUTLASS_ROOT to a pinned CUTLASS checkout}"
: "${CUDA_INSTALL_PATH:=/usr/local/cuda-13.3}"
: "${GTEST_ROOT:?Set GTEST_ROOT to googletest source directory}"
[[ ! -e "$OUT" ]] || { echo 'OUT must not exist' >&2; exit 2; }
mkdir -p "$OUT"/{bin,lib,src,artifacts}
cp "$here"/{numerics.cu,timing.cu,build.sh,README.md} "$OUT/src/"
cp "$here/collect.py" "$OUT/"
git -C "$CUTLASS_ROOT" rev-parse HEAD > "$OUT/artifacts/cutlass-commit.txt"
git -C "$CUTLASS_ROOT" diff HEAD --binary > "$OUT/artifacts/cutlass.patch"
"$CUDA_INSTALL_PATH/bin/nvcc" --version > "$OUT/artifacts/nvcc-version.txt"
g++ -O2 -std=c++17 -pthread -I"$GTEST_ROOT/include" -I"$GTEST_ROOT" -c "$GTEST_ROOT/src/gtest-all.cc" -o "$OUT/artifacts/gtest-all.o"
g++ -O2 -std=c++17 -pthread -I"$GTEST_ROOT/include" -c "$GTEST_ROOT/src/gtest_main.cc" -o "$OUT/artifacts/gtest-main.o"
flags=(-std=c++17 --expt-relaxed-constexpr -I"$CUTLASS_ROOT/include" -O3 -lineinfo -gencode arch=compute_100a,code=sm_100a -gencode arch=compute_100a,code=compute_100a -cudart shared)
"$CUDA_INSTALL_PATH/bin/nvcc" "${flags[@]}" -I"$GTEST_ROOT/include" "$OUT/src/numerics.cu" "$OUT/artifacts/gtest-all.o" "$OUT/artifacts/gtest-main.o" -Xcompiler -pthread -o "$OUT/bin/ws_numerics" > "$OUT/artifacts/numerics-build.log" 2>&1
"$CUDA_INSTALL_PATH/bin/nvcc" "${flags[@]}" "$OUT/src/timing.cu" -o "$OUT/bin/ws_timing" > "$OUT/artifacts/timing-build.log" 2>&1
cp -L "$CUDA_INSTALL_PATH/lib64/libcudart.so.13" "$OUT/lib/libcudart.so.13"
ln -s libcudart.so.13 "$OUT/lib/libcudart.so"
for name in ws_numerics ws_timing; do
  "$CUDA_INSTALL_PATH/bin/cuobjdump" --dump-sass "$OUT/bin/$name" > "$OUT/artifacts/$name.sass"
  "$CUDA_INSTALL_PATH/bin/cuobjdump" --dump-resource-usage "$OUT/bin/$name" > "$OUT/artifacts/$name.resources"
  LD_LIBRARY_PATH="$OUT/lib" ldd "$OUT/bin/$name" > "$OUT/artifacts/$name.ldd"
done
(cd "$OUT" && sha256sum bin/* lib/libcudart.so.13 src/* collect.py > artifacts/SHA256SUMS)
python3 "$OUT/collect.py" --dry-run > "$OUT/artifacts/collection-plan.json"
