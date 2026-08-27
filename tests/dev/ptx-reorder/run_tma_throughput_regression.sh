#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
bench_dir="$repo_root/tests/src/microbench/tma"
config_dir="$repo_root/configs/SM100_B200"
run_root=$(mktemp -d "${TMPDIR:-/tmp}/ptx-reorder-tma.XXXXXX")
trap 'rm -rf -- "$run_root"' EXIT

make -C "$bench_dir" ARCH=sm_100a PTX_PROFILE=compute_100a throughput
binary="$repo_root/tests/build/bin/microbench/tma/tma_throughput_bench_sm_100a"

run_case() {
  local reorder=$1
  local run_dir="$run_root/r${reorder}"
  mkdir -p "$run_dir"
  cp "$binary" "$config_dir/gpgpusim.config" \
    "$config_dir/config_ampere_islip.icnt" "$run_dir/"
  sed -i -E \
    -e "s/^-gpgpu_ptx_reorder[[:space:]]+.*/-gpgpu_ptx_reorder $reorder/" \
    -e 's/^-gpgpu_clock_domains[[:space:]]+.*/-gpgpu_clock_domains 1965:1965:1155:3996/' \
    -e 's/^-gpgpu_tma_request_granularity[[:space:]]+.*/-gpgpu_tma_request_granularity 32/' \
    -e 's/^-gpgpu_tma_request_width[[:space:]]+.*/-gpgpu_tma_request_width 4/' \
    -e 's/^-gpgpu_tma_response_width[[:space:]]+.*/-gpgpu_tma_response_width 4/' \
    -e 's/^-gpgpu_tma_max_inflight[[:space:]]+.*/-gpgpu_tma_max_inflight 3200/' \
    -e 's/^-gpgpu_tma_tx_quota[[:space:]]+.*/-gpgpu_tma_tx_quota 48/' \
    -e 's/^-dram_latency[[:space:]]+.*/-dram_latency 393/' \
    -e 's/^-gpgpu_simple_dram_model[[:space:]]+.*/-gpgpu_simple_dram_model 1/' \
    "$run_dir/gpgpusim.config"
  (cd "$run_dir" && ./tma_throughput_bench_sm_100a > simulation.log 2>&1)
  grep -q 'tma_benchmark_status = PASS' "$run_dir/simulation.log"
  awk '/tma_benchmark_bytes_per_cycle =/{print $3}' "$run_dir/simulation.log"
}

r0=$(run_case 0)
r1=$(run_case 1)
awk -v r0="$r0" -v r1="$r1" 'BEGIN {
  printf "PTX reorder TMA throughput: r0=%.6f r1=%.6f\n", r0, r1
  if (r1 < r0 * 0.99) exit 1
}'
