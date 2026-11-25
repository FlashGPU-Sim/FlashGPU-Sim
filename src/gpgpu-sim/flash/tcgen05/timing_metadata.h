#pragma once
#include "descriptor.h"
#include <algorithm>
#include <vector>

namespace flash_gpgpu_sim {
enum class tcgen05_mma_kind { half, tf32, fp8, mixed, fp4 };
enum class tcgen05_native_profile : unsigned {
  unknown = 0,
  bf16_f32 = 1,
  tf32_f32 = 2,
  e4m3_f32 = 3,
  e2m1_e4m3_f32_block32 = 4,
  e2m1_f32_block32 = 5
};
struct tcgen05_mma_metadata {
  tcgen05_mma_kind kind = tcgen05_mma_kind::half;
  tcgen05_native_profile profile = tcgen05_native_profile::unknown;
  unsigned m = 0, n = 0, k = 0, group = 1;
  bool tmem_a = false, ws = false;
  unsigned shared_bytes[2][2] = {}; // peer rank, 128KiB address region
};
// Profile lookup is deliberately exact. Different accumulators, transpose,
// scale formats and unmeasured types never alias the measured BF16 family.
inline tcgen05_mma_metadata
tcgen05_metadata(tcgen05_mma_kind kind, const tcgen05_mma_descriptor_t &d,
                 unsigned group = 1, bool tmem_a = false, bool ws = false,
                 bool collector_reuse = false) {
  tcgen05_mma_metadata r;
  r.kind = kind;
  r.m = d.m;
  r.n = d.n;
  r.k = d.k;
  r.group = group;
  r.tmem_a = tmem_a;
  r.ws = ws;
  if (d.sparse || d.transpose_a || d.transpose_b || d.d_type != 1 ||
      collector_reuse)
    return r;
  using P = tcgen05_native_profile;
  if (kind == tcgen05_mma_kind::half && d.a_type == 1 && d.b_type == 1)
    r.profile = P::bf16_f32;
  if (kind == tcgen05_mma_kind::tf32)
    r.profile = P::tf32_f32;
  if (kind == tcgen05_mma_kind::fp8 && d.a_type == 0 && d.b_type == 0)
    r.profile = P::e4m3_f32;
  if (kind == tcgen05_mma_kind::mixed && d.a_type == 5 && d.b_type == 0 &&
      d.scale_format == 1)
    r.profile = P::e2m1_e4m3_f32_block32;
  if (kind == tcgen05_mma_kind::fp4 && d.scale_format == 1)
    r.profile = P::e2m1_f32_block32;
  return r;
}
struct tcgen05_shared_read {
  uint32_t address;
  unsigned bytes, rank;
};
// Shared by frontend operand gatherers; adjacent packed-element reads of
// the same physical byte are charged once within an operand.
struct tcgen05_shared_reads {
  std::vector<tcgen05_shared_read> reads;
  bool new_operand = true;
  void begin_operand() { new_operand = true; }
  void add(uint32_t address, unsigned bytes, unsigned rank = 0) {
    if (!new_operand && !reads.empty()) {
      auto &last = reads.back();
      if (last.rank == rank && address >= last.address &&
          uint64_t(address) <= uint64_t(last.address) + last.bytes) {
        last.bytes = std::max(last.bytes, address - last.address + bytes);
        return;
      }
    }
    new_operand = false;
    reads.push_back({address, bytes, rank});
  }
};
inline void
tcgen05_account_shared(tcgen05_mma_metadata &out,
                       const std::vector<tcgen05_shared_read> &reads) {
  for (const auto &read : reads) {
    auto address = read.address;
    auto bytes = read.bytes;
    while (bytes) {
      constexpr unsigned half = 128 * 1024;
      const auto part = std::min(bytes, half - address % half);
      out.shared_bytes[read.rank][(address / half) % 2] += part;
      address += part;
      bytes -= part;
    }
  }
}
unsigned tcgen05_b200_recurrence(tcgen05_native_profile, unsigned m, unsigned n,
                                 unsigned k, unsigned group, bool tmem_a,
                                 bool ws);
} // namespace flash_gpgpu_sim
