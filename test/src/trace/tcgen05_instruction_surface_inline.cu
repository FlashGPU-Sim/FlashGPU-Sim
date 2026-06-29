// Canonical inline-PTX source for test/src/trace/ptx/
// tcgen05_instruction_surface_smoke.ptx.
//
// This file intentionally is not part of the default test build. It requires a
// Blackwell-capable CUDA toolchain, and the checked-in PTX is the parser smoke
// input used by normal CI/developer builds.

#include <cstdint>

extern "C" __global__ void tcgen05_instruction_surface_inline(uint32_t *sink) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  __shared__ uint32_t barrier_slot[8];
  uint32_t barrier_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier_slot));
  uint32_t taddr = 0;
  uint32_t ncols = 512;
  uint64_t sdesc = 0;
  uint32_t word = 0;

  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 "
               "[%0], %1;\n" : : "r"(barrier_addr), "r"(ncols));

  asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;\n"
               :
               : "r"(taddr), "l"(sdesc));
  asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;\n"
               :
               : "r"(taddr), "l"(sdesc));
  asm volatile("tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;\n"
               :
               : "r"(taddr), "l"(sdesc));
  asm volatile("tcgen05.cp.cta_group::1.64x128b.warpx2::01_23 [%0], %1;\n"
               :
               : "r"(taddr), "l"(sdesc));
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n"
               :
               : "r"(taddr), "l"(sdesc));
  asm volatile("tcgen05.cp.cta_group::1.4x256b [%0], %1;\n"
               :
               : "r"(taddr), "l"(sdesc));

  asm volatile("tcgen05.shift.cta_group::1.down [%0];\n" : : "r"(taddr));
  asm volatile("tcgen05.fence::before_thread_sync;\n");
  asm volatile("tcgen05.fence::after_thread_sync;\n");

  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one."
               "shared::cluster.b64 [%0];\n"
               :
               : "r"(barrier_addr));
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n"
               : "=r"(word) : "r"(taddr));
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.pack::16b.b32 {%0}, [%1];\n"
               : "=r"(word) : "r"(taddr));
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n"
               : : "r"(taddr), "r"(word));
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.unpack::16b.b32 [%0], {%1};\n"
               : : "r"(taddr), "r"(word));
  asm volatile("tcgen05.wait::ld.sync.aligned;\n");
  asm volatile("tcgen05.wait::st.sync.aligned;\n");

  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
               : : "r"(taddr), "r"(ncols));
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
  if (threadIdx.x == 0) sink[0] = word;
#else
  if (threadIdx.x == 0) sink[0] = 0;
#endif
}
