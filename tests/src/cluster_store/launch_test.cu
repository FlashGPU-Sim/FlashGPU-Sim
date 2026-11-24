#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>
namespace {
__host__ __device__ constexpr unsigned words(unsigned form) {
  return form == 0                                            ? 1
         : (form == 3 || form == 4 || form == 6 || form == 9) ? 2
                                                              : 4;
}
// Two independent clusters, two phases, peer/local/mixed ownership and a
// predicated nonzero lane subset. Every receiver checks payload plus sentinels.
template <unsigned Form, unsigned Mode = 0, bool Generic = false,
          bool Partial = false>
__global__ __cluster_dims__(2, 1, 1) void store(unsigned *out) {
  __shared__ __align__(16) unsigned data[128];
  __shared__ __align__(8) unsigned long long barrier;
  const unsigned t = threadIdx.x, rank = blockIdx.x % 2;
  const unsigned bar = (unsigned)__cvta_generic_to_shared(&barrier);
  const unsigned base = (unsigned)__cvta_generic_to_shared(data);
  for (unsigned j = 0; j < 4; ++j) data[t * 4 + j] = 0xabcd;
  if (!t)
    asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" ::"r"(bar) : "memory");
  __syncthreads();
  for (unsigned round = 0; round < 2; ++round) {
    if (!t)
      asm volatile(
          "mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" ::"r"(bar),
          "r"((Partial ? 8 : 32) * 4 * words(Form))
          : "memory");
    asm volatile(
        "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::
            : "memory");
    const unsigned target = Mode == 1   ? rank
                            : Mode == 2 ? (rank ^ (t & 1))
                                        : rank ^ 1;
    unsigned dest, remote_bar;
    unsigned long long generic_dest = 0, generic_bar = 0;
    if constexpr (Generic) {
      asm volatile("mapa.u64 %0,%1,%2;"
                   : "=l"(generic_dest)
                   : "l"(data + t * 4), "r"(target));
      asm volatile("mapa.u64 %0,%1,%2;"
                   : "=l"(generic_bar)
                   : "l"(&barrier), "r"(target));
    } else {
      asm volatile("mapa.shared::cluster.u32 %0,%1,%2;"
                   : "=r"(dest)
                   : "r"(base + t * 16), "r"(target));
      asm volatile("mapa.shared::cluster.u32 %0,%1,%2;"
                   : "=r"(remote_bar)
                   : "r"(bar), "r"(target));
    }
    const unsigned v = (blockIdx.x + 1) * 1000 + round * 100 + t * 4;
    if (!Partial || (t & 3) == 1) {
      if constexpr (Generic) {
        if constexpr (Form == 0)
          asm volatile(
              "st.async.weak.mbarrier::complete_tx::bytes.b32 [%0],%1,[%2];" ::
                  "l"(generic_dest),
              "r"(v), "l"(generic_bar)
              : "memory");
        else
          asm volatile(
              "{ .reg .b128 value; mov.b128 value,{%1,%2,%3,%4}; "
              "st.async.mbarrier::complete_tx::bytes.b128 [%0],value,[%5]; }" ::
                  "l"(generic_dest),
              "r"(v), "r"(v + 1), "r"(v + 2), "r"(v + 3), "l"(generic_bar)
              : "memory");
      } else {
        if constexpr (Form == 0)
          asm volatile(
              "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32 "
              "[%0],%1,[%2];" ::"r"(dest),
              "r"(v), "r"(remote_bar)
              : "memory");
        if constexpr (Form == 1)
          asm volatile(
              "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4 "
              "[%0],{%1,%2,%3,%4},[%5];" ::"r"(dest),
              "r"(v), "r"(v + 1), "r"(v + 2), "r"(v + 3), "r"(remote_bar)
              : "memory");
        if constexpr (Form == 2)
          asm volatile(
              "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.u64 "
              "[%0],{%1,%2},[%3];" ::"r"(dest),
              "l"((unsigned long long)(v + 1) << 32 | v),
              "l"((unsigned long long)(v + 3) << 32 | (v + 2)), "r"(remote_bar)
              : "memory");
        if constexpr (Form == 3)
          asm volatile(
              "st.async.shared::cluster.mbarrier::complete_tx::bytes.u64 "
              "[%0],%1,[%2];" ::"r"(dest),
              "l"((unsigned long long)(v + 1) << 32 | v), "r"(remote_bar)
              : "memory");
        if constexpr (Form == 4)
          asm volatile(
              "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.u32 "
              "[%0],{%1,%2},[%3];" ::"r"(dest),
              "r"(v), "r"(v + 1), "r"(remote_bar)
              : "memory");
        if constexpr (Form == 5)
          asm volatile(
              "{ .reg .b128 value; mov.b128 value,{%1,%2,%3,%4}; "
              "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b128 "
              "[%0],value,[%5]; }" ::"r"(dest),
              "r"(v), "r"(v + 1), "r"(v + 2), "r"(v + 3), "r"(remote_bar)
              : "memory");
        if constexpr (Form == 6)
          asm volatile(
              "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.s32 "
              "[%0],{%1,%2},[%3];" ::"r"(dest),
              "r"(v), "r"(v + 1), "r"(remote_bar)
              : "memory");
        if constexpr (Form == 7)
          asm volatile(
              "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 "
              "[%0],{%1,%2,%3,%4},[%5];" ::"r"(dest),
              "f"(__uint_as_float(v)), "f"(__uint_as_float(v + 1)),
              "f"(__uint_as_float(v + 2)), "f"(__uint_as_float(v + 3)),
              "r"(remote_bar)
              : "memory");
        if constexpr (Form == 8)
          asm volatile(
              "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f64 "
              "[%0],{%1,%2},[%3];" ::"r"(dest),
              "d"(__longlong_as_double((unsigned long long)(v + 1) << 32 | v)),
              "d"(__longlong_as_double((unsigned long long)(v + 3) << 32 |
                                       (v + 2))),
              "r"(remote_bar)
              : "memory");
        if constexpr (Form == 9)
          asm volatile(
              "st.async.shared::cluster.mbarrier::complete_tx::bytes.s64 "
              "[%0],%1,[%2];" ::"r"(dest),
              "l"((unsigned long long)(v + 1) << 32 | v), "r"(remote_bar)
              : "memory");
      }
    }
    asm volatile(
        "{ .reg .pred p; wait: mbarrier.try_wait.parity.shared::cta.b64 "
        "p,[%0],%1,10000000; @!p bra wait; }" ::"r"(bar),
        "r"(round)
        : "memory");
    for (unsigned j = 0; j < 4; ++j)
      out[(round * 4 + blockIdx.x) * 128 + t * 4 + j] = data[t * 4 + j];
    asm volatile(
        "barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::
            : "memory");
  }
}
template <unsigned Form, unsigned Mode = 0, bool Generic = false,
          bool Partial = false>
void check() {
  unsigned *dst;
  ASSERT_EQ(cudaMalloc(&dst, 1024 * 4), cudaSuccess);
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(4);
  config.blockDim = dim3(32);
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {2, 1, 1};
  config.attrs = &attr;
  config.numAttrs = 1;
  ASSERT_EQ(
      cudaLaunchKernelEx(&config, store<Form, Mode, Generic, Partial>, dst),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<unsigned> out(1024);
  ASSERT_EQ(cudaMemcpy(out.data(), dst, 4096, cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned round = 0; round < 2; ++round)
    for (unsigned block = 0; block < 4; ++block)
      for (unsigned i = 0; i < 128; ++i) {
        const unsigned sender = Mode == 1   ? block
                                : Mode == 2 ? block ^ ((i / 4) & 1)
                                            : block ^ 1;
        const bool active = !Partial || ((i / 4) & 3) == 1;
        const unsigned expected = !active || i % 4 >= words(Form)
                                      ? 0xabcd
                                      : (sender + 1) * 1000 + round * 100 + i;
        ASSERT_EQ(out[(round * 4 + block) * 128 + i], expected)
            << Form << ":" << Mode << ":" << Generic << ":" << round << ":"
            << block << ":" << i;
      }
  EXPECT_EQ(cudaFree(dst), cudaSuccess);
}
TEST(ClusterStore, WidthsTypesAndBothVectorOrders) {
  check<0>();
  check<1>();
  check<2>();
  check<3>();
  check<4>();
  check<5>();
  check<6>();
  check<7>();
  check<8>();
  check<9>();
}
TEST(ClusterStore, LocalMixedAndPredicatedLaneCompletion) {
  check<1, 1>();
  check<1, 2>();
  check<1, 2, false, true>();
}
TEST(ClusterStore, GenericAddressesAndExplicitWeak) {
  check<0, 0, true>();
  check<5, 0, true>();
  check<5, 1, true>();
  check<5, 2, true, true>();
}
}  // namespace
