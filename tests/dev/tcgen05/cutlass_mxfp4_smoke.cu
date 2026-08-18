// CUTLASS strict-MXFP4 integration smoke for the SM100 TCGen05 path.
//
// CUTLASS_ROOT must point at an NVIDIA CUTLASS checkout.  This intentionally
// uses the CUTLASS testbed and block-scaled collective builder instead of a
// hand-written traffic proxy.

#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "gemm_testbed_3x_ptr_array.hpp"

using namespace cute;

#if !defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#error "CUTLASS SM100 MMA support was not enabled"
#endif

template <int TileN>
bool run_strict_1sm_mxfp4() {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementA = cutlass::float_e2m1_t;
  using ElementB = cutlass::float_e2m1_t;
  using ElementC = float;
  using ElementD = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementSF = cutlass::float_ue8m0_t;
  using MmaTypePairA = cute::tuple<ElementA, ElementSF>;
  using MmaTypePairB = cute::tuple<ElementB, ElementSF>;

  using MmaTileShape = Shape<_128, Int<TileN>, _256>;
  using ClusterShape = Shape<_1, _1, _1>;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm;
  using MainloopSchedule =
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf4Sm100;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape,
          ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute, ElementC, LayoutC, 4, ElementD,
          LayoutC, 4, EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp,
          MmaTypePairA, LayoutA, 32, MmaTypePairB, LayoutB, 32,
          ElementAccumulator, MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cutlass::gemm::ArrayProblemShape<Shape<int, int, int, int>>,
      CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using ProblemShape = typename GemmKernel::ProblemShape;

  test::gemm::device::Testbed3x<Gemm> testbed;
  ProblemShape problem{{128, TileN, 256, 1}};
  return testbed.run(problem, 1.0f, 0.0f,
                     test::gemm::device::detail::Iterations(1));
}

TEST(FlashGpuSimCutlassMxfp4, Strict1Sm128x64x256) {
  EXPECT_TRUE(run_strict_1sm_mxfp4<64>());
}

TEST(FlashGpuSimCutlassMxfp4, Strict1Sm128x128x256) {
  EXPECT_TRUE(run_strict_1sm_mxfp4<128>());
}

TEST(FlashGpuSimCutlassMxfp4, Strict1Sm128x256x256) {
  EXPECT_TRUE(run_strict_1sm_mxfp4<256>());
}
