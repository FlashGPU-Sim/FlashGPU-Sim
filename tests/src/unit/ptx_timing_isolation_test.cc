#include <gtest/gtest.h>

#include "../../../libcuda/gpgpu_context.h"
#include "gpgpu-sim/gpu-sim.h"
#include "option_parser.h"

namespace {
class SharedTestConfig : public core_config {
 public:
  explicit SharedTestConfig(gpgpu_context *context) : core_config(context) {
    warp_size = num_shmem_bank = 32;
    mem_warp_parts = 1;
    m_valid = true;
  }
  void init() override {}
};

TEST(PtxTimingIsolationTest, SharedServicePreservesPublicModelUnlessSelected) {
  auto *ctx = GPGPU_Context();
  SharedTestConfig config(ctx);
  // Frozen public dispatch counts and the separate native physical policy.
  struct Case { unsigned bytes, elements, stride, ptx_cycles, physical_cycles; };
  const Case cases[] = {
      {4, 1, 0, 1, 1},
      {4, 1, 4, 1, 1},
      {4, 1, 16, 4, 4},
      {4, 1, 32, 8, 8},
      {4, 2, 0, 1, 2},
      {4, 2, 4, 2, 2},
      {4, 2, 16, 4, 4},
      {4, 2, 32, 8, 8},
      {4, 4, 0, 1, 4},
      {4, 4, 4, 2, 4},
      {4, 4, 16, 4, 4},
      {4, 4, 32, 8, 8},
      {8, 1, 0, 1, 2},
      {8, 1, 4, 1, 2},
      {8, 1, 16, 4, 4},
      {8, 1, 32, 8, 8},
      {8, 2, 0, 1, 4},
      {8, 2, 4, 2, 4},
      {8, 2, 16, 4, 4},
      {8, 2, 32, 8, 8},
      {8, 4, 0, 1, 8},
      {8, 4, 4, 2, 8},
      {8, 4, 16, 5, 8},
      {8, 4, 32, 8, 8},
  };
  for (const auto &c : cases) for (bool physical : {false, true})
    for (bool store : {false, true}) {
      SCOPED_TRACE(::testing::Message() << c.bytes << "x" << c.elements
                   << " stride=" << c.stride << " physical=" << physical
                   << " store=" << store);
      warp_inst_t inst(&config);
      inst.pc = 0;
      inst.op = store ? STORE_OP : LOAD_OP;
      inst.memory_op = store ? memory_store : memory_load;
      inst.space = memory_space_t(shared_space);
      inst.data_size = c.bytes;
      inst.vector_elements = c.elements;
      if (physical) inst.set_physical_shared_transactions();
      active_mask_t mask; mask.set();
      inst.issue(mask, 0, 0, 0, 0, 0);
      for (unsigned lane = 0; lane < 32; ++lane) {
        inst.set_addr(lane, lane * c.stride);
        inst.set_per_thread_memory_access_size(lane, c.bytes * c.elements);
      }
      inst.generate_mem_accesses();
      unsigned cycles = 1;
      while (inst.dispatch_delay()) ++cycles;
      EXPECT_EQ(cycles, physical ? c.physical_cycles : c.ptx_cycles);
      if (!physical) {
        EXPECT_TRUE(inst.shared_dispatch_delay_is_bank_conflict());
      }
    }
}

TEST(PtxTimingIsolationTest, TmaOobTrafficDefaultsOnAndRemainsConfigurable) {
  shader_core_config config(GPGPU_Context());
  option_parser_t options = option_parser_create();
  config.reg_options(options);
  EXPECT_TRUE(config.gpgpu_tma_oob_l2_traffic);
  const char *off[] = {"test", "-gpgpu_tma_oob_l2_traffic", "0"};
  option_parser_cmdline(options, 3, off);
  EXPECT_FALSE(config.gpgpu_tma_oob_l2_traffic);
  const char *on[] = {"test", "-gpgpu_tma_oob_l2_traffic", "1"};
  option_parser_cmdline(options, 3, on);
  EXPECT_TRUE(config.gpgpu_tma_oob_l2_traffic);
  option_parser_destroy(options);
}
TEST(PtxTimingIsolationTest, WgmmaRfPriorityPreservesPublicDefault) {
  shader_core_config config(GPGPU_Context());
  option_parser_t options = option_parser_create();
  config.reg_options(options);
  EXPECT_FALSE(config.gpgpu_wgmma_rf_ordinary_read_priority);
  const char *on[] = {"test", "-gpgpu_wgmma_rf_ordinary_read_priority", "1"};
  option_parser_cmdline(options, 3, on);
  EXPECT_TRUE(config.gpgpu_wgmma_rf_ordinary_read_priority);
  const char *off[] = {"test", "-gpgpu_wgmma_rf_ordinary_read_priority", "0"};
  option_parser_cmdline(options, 3, off);
  EXPECT_FALSE(config.gpgpu_wgmma_rf_ordinary_read_priority);
  option_parser_destroy(options);
}
}  // namespace
