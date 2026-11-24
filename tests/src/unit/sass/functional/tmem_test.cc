#include "../test_support.h"
#include "gpgpu-sim/flash/sass/functional/tmem.h"
namespace {
TEST(SassFunctionalTest, NativeTmemStore32WritesEveryWord) {
  kernel image;
  image.name = "store32";
  image.arch = architecture::kSm103;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "STTM.x32", "tmem[UR0],R8")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm100_functional_semantics(sass);
  cta_execution_state cta(4);
  cta.tcgen = std::make_shared<flash_gpgpu_sim::sass::functional_tcgen_state>();
  cta.tcgen->allocated = 1;
  execution_context context;
  context.cta = &cta;
  warp_state state;
  for (unsigned lane = 0; lane < 32; ++lane)
    for (unsigned word = 0; word < 32; ++word)
      state.write_register(lane, 8 + word, 1000 + lane * 32 + word);
  ASSERT_EQ(sass.step("store32", state, context).status,
            step_status::kAdvanced);
  const auto words = cta.tcgen->memory.read_matrix_words({0, 0}, 0, 32, 32);
  for (unsigned i = 0; i < words.size(); ++i)
    EXPECT_EQ(words[i], 1000 + i) << "word " << i;
}

TEST(SassFunctionalTest, NativeTmemStore64WritesEveryWord) {
  kernel image;
  image.name = "store64";
  image.arch = architecture::kSm103;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "STTM.x64", "tmem[UR0],R8")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm100_functional_semantics(sass);
  cta_execution_state cta(4);
  cta.tcgen = std::make_shared<flash_gpgpu_sim::sass::functional_tcgen_state>();
  cta.tcgen->allocated = 3;
  execution_context context;
  context.cta = &cta;
  warp_state state;
  for (unsigned lane = 0; lane < 32; ++lane)
    for (unsigned word = 0; word < 64; ++word)
      state.write_register(lane, 8 + word, 1000 + lane * 64 + word);
  auto result = sass.step("store64", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  const auto words = cta.tcgen->memory.read_matrix_words({0, 0}, 0, 32, 64);
  for (unsigned i = 0; i < words.size(); ++i)
    EXPECT_EQ(words[i], 1000 + i) << "word " << i;
}

TEST(SassFunctionalTest, Native16DpLoadSupportsCheckedNegativeTmemOffsets) {
  kernel image;
  image.name = "load16";
  image.arch = architecture::kSm103;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "LDTM.16dp256bit", "R4,tmem[UR4+-0xffffc]")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm100_functional_semantics(sass);
  cta_execution_state cta(1);
  cta.tcgen = std::make_shared<flash_gpgpu_sim::sass::functional_tcgen_state>();
  cta.tcgen->allocated = 1;
  std::vector<uint32_t> values(128 * 32);
  for (unsigned r = 0; r < 128; ++r)
    for (unsigned c = 0; c < 32; ++c) values[r * 32 + c] = r * 100 + c;
  cta.tcgen->memory.write_matrix_words({0, 0}, 0, values, 128, 32);
  execution_context context;
  context.cta = &cta;
  warp_state state;
  state.write_uniform_register(4, 0x200000);
  ASSERT_EQ(sass.step("load16", state, context).status, step_status::kAdvanced);
  for (unsigned lane = 0; lane < 32; ++lane)
    for (unsigned w = 0; w < 4; ++w)
      EXPECT_EQ(
          state.read_register(lane, 4 + w),
          (16 + lane / 4 + (w / 2) * 8) * 100 + 4 + (lane % 4) * 2 + w % 2);
  state.pc = 0;
  state.write_uniform_register(4, 0xffffb);
  EXPECT_EQ(sass.step("load16", state, context).status,
            step_status::kUnsupported);
}

TEST(SassFunctionalTest, NativeTmemTransfersSelectIssuingWarpPartition) {
  kernel image;
  image.name = "warp_tmem";
  image.arch = architecture::kSm103;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "STTM.x4", "tmem[UR0],R4"),
                        official_instruction(16, "LDTM.x4", "R8,tmem[UR0]")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm100_functional_semantics(sass);
  cta_execution_state cta(8);
  cta.tcgen = std::make_shared<flash_gpgpu_sim::sass::functional_tcgen_state>();
  cta.tcgen->allocated = 1;
  execution_context context;
  context.cta = &cta;
  for (unsigned warp = 0; warp < 4; ++warp) {
    context.warp_id = warp;
    warp_state state;
    // Identical column-only address, as emitted by production FlashMLA.
    for (unsigned lane = 0; lane < 32; ++lane)
      for (unsigned word = 0; word < 4; ++word)
        state.write_register(lane, 4 + word, warp * 10000 + lane * 4 + word);
    ASSERT_EQ(sass.step("warp_tmem", state, context).status,
              step_status::kAdvanced);
  }
  for (unsigned warp = 0; warp < 8; ++warp) {
    context.warp_id = warp;
    warp_state state;
    state.pc = 16;
    // Upper row bits cannot redirect a warp to another partition.
    state.write_uniform_register(0, ((3 - warp % 4) * 32) << 16);
    ASSERT_EQ(sass.step("warp_tmem", state, context).status,
              step_status::kAdvanced);
    for (unsigned lane = 0; lane < 32; ++lane)
      for (unsigned word = 0; word < 4; ++word)
        EXPECT_EQ(state.read_register(lane, 8 + word),
                  (warp % 4) * 10000 + lane * 4 + word);
  }
}

TEST(SassFunctionalTest, NativeEightWordTmemLoadPreservesWarpPartition) {
  kernel image;
  image.name = "load8";
  image.arch = architecture::kSm103;
  image.instruction_bytes = 16;
  image.instructions = {official_instruction(0, "LDTM.x8", "R20,tmem[UR4]")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm100_functional_semantics(sass);
  cta_execution_state cta(4);
  cta.tcgen = std::make_shared<flash_gpgpu_sim::sass::functional_tcgen_state>();
  cta.tcgen->allocated = 1;
  std::vector<uint32_t> values(128 * 16);
  for (unsigned r = 0; r < 128; ++r)
    for (unsigned c = 0; c < 16; ++c) values[r * 16 + c] = r * 100 + c;
  cta.tcgen->memory.write_matrix_words({0, 0}, 0, values, 128, 16);
  execution_context context;
  context.cta = &cta;
  context.warp_id = 2;
  warp_state state;
  state.write_uniform_register(4, 4);
  auto result = sass.step("load8", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  for (unsigned lane = 0; lane < 32; ++lane)
    for (unsigned word = 0; word < 8; ++word)
      EXPECT_EQ(state.read_register(lane, 20 + word),
                (64 + lane) * 100 + 4 + word);
}

TEST(SassFunctionalTest,
     SmallTmemStoresPreserveAdjacentColumnsAndWarpPartitions) {
  for (unsigned words : {1u, 2u}) {
    kernel image;
    image.name = "small_store";
    image.arch = architecture::kSm103;
    image.instruction_bytes = 16;
    image.instructions = {official_instruction(
        0, words == 1 ? "STTM" : "STTM.x2", "tmem[UR4],R8")};
    frontend sass;
    sass.add_kernel(std::move(image));
    register_sm100_functional_semantics(sass);
    cta_execution_state cta(4);
    cta.tcgen =
        std::make_shared<flash_gpgpu_sim::sass::functional_tcgen_state>();
    cta.tcgen->allocated = 1;
    std::vector<uint32_t> initial(128 * 32, 0xdeadbeef);
    cta.tcgen->memory.write_matrix_words({0, 0}, 0, initial, 128, 32);
    execution_context context;
    context.cta = &cta;
    for (unsigned warp = 0; warp < 4; ++warp) {
      context.warp_id = warp;
      warp_state state;
      state.write_uniform_register(4, 8);
      for (unsigned lane = 0; lane < 32; ++lane)
        for (unsigned w = 0; w < words; ++w)
          state.write_register(lane, 8 + w, 1000 * (warp * 32 + lane) + w + 1);
      auto result = sass.step("small_store", state, context);
      ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
      for (unsigned row = 0; row < 128; ++row) {
        auto values = cta.tcgen->memory.read_words({0, 0}, row << 16, 32);
        for (unsigned col = 0; col < 32; ++col)
          EXPECT_EQ(values[col], row / 32 <= warp && col >= 8 && col < 8 + words
                                     ? 1000 * row + col - 8 + 1
                                     : 0xdeadbeefu);
      }
    }
    if (words == 2) {
      warp_state state;
      state.write_uniform_register(4, 31);
      EXPECT_EQ(sass.step("small_store", state, context).status,
                step_status::kUnsupported);
      for (unsigned row = 0; row < 128; ++row)
        EXPECT_EQ(cta.tcgen->memory.read_words({0, 0}, (row << 16) + 31, 1)[0],
                  0xdeadbeefu);
    }
  }
}

TEST(SassFunctionalTest, PackedTmemUsesAdjacentLowHalvesAndKeepsOtherWarps) {
  for (unsigned words : {1u, 2u, 4u, 8u, 16u, 32u, 64u}) {
    const std::string width = words == 1 ? "" : ".x" + std::to_string(words);
    kernel image;
    image.name = "packed";
    image.arch = architecture::kSm100;
    image.instruction_bytes = 16;
    image.instructions = {
        official_instruction(0, "STTM" + width + ".EXPAND16BIT",
                             "tmem[UR0],R8"),
        official_instruction(16, "LDTM" + width + ".PACK16BIT",
                             "R80,tmem[UR0]")};
    frontend sass;
    sass.add_kernel(std::move(image));
    register_sm100_functional_semantics(sass);
    cta_execution_state cta(4);
    cta.tcgen =
        std::make_shared<flash_gpgpu_sim::sass::functional_tcgen_state>();
    cta.tcgen->allocated = 15;
    execution_context context;
    context.cta = &cta;
    context.warp_id = 2;
    warp_state state;
    for (unsigned lane = 0; lane < 32; ++lane)
      for (unsigned w = 0; w < words; ++w)
        state.write_register(
            lane, 8 + w,
            (0x8000u + lane * 64 + w) << 16 | (0x1000u + lane * 64 + w));
    ASSERT_EQ(sass.step("packed", state, context).status,
              step_status::kAdvanced);
    // Check physical storage independently of the load: a round trip alone
    // would hide the same permutation bug in both directions.
    for (unsigned lane = 0; lane < 32; ++lane)
      for (unsigned w = 0; w < words; ++w) {
        auto v = cta.tcgen->memory.read_words({0, 0},
                                              ((64 + lane) << 16) + 2 * w, 2);
        EXPECT_EQ(v[0], 0x1000u + lane * 64 + w);
        EXPECT_EQ(v[1], 0x8000u + lane * 64 + w);
        // Upper bits are ignored by PACK16BIT.
        cta.tcgen->memory.write_words({0, 0}, ((64 + lane) << 16) + 2 * w,
                                      {v[0] | 0xa5a50000u, v[1] | 0x5a5a0000u});
      }
    EXPECT_EQ(cta.tcgen->memory.read_words({0, 0}, 0, 1)[0], 0u);
    ASSERT_EQ(sass.step("packed", state, context).status,
              step_status::kAdvanced);
    for (unsigned lane = 0; lane < 32; ++lane)
      for (unsigned w = 0; w < words; ++w)
        EXPECT_EQ(state.read_register(lane, 80 + w),
                  state.read_register(lane, 8 + w));
  }
}

TEST(SassFunctionalTest, PackedTmemHalfSplitsKeepUnselectedLanesAndColumns) {
  for (unsigned half = 0; half < 2; ++half) {
    const std::string shape =
        half ? ".16dp32bit_t16_t31.x8" : ".16dp32bit_t0_t15.x8";
    kernel image;
    image.name = "split";
    image.arch = architecture::kSm100;
    image.instruction_bytes = 16;
    image.instructions = {
        official_instruction(0, "STTM" + shape + ".EXPAND16BIT",
                             "tmem[UR0],R8"),
        official_instruction(16, "LDTM" + shape + ".PACK16BIT",
                             "R32,tmem[UR0]")};
    frontend sass;
    sass.add_kernel(std::move(image));
    register_sm100_functional_semantics(sass);
    cta_execution_state cta(4);
    cta.tcgen =
        std::make_shared<flash_gpgpu_sim::sass::functional_tcgen_state>();
    cta.tcgen->allocated = 3;
    execution_context context;
    context.cta = &cta;
    context.warp_id = 3;
    warp_state state;
    state.write_uniform_register(0, 0x100000 + 16 * half);
    for (unsigned lane = 0; lane < 32; ++lane)
      for (unsigned w = 0; w < 8; ++w) {
        state.write_register(
            lane, 8 + w, (0x9000u + lane * 8 + w) << 16 | (lane * 8 + w + 1));
        state.write_register(lane, 32 + w, 0xdeadbeef);
      }
    ASSERT_EQ(sass.step("split", state, context).status,
              step_status::kAdvanced);
    for (unsigned row = 0; row < 16; ++row)
      for (unsigned w = 0; w < 8; ++w) {
        auto v = cta.tcgen->memory.read_words(
            {0, 0}, ((112 + row) << 16) + 16 * half + 2 * w, 2);
        EXPECT_EQ(v[0], (half * 16 + row) * 8 + w + 1);
        EXPECT_EQ(v[1], 0x9000u + (half * 16 + row) * 8 + w);
        EXPECT_EQ(
            cta.tcgen->memory.read_words(
                {0, 0}, ((112 + row) << 16) + 16 * (1 - half) + 2 * w, 1)[0],
            0u);
      }
    ASSERT_EQ(sass.step("split", state, context).status,
              step_status::kAdvanced);
    for (unsigned lane = 0; lane < 32; ++lane)
      for (unsigned w = 0; w < 8; ++w)
        EXPECT_EQ(
            state.read_register(lane, 32 + w),
            lane / 16 == half ? state.read_register(lane, 8 + w) : 0xdeadbeef);
  }
}

TEST(SassFunctionalTest, UnpackedTmemHalfSplitsPreserveAllBitsAndOtherLanes) {
  for (unsigned half = 0; half < 2; ++half)
    for (unsigned words : {1u, 2u, 4u, 8u, 16u, 32u, 64u}) {
      const std::string shape =
          std::string(half ? ".16dp32bit_t16_t31" : ".16dp32bit_t0_t15") +
          (words == 1 ? "" : ".x" + std::to_string(words));
      kernel image;
      image.name = "split32";
      image.arch = architecture::kSm100;
      image.instruction_bytes = 16;
      image.instructions = {
          official_instruction(0, "STTM" + shape, "tmem[UR0],R8"),
          official_instruction(16, "LDTM" + shape, "R80,tmem[UR0]"),
          official_instruction(32, "STTM" + shape, "tmem[UR0],RZ")};
      frontend sass;
      sass.add_kernel(std::move(image));
      register_sm100_functional_semantics(sass);
      cta_execution_state cta(4);
      cta.tcgen =
          std::make_shared<flash_gpgpu_sim::sass::functional_tcgen_state>();
      cta.tcgen->allocated = 15;
      execution_context context;
      context.cta = &cta;
      context.warp_id = 2;
      warp_state state;
      state.write_uniform_register(0, 0x100000 + 64 * half);
      for (unsigned lane = 0; lane < 32; ++lane)
        for (unsigned w = 0; w < words; ++w) {
          state.write_register(lane, 8 + w, 0x89abcdefu + lane * 67 + w);
          state.write_register(lane, 80 + w, 0xdeadbeef);
        }
      ASSERT_EQ(sass.step("split32", state, context).status,
                step_status::kAdvanced);
      for (unsigned row = 0; row < 16; ++row)
        for (unsigned w = 0; w < words; ++w) {
          const uint32_t address = ((80 + row) << 16) + 64 * half + w;
          EXPECT_EQ(cta.tcgen->memory.read_words({0, 0}, address, 1)[0],
                    0x89abcdefu + (half * 16 + row) * 67 + w);
          EXPECT_EQ(cta.tcgen->memory.read_words(
                        {0, 0}, ((80 + row) << 16) + 64 * (1 - half) + w, 1)[0],
                    0u);
        }
      ASSERT_EQ(sass.step("split32", state, context).status,
                step_status::kAdvanced);
      for (unsigned lane = 0; lane < 32; ++lane)
        for (unsigned w = 0; w < words; ++w)
          EXPECT_EQ(state.read_register(lane, 80 + w),
                    lane / 16 == half ? state.read_register(lane, 8 + w)
                                      : 0xdeadbeef);
      ASSERT_EQ(sass.step("split32", state, context).status,
                step_status::kAdvanced);
      for (unsigned row = 0; row < 16; ++row)
        for (unsigned w = 0; w < words; ++w)
          EXPECT_EQ(cta.tcgen->memory.read_words(
                        {0, 0}, ((80 + row) << 16) + 64 * half + w, 1)[0],
                    0u);
    }
}

TEST(SassFunctionalTest, TmemZeroBroadcastAndPackedAllocationFailureAreAtomic) {
  for (const std::string opcode : {"STTM.x128", "STTM.x32.EXPAND16BIT"}) {
    kernel image;
    image.name = "zero";
    image.arch = architecture::kSm100;
    image.instruction_bytes = 16;
    image.instructions = {official_instruction(0, opcode, "tmem[UR0],RZ")};
    frontend sass;
    sass.add_kernel(std::move(image));
    register_sm100_functional_semantics(sass);
    cta_execution_state cta(4);
    cta.tcgen =
        std::make_shared<flash_gpgpu_sim::sass::functional_tcgen_state>();
    cta.tcgen->allocated = 15;
    cta.tcgen->memory.write_matrix_words(
        {0, 0}, 0, std::vector<uint32_t>(32 * 128, 0xdeadbeef), 32, 128);
    execution_context context;
    context.cta = &cta;
    warp_state state;
    ASSERT_EQ(sass.step("zero", state, context).status, step_status::kAdvanced);
    const unsigned columns = opcode == "STTM.x128" ? 128 : 64;
    auto v = cta.tcgen->memory.read_matrix_words({0, 0}, 0, 32, 128);
    for (unsigned row = 0; row < 32; ++row)
      for (unsigned col = 0; col < 128; ++col)
        EXPECT_EQ(v[row * 128 + col], col < columns ? 0u : 0xdeadbeef);
  }
  kernel image;
  image.name = "bounds";
  image.arch = architecture::kSm100;
  image.instruction_bytes = 16;
  image.instructions = {
      official_instruction(0, "STTM.EXPAND16BIT", "tmem[UR0],R8")};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm100_functional_semantics(sass);
  cta_execution_state cta(4);
  cta.tcgen = std::make_shared<flash_gpgpu_sim::sass::functional_tcgen_state>();
  cta.tcgen->allocated = 1;
  cta.tcgen->memory.write_words({0, 0}, 31, {0xabcdef01});
  execution_context context;
  context.cta = &cta;
  warp_state state;
  state.write_uniform_register(0, 31);
  EXPECT_EQ(sass.step("bounds", state, context).status,
            step_status::kUnsupported);
  EXPECT_EQ(cta.tcgen->memory.read_words({0, 0}, 31, 1)[0], 0xabcdef01u);
  EXPECT_EQ(state.pc, 0u);
}
}  // namespace
