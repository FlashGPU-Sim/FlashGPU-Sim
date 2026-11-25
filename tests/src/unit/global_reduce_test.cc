#include <fstream>
#include "libcuda/gpgpu_context.h"
#include "ptx.tab.h"
#include "sass/test_support.h"
CUctx_st *GPGPUSim_Context(gpgpu_context *ctx);
namespace {
using flash_gpgpu_sim::sass::kZeroRegister;
TEST(GlobalReduction, Atomic64ReturnedPairsPredicationAndDependencies) {
  for (const char *scope : {"GPU", "SYS"})
    for (const char *destination : {"R2", "R4", "R8", "RZ"}) {
      const auto inst = official_instruction(
          0, std::string("ATOMG.E.ADD.64.STRONG.") + scope,
          std::string("PT,") + destination + ",desc[UR8][R2.64],R4", "@P0");
      kernel image;
      image.name = "atomic64";
      image.arch = architecture::kSm100;
      image.instruction_bytes = 16;
      image.instructions = {inst};
      frontend sass;
      sass.add_kernel(std::move(image));
      register_sm100_functional_semantics(sass);
      mapped_functional_memory memory;
      constexpr uint64_t at = 0x200000100ull, initial = 0xfffffffeffffffffull;
      memory.map(memory_space::kGlobal, at, 16, true);
      memory.initialize<uint64_t>(memory_space::kGlobal, at, initial);
      memory.initialize<uint64_t>(memory_space::kGlobal, at + 8,
                                  0xabcdefff12345678ull);
      execution_context context{&memory};
      warp_state state;
      state.active_mask = 7;
      for (unsigned lane = 0; lane < 3; ++lane) {
        state.write_register(lane, 2, uint32_t(at));
        state.write_register(lane, 3, at >> 32);
        state.write_register(lane, 4, 3);
        state.write_register(lane, 5, 1);
        state.write_register(lane, 8, 99);
        state.write_register(lane, 9, 100);
        state.write_predicate(lane, 0, lane != 1);
      }
      const auto result = sass.step("atomic64", state, context);
      ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
      EXPECT_EQ(memory.inspect<uint64_t>(memory_space::kGlobal, at),
                initial + 2 * 0x100000003ull);
      EXPECT_EQ(memory.inspect<uint64_t>(memory_space::kGlobal, at + 8),
                0xabcdefff12345678ull);
      EXPECT_TRUE(context.memory_accesses[1].empty());
      EXPECT_EQ(context.memory_accesses[0][0].bytes, 8u);
      EXPECT_EQ(state.read_register(1, 2), uint32_t(at));
      EXPECT_EQ(state.read_register(1, 4), 3u);
      EXPECT_EQ(state.read_register(1, 8), 99u);
      const unsigned dst = inst.operands[1].index;
      if (dst != kZeroRegister) {
        EXPECT_EQ(uint64_t(state.read_register(0, dst)) |
                      (uint64_t(state.read_register(0, dst + 1)) << 32),
                  initial);
        EXPECT_EQ(uint64_t(state.read_register(2, dst)) |
                      (uint64_t(state.read_register(2, dst + 1)) << 32),
                  initial + 0x100000003ull);
      }
      timing_instruction projected(inst);
      EXPECT_EQ(projected.data_size, 8u);
      EXPECT_TRUE(projected.isatomic());
      EXPECT_EQ(projected.outcount, dst == kZeroRegister ? 0u : 2u);
      ASSERT_EQ(projected.get_register_file_destination_count(),
                dst == kZeroRegister ? 0u : 2u);
      if (dst != kZeroRegister) {
        EXPECT_EQ(projected.get_register_file_destination(0).reg, dst);
        EXPECT_EQ(projected.get_register_file_destination(1).reg, dst + 1);
      }
      std::vector<unsigned> sources;
      for (unsigned i = 0; i < projected.get_register_file_source_count(); ++i)
        sources.push_back(projected.get_register_file_source(i).reg);
      EXPECT_NE(std::find(sources.begin(), sources.end(), 4u), sources.end());
      EXPECT_NE(std::find(sources.begin(), sources.end(), 5u), sources.end());
    }
}
TEST(GlobalReduction, Atomic64RejectsInvalidPairsBeforeMemoryMutation) {
  for (const char *operands :
       {"PT,R254,desc[UR8][R2.64],R4", "PT,R8,desc[UR8][R2.64],R254",
        "PT,R8,desc[UR8][R2.64],-R4"}) {
    kernel image;
    image.name = "bad64";
    image.arch = architecture::kSm100;
    image.instruction_bytes = 16;
    image.instructions = {
        official_instruction(0, "ATOMG.E.ADD.64.STRONG.GPU", operands)};
    frontend sass;
    sass.add_kernel(std::move(image));
    register_sm100_functional_semantics(sass);
    mapped_functional_memory memory;
    memory.map(memory_space::kGlobal, 256, 8, true);
    memory.initialize<uint64_t>(memory_space::kGlobal, 256, 17);
    execution_context context{&memory};
    warp_state state;
    state.active_mask = 1;
    state.write_register(0, 2, 256);
    EXPECT_EQ(sass.step("bad64", state, context).status,
              step_status::kUnsupported);
    EXPECT_EQ(memory.inspect<uint64_t>(memory_space::kGlobal, 256), 17u);
    EXPECT_TRUE(context.memory_accesses[0].empty());
  }
}

TEST(GlobalReduction, UniformWidthsScopesMasksAndDependencies) {
  for (const bool wide : {false, true})
    for (const char *scope : {"GPU", "SYS"})
      for (const unsigned mask : {0xffffffffu, 1u << 7}) {
        const auto inst =
            official_instruction(0,
                                 std::string("UREDGR.E.ADD.NA.") +
                                     (wide ? "64." : "") + "STRONG." + scope,
                                 "[UR4],UR6");
        kernel image;
        image.name = "uniform_red";
        image.arch = architecture::kSm100;
        image.instruction_bytes = 16;
        image.instructions = {inst};
        frontend sass;
        sass.add_kernel(std::move(image));
        register_sm100_functional_semantics(sass);
        constexpr uint64_t at = 0x200000100ull;
        mapped_functional_memory memory;
        memory.map(memory_space::kGlobal, at, 16, true);
        memory.initialize<uint64_t>(memory_space::kGlobal, at,
                                    0xffffffffffffffffull);
        memory.initialize<uint64_t>(memory_space::kGlobal, at + 8,
                                    0xabcdefff12345678ull);
        execution_context context{&memory};
        warp_state state;
        state.active_mask = mask;
        state.write_uniform_register(4, uint32_t(at));
        state.write_uniform_register(5, at >> 32);
        state.write_uniform_register(6, 2);
        state.write_uniform_register(7, 1);
        const auto result = sass.step("uniform_red", state, context);
        ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
        EXPECT_EQ(memory.inspect<uint64_t>(memory_space::kGlobal, at),
                  wide ? 0x100000001ull : 0xffffffff00000001ull);
        EXPECT_EQ(memory.inspect<uint64_t>(memory_space::kGlobal, at + 8),
                  0xabcdefff12345678ull);
        unsigned accesses = 0;
        for (const auto &lane : context.memory_accesses)
          for (const auto &a : lane) {
            ++accesses;
            EXPECT_EQ(a.bytes, wide ? 8u : 4u);
            EXPECT_EQ(a.address, at);
          }
        EXPECT_EQ(accesses, 1u);
        timing_instruction projected(inst);
        EXPECT_EQ(projected.outcount, 0u);
        EXPECT_TRUE(projected.isatomic());
        EXPECT_TRUE(projected.is_release_memory());
        EXPECT_TRUE(projected.is_async_reduction());
        EXPECT_EQ(projected.data_size, wide ? 8u : 4u);
        std::vector<unsigned> deps(projected.in,
                                   projected.in + projected.incount);
        deps.insert(deps.end(), projected.extra_in.begin(),
                    projected.extra_in.end());
        // Uniform source and address pairs must retain their high-word
        // dependencies.
        EXPECT_EQ(deps.size(), wide ? 4u : 3u);
      }
}
TEST(GlobalReduction, WideVectorPredicationAndNoDestinations) {
  const auto inst = official_instruction(0, "REDG.E.ADD.64.STRONG.GPU",
                                         "desc[UR8][R2.64],R4", "@P0");
  kernel image;
  image.name = "vector_red";
  image.arch = architecture::kSm100;
  image.instruction_bytes = 16;
  image.instructions = {inst};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm100_functional_semantics(sass);
  mapped_functional_memory memory;
  memory.map(memory_space::kGlobal, 256, 8, true);
  memory.initialize<uint64_t>(memory_space::kGlobal, 256, 9);
  execution_context context{&memory};
  warp_state state;
  state.active_mask = 3;
  for (unsigned lane = 0; lane < 2; ++lane) {
    state.write_register(lane, 2, 256);
    state.write_register(lane, 3, 0);
    state.write_register(lane, 4, 1);
    state.write_register(lane, 5, 2);
    state.write_predicate(lane, 0, lane == 1);
  }
  const auto result = sass.step("vector_red", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  EXPECT_EQ(memory.inspect<uint64_t>(memory_space::kGlobal, 256),
            0x20000000aull);
  EXPECT_TRUE(context.memory_accesses[0].empty());
  EXPECT_EQ(context.memory_accesses[1].size(), 1u);
  timing_instruction projected(inst);
  EXPECT_EQ(projected.outcount, 0u);
  EXPECT_EQ(projected.data_size, 8u);
  EXPECT_EQ(projected.get_register_file_source_count(), 2u);
}
TEST(GlobalReduction,
     PtxPredecodeIsMemoryWithoutDestinationAndPreservesRelease) {
  auto *ctx = GPGPU_Context();
  GPGPUSim_Context(ctx);
  const char *path = "global_reduction_predecode.ptx";
  {
    std::ofstream f(path);
    f << R"(.version 9.0
.target sm_100
.address_size 64
.visible .entry global_red_predecode(){
.reg .u64 a,b;
red.global.add.u64 [a],b;
red.async.release.gpu.global.add.u64 [a],b;
red.global.xor.b64 [a],b;
ret;
})";
  }
  auto *symbols = ctx->gpgpu_ptx_sim_load_ptx_from_filename(path);
  std::remove(path);
  ASSERT_NE(symbols, nullptr);
  auto *f = symbols->lookup_function("global_red_predecode");
  ASSERT_NE(f, nullptr);
  f->do_pdom();
  unsigned pc = f->get_start_PC();
  for (bool release : {false, true, false}) {
    auto *inst = f->get_dyn_inst(pc);
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->op, LOAD_OP);
    EXPECT_EQ(inst->cache_op, CACHE_GLOBAL);
    EXPECT_TRUE(inst->has_memory_read());
    EXPECT_FALSE(inst->has_memory_write());
    EXPECT_EQ(inst->outcount, 0u);
    EXPECT_EQ(inst->is_release_memory(), release);
    pc += inst->inst_size();
  }
}
}  // namespace

namespace {
TEST(GlobalReduction, ExistingAtomicAddPreservesReturnedValueAndAddressAlias) {
  const auto inst = official_instruction(0, "ATOMG.E.ADD.STRONG.GPU",
                                         "PT,R2,desc[UR8][R2.64],R4");
  kernel image;
  image.name = "old_add";
  image.arch = architecture::kSm100;
  image.instruction_bytes = 16;
  image.instructions = {inst};
  frontend sass;
  sass.add_kernel(std::move(image));
  register_sm100_functional_semantics(sass);
  mapped_functional_memory memory;
  memory.map(memory_space::kGlobal, 256, 4, true);
  memory.initialize<uint32_t>(memory_space::kGlobal, 256, 9);
  execution_context context{&memory};
  warp_state state;
  state.active_mask = 3;
  for (unsigned lane = 0; lane < 2; ++lane) {
    state.write_register(lane, 2, 256);
    state.write_register(lane, 3, 0);
    state.write_register(lane, 4, lane + 1);
  }
  const auto result = sass.step("old_add", state, context);
  ASSERT_EQ(result.status, step_status::kAdvanced) << result.detail;
  EXPECT_EQ(memory.inspect<uint32_t>(memory_space::kGlobal, 256), 12u);
  EXPECT_EQ(state.read_register(0, 2), 9u);
  EXPECT_EQ(state.read_register(1, 2), 10u);
}
}  // namespace
