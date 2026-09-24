#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "libcuda/gpgpu_context.h"
#include "ptx.tab.h"

CUctx_st *GPGPUSim_Context(gpgpu_context *ctx);

namespace {
class PackedTimingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ctx = GPGPU_Context();
    GPGPUSim_Context(ctx);
    char dir[] = "/tmp/packed-timing-test-XXXXXX";
    ASSERT_NE(mkdtemp(dir), nullptr);
    work = dir;
    auto *sim = ctx->func_sim;
    saved = {sim->opcode_latency_f32x2, sim->opcode_initiation_f32x2,
             sim->opcode_latency_cvt_f16x2_f32,
             sim->opcode_initiation_cvt_f16x2_f32};
    reorder = ctx->ptx_reorder_enabled;
    ctx->ptx_reorder_enabled = false;
    sim->opcode_latency_f32x2 = sim->opcode_initiation_f32x2 = 0;
    sim->opcode_latency_cvt_f16x2_f32 = 1;
    sim->opcode_initiation_cvt_f16x2_f32 = 1;
  }
  void TearDown() override {
    auto *sim = ctx->func_sim;
    sim->opcode_latency_f32x2 = saved[0];
    sim->opcode_initiation_f32x2 = saved[1];
    sim->opcode_latency_cvt_f16x2_f32 = saved[2];
    sim->opcode_initiation_cvt_f16x2_f32 = saved[3];
    ctx->ptx_reorder_enabled = reorder;
    if (!work.empty()) std::filesystem::remove_all(work);
  }
  std::vector<const ptx_instruction *> decode() {
    static unsigned id = 0;
    const std::string name = "packed_timing_" + std::to_string(++id);
    const std::string source =
        ".version 8.8\n.target sm_100\n.address_size 64\n.visible .entry " +
        name +
        "() {\n.reg .b64 a,b,c;\n.reg .f32 x,y,z;\n.reg .b32 h;\n"
        "add.rn.f32x2 a,b,c;\nadd.f32 x,y,z;\n"
        "cvt.rn.f16x2.f32 h,x,y;\ncvt.rn.f16.f32 h,x;\nret;\n}\n";
    const auto path = (work / "input.ptx").string();
    std::ofstream(path) << source;
    auto *symbols = ctx->gpgpu_ptx_sim_load_ptx_from_filename(path.c_str());
    if (!symbols) {
      ADD_FAILURE();
      return {};
    }
    auto *func = symbols->lookup_function(name.c_str());
    if (!func) {
      ADD_FAILURE();
      return {};
    }
    func->do_pdom();
    std::vector<const ptx_instruction *> result;
    for (unsigned pc = func->get_start_PC(); result.size() < 4;) {
      const auto *inst = func->get_dyn_inst(pc);
      if (!inst || inst->get_opcode() == RET_OP) break;
      result.push_back(inst);
      pc += inst->inst_size();
    }
    return result;
  }
  gpgpu_context *ctx = nullptr;
  std::array<unsigned, 4> saved;
  bool reorder = false;
  std::filesystem::path work;
};

TEST_F(PackedTimingTest, DefaultsInheritScalarAndKeepLegacyConversion) {
  const auto inst = decode();
  ASSERT_EQ(inst.size(), 4u);
  EXPECT_EQ(inst[0]->latency, inst[1]->latency);
  EXPECT_EQ(inst[0]->initiation_interval, inst[1]->initiation_interval);
  EXPECT_EQ(inst[0]->op, SP_OP);
  EXPECT_EQ(inst[2]->latency, 1u);
  EXPECT_EQ(inst[2]->initiation_interval, 1u);
  EXPECT_EQ(inst[2]->op, ALU_OP);
}

TEST_F(PackedTimingTest, PackedOverridesDoNotChangeScalarInstructions) {
  ctx->func_sim->opcode_latency_f32x2 = 7;
  ctx->func_sim->opcode_initiation_f32x2 = 2;
  ctx->func_sim->opcode_latency_cvt_f16x2_f32 = 4;
  ctx->func_sim->opcode_initiation_cvt_f16x2_f32 = 2;
  const auto inst = decode();
  ASSERT_EQ(inst.size(), 4u);
  EXPECT_EQ(inst[0]->latency, 7u);
  EXPECT_EQ(inst[0]->initiation_interval, 2u);
  EXPECT_EQ(inst[0]->op, SP_OP);
  unsigned scalar_add = 0;
  ASSERT_EQ(sscanf(ctx->func_sim->opcode_latency_fp, "%u", &scalar_add), 1);
  EXPECT_EQ(inst[1]->latency, scalar_add);
  EXPECT_EQ(inst[2]->latency, 4u);
  EXPECT_EQ(inst[2]->initiation_interval, 2u);
  EXPECT_EQ(inst[2]->op, ALU_OP);
  EXPECT_EQ(inst[3]->latency, 1u);
  EXPECT_EQ(inst[3]->initiation_interval, 1u);
}

TEST_F(PackedTimingTest, IssueOverridePreservesScalarDependencyLatency) {
  ctx->func_sim->opcode_initiation_f32x2 = 2;
  const auto inst = decode();
  ASSERT_EQ(inst.size(), 4u);
  EXPECT_EQ(inst[0]->latency, inst[1]->latency);
  EXPECT_EQ(inst[0]->initiation_interval, 2u);
}

TEST_F(PackedTimingTest, RejectsInitiationLongerThanLatency) {
  ctx->func_sim->opcode_latency_f32x2 = 1;
  ctx->func_sim->opcode_initiation_f32x2 = 2;
  EXPECT_DEATH(decode(), "invalid packed timing");
}

TEST_F(PackedTimingTest, RejectsZeroConversionInitiation) {
  ctx->func_sim->opcode_initiation_cvt_f16x2_f32 = 0;
  EXPECT_DEATH(decode(), "invalid packed timing");
}

TEST_F(PackedTimingTest, PipelineDepthCoversPackedOverrides) {
  ctx->func_sim->opcode_latency_f32x2 = 17;
  ctx->func_sim->opcode_latency_cvt_f16x2_f32 = 23;
  shader_core_config config(ctx);
  config.set_pipeline_latency();
  EXPECT_GE(config.max_sp_latency, 23u);
  EXPECT_GE(config.max_int_latency, 23u);
}

TEST_F(PackedTimingTest, RejectsLatencyOutsideReservationWindow) {
  ctx->func_sim->opcode_latency_f32x2 = simd_function_unit::MAX_ALU_LATENCY;
  EXPECT_DEATH(decode(), "invalid packed timing");
}
}  // namespace
