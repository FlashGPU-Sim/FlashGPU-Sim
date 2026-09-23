#include <gtest/gtest.h>

// Exercise the production parser and scheduler with an explicit synthetic guide.
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include "libcuda/gpgpu_context.h"
#include "ptx.tab.h"


#include "gpgpu-sim/flash/ptx_sched/ptx_scheduler.h"

namespace {

using flash_gpgpu_sim::detail::ptx_schedule_priority_precedes;
using flash_gpgpu_sim::detail::ptx_schedule_priority_t;

TEST(PtxSchedulerPriorityTest, EarlierIssueWinsOverLongerRemainingPath) {
  const ptx_schedule_priority_t ready_now = {12, 8, 20};
  const ptx_schedule_priority_t stalled_critical_path = {28, 200, 4};

  EXPECT_TRUE(
      ptx_schedule_priority_precedes(ready_now, stalled_critical_path));
  EXPECT_FALSE(
      ptx_schedule_priority_precedes(stalled_critical_path, ready_now));
}

TEST(PtxSchedulerPriorityTest, LongerRemainingPathBreaksIssueTimeTie) {
  const ptx_schedule_priority_t longer_path = {12, 40, 20};
  const ptx_schedule_priority_t shorter_path = {12, 8, 4};

  EXPECT_TRUE(ptx_schedule_priority_precedes(longer_path, shorter_path));
  EXPECT_FALSE(ptx_schedule_priority_precedes(shorter_path, longer_path));
}

TEST(PtxSchedulerPriorityTest, SourceOrderBreaksEquivalentTie) {
  const ptx_schedule_priority_t earlier_source = {12, 40, 4};
  const ptx_schedule_priority_t later_source = {12, 40, 20};

  EXPECT_TRUE(ptx_schedule_priority_precedes(earlier_source, later_source));
  EXPECT_FALSE(ptx_schedule_priority_precedes(later_source, earlier_source));
}

}  // namespace

CUctx_st *GPGPUSim_Context(gpgpu_context *ctx);

namespace {
class PtxSchedulerGuidedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ctx = GPGPU_Context();
    GPGPUSim_Context(ctx);
    enabled = ctx->ptx_reorder_enabled;
    guided = ctx->ptx_reorder_sass_guided;
    guide_path = ctx->ptx_reorder_sass_ptxline_file;
    const char *explicit_guide = getenv("GPGPUSIM_SASS_PTXLINE_GUIDE");
    had_override = explicit_guide != nullptr;
    if (explicit_guide) previous_override = explicit_guide;
    cwd = std::filesystem::current_path();
    char dir[] = "/tmp/ptx-guided-test-XXXXXX";
    ASSERT_NE(mkdtemp(dir), nullptr);
    work = dir;
    std::filesystem::current_path(work);
    ctx->ptx_reorder_enabled = true;
    ctx->ptx_reorder_sass_guided = true;
    ctx->ptx_reorder_sass_ptxline_file = (work / "guide.sass").string();
    setenv("GPGPUSIM_SASS_PTXLINE_GUIDE",
           ctx->ptx_reorder_sass_ptxline_file.c_str(), 1);
    rules_path = work.filename().string() + ".rules";
    std::ofstream(rules_path) <<
        "[primary_opcode]\nadd.u32 = IADD3\n"
        "mbarrier.arrive.release.cta.shared::cta.b64 = SYNCS.ARRIVE\n"
        "[policy]\nfallback = disabled\n";
  }
  void TearDown() override {
    if (had_override)
      setenv("GPGPUSIM_SASS_PTXLINE_GUIDE", previous_override.c_str(), 1);
    else
      unsetenv("GPGPUSIM_SASS_PTXLINE_GUIDE");
    ctx->ptx_reorder_enabled = enabled;
    ctx->ptx_reorder_sass_guided = guided;
    ctx->ptx_reorder_sass_ptxline_file = guide_path;
    std::filesystem::current_path(cwd);
    if (!work.empty()) std::filesystem::remove_all(work);
  }
  std::vector<unsigned> schedule(const std::string &name,
                                 const std::string &body,
                                 const std::vector<unsigned> &guide_order) {
    // Body starts at physical PTX line 8.
    std::ofstream("input.ptx") <<
        ".version 8.0\n.target sm_90\n.address_size 64\n.visible .entry " << name <<
        "() {\n.reg .b32 %a, %b, %c;\n.reg .b64 %state;\n"
        ".shared .align 8 .b64 mb;\n" << body << "ret;\n}\n";
    {
      std::ofstream guide("guide.sass");
      guide << ".section .text." << name << ",\"ax\",@progbits\n";
      unsigned offset = 0;
      for (unsigned line : guide_order) {
        guide << "//## File \"input.ptx\", line " << line << "\n/*"
              << std::hex << offset << std::dec << "*/ "
              << (line == 9 ? "SYNCS.ARRIVE;" : "IADD3 R0, R1, R2;") << "\n";
        offset += 16;
      }
    }
    auto *symbols = ctx->gpgpu_ptx_sim_load_ptx_from_filename("input.ptx");
    if (!symbols) { ADD_FAILURE() << "PTX load failed"; return {}; }
    auto *func = symbols->lookup_function(name.c_str());
    if (!func) { ADD_FAILURE() << "function missing"; return {}; }
    func->do_pdom();
    std::vector<unsigned> lines;
    for (unsigned pc = func->get_start_PC(); lines.size() < 16;) {
      const auto *inst = func->get_dyn_inst(pc);
      if (!inst || inst->get_opcode() == RET_OP) break;
      lines.push_back(inst->source_line());
      pc += inst->inst_size();
    }
    return lines;
  }
  std::vector<const ptx_instruction *> fusion(const std::string &name,
                                              const std::string &body) {
    ctx->ptx_reorder_sass_guided = false;
    unsetenv("GPGPUSIM_SASS_PTXLINE_GUIDE");
    std::ofstream("fusion.ptx") <<
        ".version 8.0\n.target sm_90\n.address_size 64\n.visible .entry " <<
        name << "() {\n.reg .f32 a,b,product,zero,result;\n"
        ".reg .pred source,inverted;\n.reg .b32 value;\n" <<
        body << "ret;\n}\n";
    auto *symbols = ctx->gpgpu_ptx_sim_load_ptx_from_filename("fusion.ptx");
    if (!symbols) { ADD_FAILURE(); return {}; }
    auto *func = symbols->lookup_function(name.c_str());
    if (!func) { ADD_FAILURE(); return {}; }
    func->do_pdom();
    std::vector<const ptx_instruction *> result;
    for (unsigned pc = func->get_start_PC(); result.size() < 32;) {
      const auto *inst = func->get_dyn_inst(pc);
      if (!inst || inst->get_opcode() == RET_OP) break;
      result.push_back(inst);
      pc += inst->inst_size();
    }
    return result;
  }
  gpgpu_context *ctx = nullptr;
  bool enabled = false, guided = false;
  std::string guide_path, previous_override, rules_path;
  bool had_override = false;
  std::filesystem::path cwd, work;
};

TEST_F(PtxSchedulerGuidedTest, NegatedMultiplyOnlyFusesPlainRoundNearest) {
  unsigned index = 0;
  for (const auto &options : std::vector<std::pair<std::string, std::string>>{
           {"", ""}, {".rn", ".rn"}, {".sat", ""}, {"", ".sat"},
           {".rz", ""}, {"", ".rm"}, {".ftz", ""}, {"", ".ftz"}}) {
    const auto insts = fusion("negated_mul_" + std::to_string(index),
        "mul" + options.first + ".f32 product,a,b;\n"
        "mov.b32 zero,0f00000000;\nsub" + options.second +
        ".f32 result,zero,product;\n");
    ASSERT_FALSE(insts.empty());
    unsigned fused = 0, subtracts = 0;
    for (const auto *inst : insts) {
      fused += inst->is_compiler_negated_mul();
      subtracts += inst->get_opcode() == SUB_OP;
    }
    EXPECT_EQ(fused, index < 2 ? 1u : 0u);
    EXPECT_EQ(subtracts, index < 2 ? 0u : 1u);
    ++index;
  }
}

TEST_F(PtxSchedulerGuidedTest, PredicateGuardFusionPreservesDataOperandUses) {
  for (bool data_use : {false, true}) {
    const auto insts = fusion(data_use ? "guard_data_use" : "guard_only",
        std::string("setp.eq.u32 source,0,1;\nnot.pred inverted,source;\n") +
        (data_use ? "@inverted selp.u32 value,7,9,inverted;\n"
                  : "@inverted mov.u32 value,7;\n"));
    ASSERT_FALSE(insts.empty());
    unsigned inversions = 0;
    for (const auto *inst : insts) inversions += inst->get_opcode() == NOT_OP;
    EXPECT_EQ(inversions, data_use ? 1u : 0u);
  }
}

TEST_F(PtxSchedulerGuidedTest, MovesIndependentWorkOnlyForwardAcrossAnchor) {
  EXPECT_EQ(schedule("guided_forward",
      "add.u32 %a, %b, 1;\n"
      "mbarrier.arrive.release.cta.shared::cta.b64 %state, [mb];\n"
      "add.u32 %c, %b, 2;\n", {10, 9, 8}),
      (std::vector<unsigned>{9, 10, 8}));
}

TEST_F(PtxSchedulerGuidedTest, RegisterDependencyPreventsGuideRequestedCrossing) {
  EXPECT_EQ(schedule("guided_dependency",
      "add.u32 %a, %b, 1;\n"
      "mbarrier.arrive.release.cta.shared::cta.b64 %state, [%a];\n", {9, 8}),
      (std::vector<unsigned>{8, 9}));
}

TEST_F(PtxSchedulerGuidedTest, WorkWithoutGuideEvidenceStaysBeforeAnchor) {
  EXPECT_EQ(schedule("guided_no_evidence",
      "add.u32 %a, %b, 1;\n"
      "mbarrier.arrive.release.cta.shared::cta.b64 %state, [mb];\n", {9}),
      (std::vector<unsigned>{8, 9}));
}
TEST(PtxSchedulerGuideGeneratorTest, LineMappingAndDisassemblyParsing) {
  const char *root = getenv("GPGPUSIM_ROOT");
  ASSERT_NE(root, nullptr);
  const std::string script = std::string(root) +
      "/tests/src/unit/test_generate_sass_ptxline_guide.py";
  const pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    execlp("python3", "python3", script.c_str(), static_cast<char *>(nullptr));
    _exit(127);
  }
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

// Actual ptxas/nvdisasm generation exercises the shared loader boundary,
// independent of any frontend or application-binary extraction.
TEST_F(PtxSchedulerGuidedTest, GeneratesDistinctGuidesForFileAndStringModules) {
  unsetenv("GPGPUSIM_SASS_PTXLINE_GUIDE");
  std::ofstream(rules_path) <<
      "[primary_opcode]\nex2.approx.ftz.f32 = MUFU.EX2\n"
      "[policy]\nfallback = disabled\n";
  std::string previous;
  for (bool from_string : {false, true}) {
    const std::string name = from_string ? "auto_string" : "auto_file";
    std::string source =
        ".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry " + name +
        "(.param .u64 param_out, .param .f32 param_in) {\n"
        ".reg .b64 %rd;\n.reg .f32 %x, %y;\n"
        "ld.param.u64 %rd, [param_out];\n"
        "ld.param.f32 %x, [param_in];\n"
        "ex2.approx.ftz.f32 %y, %x;\n"
        "st.global.f32 [%rd], %y;\nret;\n}\n";
    if (from_string) {
      // Keep the module directives on separate lines, but compact the body.
      const auto body = source.find(".visible");
      for (size_t i = body; i < source.size(); ++i)
        if (source[i] == '\n') source[i] = ' ';
    } else {
      const auto operand = source.find("%y, %x;");
      source.insert(operand, "\n  ");
    }
    symbol_table *symbols;
    if (from_string) {
      symbols = ctx->gpgpu_ptx_sim_load_ptx_from_string(source.c_str(), 9001);
    } else {
      std::ofstream("auto.ptx") << source;
      symbols = ctx->gpgpu_ptx_sim_load_ptx_from_filename("auto.ptx");
    }
    ASSERT_NE(symbols, nullptr);
    ASSERT_NE(symbols->lookup_function(name.c_str()), nullptr);
    const std::string path = ctx->ptx_reorder_sass_ptxline_file;
    EXPECT_NE(path, previous);
    ASSERT_TRUE(std::filesystem::is_regular_file(path));
    std::ifstream guide(path);
    const std::string text((std::istreambuf_iterator<char>(guide)), {});
    EXPECT_NE(text.find(name), std::string::npos);
    EXPECT_NE(text.find("MUFU.EX2"), std::string::npos);
    previous = path;
  }
}
}  // namespace
