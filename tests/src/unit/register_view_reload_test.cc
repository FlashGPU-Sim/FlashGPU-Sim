// Exercise repeated function definitions through the production PTX loader.
#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>

#include "libcuda/gpgpu_context.h"
#include "ptx.tab.h"
#include "src/gpgpu-sim/scoreboard.h"

CUctx_st *GPGPUSim_Context(gpgpu_context *ctx);

TEST(RegisterViewReloadTest, RebuildsStateForRepeatedDefinitions) {
  gpgpu_context *ctx = GPGPU_Context();
  GPGPUSim_Context(ctx);
  ctx->ptx_reorder_enabled = true;
  ctx->ptx_reorder_sass_guided = false;
  const std::string header =
      ".version 8.0\n.target sm_80\n.address_size 64\n"
      ".visible .entry reload_view() {\n"
      ".reg .b64 %rd;\n.reg .b32 %lo, %hi, %result;\n"
      "mov.b64 %rd, 0x1234567887654321;\n"
      "mov.b64 {%lo, %hi}, %rd;\n";
  const std::string body = "add.u32 %result, %lo, %hi;\nret;\n}\n";
  function_info *first = nullptr;
  // The first replacement happens before predecode, as in a multi-cubin
  // executable; subsequent replacements also exercise an assembled function.
  for (unsigned pass = 0; pass < 4; ++pass) {
    const bool optimized = pass != 2;
    ctx->ptx_reorder_enabled = optimized;
    const std::string path =
        "register_view_reload_" + std::to_string(pass) + ".ptx";
    {
      std::ofstream fixture(path);
      fixture << header << body;
      ASSERT_TRUE(fixture.good()) << "could not write PTX fixture";
    }
    auto *symbols = ctx->gpgpu_ptx_sim_load_ptx_from_filename(path.c_str());
    std::remove(path.c_str());
    ASSERT_NE(symbols, nullptr);
    auto *func = symbols->lookup_function("reload_view");
    ASSERT_NE(func, nullptr);
    if (!first) first = func;
    ASSERT_TRUE(func == first) << "fixture did not reuse the function object";
    auto *lo = func->get_symtab()->lookup("%lo");
    auto *hi = func->get_symtab()->lookup("%hi");
    auto *rd = func->get_symtab()->lookup("%rd");
    ASSERT_TRUE(func->get_compiler_register_view(lo, nullptr, nullptr) ==
                optimized)
        << "stale or missing low-half view";
    ASSERT_TRUE(func->get_compiler_register_view(hi, nullptr, nullptr) ==
                optimized)
        << "stale or missing high-half view";
    if (optimized) {
      ASSERT_TRUE(lo->reg_num() == rd->reg_num() &&
                  hi->reg_num() == rd->reg_num())
          << "view register IDs were not installed";
    } else {
      ASSERT_TRUE(lo->reg_num() != hi->reg_num() &&
                  lo->reg_num() != rd->reg_num() &&
                  hi->reg_num() != rd->reg_num())
          << "original register IDs were not restored";
    }
    if (pass == 0) continue;
    ASSERT_TRUE(!func->is_pdom_set())
        << "replacement retained old control-flow state";
    func->do_pdom();
    func->set_pdom();
    Scoreboard scoreboard(0, 1, nullptr);
    unsigned count = 0;
    for (unsigned pc = func->get_start_PC();;) {
      const auto *inst = func->get_dyn_inst(pc);
      ASSERT_TRUE(inst != nullptr) << "missing assembled instruction";
      std::set<unsigned> outputs;
      for (unsigned reg : inst->out)
        ASSERT_TRUE(!reg || outputs.insert(reg).second)
            << "duplicate decoded output";
      scoreboard.reserveRegistersForWarp(inst, 0);
      scoreboard.releaseRegistersForWarp(inst, 0);
      ++count;
      if (inst->get_opcode() == RET_OP) break;
      ASSERT_TRUE(count < 10)
          << "replacement instruction stream did not terminate";
      pc += inst->inst_size();
    }
    ASSERT_TRUE(count == (optimized ? 3u : 4u))
        << "replacement retained an eliminated unpack or skipped assembly";
  }
}
TEST(RegisterViewReloadTest, RestoresScalarCopyRegisterIds) {
  gpgpu_context *ctx = GPGPU_Context();
  GPGPUSim_Context(ctx);
  ctx->ptx_reorder_sass_guided = false;
  function_info *first = nullptr;
  for (bool optimized : {true, false, true}) {
    ctx->ptx_reorder_enabled = optimized;
    const char *path = "scalar_copy_reload.ptx";
    {
      std::ofstream fixture(path);
      fixture << ".version 8.0\n.target sm_80\n.address_size 64\n"
                 ".visible .entry reload_scalar_copy() {\n"
                 ".reg .b32 %source, %copy, %result;\n"
                 "mov.b32 %source, 7;\nmov.b32 %copy, %source;\n"
                 "add.u32 %result, %copy, 1;\nret;\n}\n";
      ASSERT_TRUE(fixture.good());
    }
    auto *symbols = ctx->gpgpu_ptx_sim_load_ptx_from_filename(path);
    std::remove(path);
    ASSERT_NE(symbols, nullptr);
    auto *func = symbols->lookup_function("reload_scalar_copy");
    ASSERT_NE(func, nullptr);
    if (!first) first = func;
    ASSERT_EQ(func, first);
    const auto *source = func->get_symtab()->lookup("%source");
    const auto *copy = func->get_symtab()->lookup("%copy");
    EXPECT_EQ(source->reg_num() == copy->reg_num(), optimized);
    EXPECT_EQ(func->get_compiler_register_view(copy, nullptr, nullptr), optimized);
  }
}
