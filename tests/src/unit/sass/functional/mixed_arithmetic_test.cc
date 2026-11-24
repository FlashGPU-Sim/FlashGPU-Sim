#include "libcuda/gpgpu_context.h"
#include "ptx.tab.h"
#include "../test_support.h"
CUctx_st *GPGPUSim_Context(gpgpu_context *ctx);
#include "gpgpu-sim/flash/mixed_precision.h"
#include "gpgpu-sim/flash/sass/decode/mixed_precision_operand.h"
namespace {
TEST(MixedArithmeticNative, ModifiersSelectorsAliasAndInactiveLanes) {
  for (bool bf : {false,true})
    for (bool fused : {false,true})
      for (const std::string rounding : {"",".RM",".RP",".RZ"})
        for (bool sat : {false,true}) {
          const std::string opcode=std::string(fused?"FHFMA":"FHADD")+(bf?".BF16":"")+rounding+(sat?".SAT":"");
          kernel image;image.name="mixed";image.arch=architecture::kSm100;image.instruction_bytes=16;
          image.instructions={official_instruction(0,opcode,fused?"R2,R2.H1,-|R3.H1|,R4":"R2,R2.H1,R4")};
          frontend sass;sass.add_kernel(std::move(image));register_sm100_functional_semantics(sass);
          warp_state state;state.active_mask=2;
          const unsigned a=(bf?0x3fa0u:0x3d00u)<<16;
          state.write_register(0,2,0x12345678);
          state.write_register(1,2,a);state.write_register(1,3,(bf?0x4000u:0x4000u)<<16);
          state.write_register(1,4,0x40400000); // 3.0
          execution_context context;
          const auto result=sass.step("mixed",state,context);
          ASSERT_EQ(result.status,step_status::kAdvanced) << opcode << ':' << result.detail;
          EXPECT_EQ(state.read_register(0,2),0x12345678u);
          EXPECT_EQ(state.read_register(1,2),fused?0x3f000000u:sat?0x3f800000u:0x40880000u);
        }
}
TEST(MixedArithmeticCommon, DirectedTiesSignedZeroAndRoundingRestore) {
  using flash_gpgpu_sim::mixed_f32_arithmetic;
  using flash_gpgpu_sim::mixed_operation;
  const int saved=std::fegetround();std::fesetround(FE_DOWNWARD);
  const float half_ulp=std::ldexp(1.0f,-24);
  EXPECT_EQ(mixed_f32_arithmetic(mixed_operation::add,1,1,half_ulp,FE_TONEAREST,false),1.0f);
  EXPECT_EQ(mixed_f32_arithmetic(mixed_operation::add,1,1,half_ulp,FE_UPWARD,false),std::nextafter(1.0f,INFINITY));
  EXPECT_EQ(std::fegetround(),FE_DOWNWARD);
  EXPECT_TRUE(std::signbit(mixed_f32_arithmetic(mixed_operation::subtract,1,1,1,FE_DOWNWARD,false)));
  EXPECT_FALSE(std::signbit(mixed_f32_arithmetic(mixed_operation::subtract,1,1,1,FE_TONEAREST,false)));
  EXPECT_EQ(mixed_f32_arithmetic(mixed_operation::add,NAN,1,0,FE_TONEAREST,true),0.0f);
  std::fesetround(saved);
}
TEST(MixedArithmeticPtx, LiteralAccumulatorIsConvertedAsFp32Only) {
  auto *ctx=GPGPU_Context();GPGPUSim_Context(ctx);
  const char *path="mixed_literal_predecode.ptx";
  { std::ofstream f(path);f << R"(.version 9.0
.target sm_100
.address_size 64
.visible .entry mixed_literals() {
.reg .b16 a,b;
.reg .f32 d;
add.rn.f32.bf16 d,a,0.5;
sub.rp.f32.f16 d,a,0.5;
fma.rm.f32.bf16 d,a,b,0.5;
ret;
})"; }
  auto *symbols=ctx->gpgpu_ptx_sim_load_ptx_from_filename(path);
  std::remove(path);ASSERT_NE(symbols,nullptr);
  auto *func=symbols->lookup_function("mixed_literals");ASSERT_NE(func,nullptr);
  func->do_pdom();unsigned pc=func->get_start_PC();
  for(unsigned i=0;i<3;++i) {
    const auto *inst=func->get_dyn_inst(pc);ASSERT_NE(inst,nullptr);
    const auto &literal=i==2?inst->src3():inst->src2();
    EXPECT_EQ(literal.get_type(),float_op_t);
    EXPECT_EQ(literal.get_literal_value().f32,0.5f);
    EXPECT_EQ(inst->get_type(),F32_TYPE);
    pc+=inst->inst_size();
  }
}
} // namespace
