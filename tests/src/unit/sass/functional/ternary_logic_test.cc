#include "../test_support.h"
#include "gpgpu-sim/flash/ternary_logic.h"
namespace {
TEST(TernaryLogicCommon, AllTruthTables) {
  for (unsigned lut=0;lut<256;++lut)
    for (unsigned input=0;input<8;++input)
      EXPECT_EQ(flash_gpgpu_sim::ternary_logic(input&4?~0u:0,input&2?~0u:0,input&1?~0u:0,lut),
                lut&(1u<<input)?~0u:0) << lut << ':' << input;
}
TEST(TernaryLogicNative, PredicateAliasAndUniform) {
  for (bool uniform : {false,true})
    for (bool use_and : {false,true}) {
      if (uniform && use_and) continue; // CUDA exposes ULOP3.LUT (OR), not PAND.
      for (bool q : {false,true}) for (bool nonzero : {false,true}) {
        kernel image;image.name="lut";image.arch=architecture::kSm100;image.instruction_bytes=16;
        image.instructions={official_instruction(0,uniform?"ULOP3.LUT":use_and?"LOP3.LUT.PAND":"LOP3.LUT",
                              uniform?"UP0,UR0,UR0,UR1,UR2,0xf0,!UP0":"P0,R0,R0,R1,R2,0xf0,!P0")};
        frontend sass;sass.add_kernel(std::move(image));register_sm100_functional_semantics(sass);
        warp_state state;state.active_mask=2;
        if(uniform) {
          state.write_uniform_register(0,nonzero?7:0);state.write_uniform_register(1,0);state.write_uniform_register(2,0);
          state.write_uniform_predicate(0,q);
        } else {
          state.write_register(0,0,123);state.write_predicate(0,0,q);
          state.write_register(1,0,nonzero?7:0);state.write_register(1,1,0);state.write_register(1,2,0);
          state.write_predicate(1,0,q);
        }
        execution_context context;const auto result=sass.step("lut",state,context);
        ASSERT_EQ(result.status,step_status::kAdvanced) << result.detail;
        const bool expected=use_and?nonzero&&!q:nonzero||!q;
        EXPECT_EQ(uniform?state.read_uniform_predicate(0):state.read_predicate(1,0),expected);
        if(!uniform) {EXPECT_EQ(state.read_register(0,0),123u);EXPECT_EQ(state.read_predicate(0,0),q);}
      }
    }
}
TEST(TernaryLogicNative, GuardedOffPreservesOutputs) {
  kernel image;image.name="guard";image.arch=architecture::kSm100;image.instruction_bytes=16;
  image.instructions={official_instruction(0,"LOP3.LUT","P0,R0,R0,R1,R2,0xff,P0","@P1")};
  frontend sass;sass.add_kernel(std::move(image));register_sm100_functional_semantics(sass);
  warp_state state;state.active_mask=1;state.write_predicate(0,1,false);
  state.write_predicate(0,0,false);state.write_register(0,0,17);
  execution_context context;
  EXPECT_EQ(sass.step("guard",state,context).status,step_status::kAdvanced);
  EXPECT_EQ(state.read_register(0,0),17u);EXPECT_FALSE(state.read_predicate(0,0));
}
TEST(TernaryLogicNative, InvalidLut) {
  for (const char *operand : {"R0,R1,R2,R3,0x100,!PT", "R0,R1,R2,R3,-1,!PT"}) {
    kernel image;image.name="invalid";image.arch=architecture::kSm100;image.instruction_bytes=16;
    image.instructions={official_instruction(0,"LOP3.LUT",operand)};
    frontend sass;sass.add_kernel(std::move(image));register_sm100_functional_semantics(sass);
    warp_state state;execution_context context;
    EXPECT_EQ(sass.step("invalid",state,context).status,step_status::kUnsupported);
  }
}
}
