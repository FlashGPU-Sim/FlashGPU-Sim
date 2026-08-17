#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>

#include "libcuda/gpgpu_context.h"
#include "ptx.tab.h"
#include "src/gpgpu-sim/scoreboard.h"

CUctx_st *GPGPUSim_Context(gpgpu_context *ctx);

namespace {

[[noreturn]] void fail(const std::string &message) {
  std::fprintf(stderr, "vector literal parser smoke failed: %s\n",
               message.c_str());
  std::exit(1);
}

void expect_input_registers(const ptx_instruction *inst,
                            std::initializer_list<const symbol *> expected) {
  if (inst->incount != expected.size())
    fail("predecode input count is " + std::to_string(inst->incount) +
         ", expected " + std::to_string(expected.size()));

  unsigned index = 0;
  for (const symbol *reg : expected) {
    if (inst->in[index] != reg->reg_num() ||
        inst->arch_reg.src[index] != static_cast<int>(reg->arch_reg_num()))
      fail("predecode did not compact a vector source register");
    ++index;
  }
}

void expect_scoreboard_raw_hazard(const symbol *pending_reg,
                                  const ptx_instruction *consumer) {
  warp_inst_t producer;
  std::fill_n(producer.out, MAX_OUTPUT_VALUES, 0);
  producer.out[0] = pending_reg->reg_num();
  producer.outcount = 1;

  Scoreboard scoreboard(0, 1, nullptr);
  scoreboard.reserveRegistersForWarp(&producer, 0);
  if (!scoreboard.checkCollision(0, consumer))
    fail("consumer issued while its vector source producer was pending");
  scoreboard.releaseRegistersForWarp(&producer, 0);
  if (scoreboard.checkCollision(0, consumer))
    fail("consumer remained blocked after its producer completed");
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <ptx-file>\n", argv[0]);
    return 2;
  }

  gpgpu_context *ctx = GPGPU_Context();
  GPGPUSim_Context(ctx);
  symbol_table *symtab = ctx->gpgpu_ptx_sim_load_ptx_from_filename(argv[1]);
  if (symtab == nullptr)
    fail("parser returned a null symbol table");
  symtab->lookup_function("vector_literal_valid")->do_pdom();

  std::ifstream input(argv[1]);
  if (!input) fail("could not reopen PTX fixture");
  std::vector<const ptx_instruction *> stores;
  std::string line;
  unsigned line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.find("st.local.v") == std::string::npos) continue;
    const ptx_instruction *inst =
        ctx->ptx_parser->ptx_instruction_lookup(argv[1], line_number);
    if (inst == nullptr)
      fail("missing parsed store at line " + std::to_string(line_number));
    stores.push_back(inst);
  }
  if (stores.size() != 7) fail("expected seven vector stores");

  for (const ptx_instruction *inst : stores) {
    const operand_info &source = inst->src1();
    if (!source.is_vector())
      fail("store source was not preserved as a vector operand");
  }

  const operand_info &zeros = stores[0]->src1();
  if (!zeros.vec_is_literal(0) || !zeros.vec_is_literal(1) ||
      zeros.vec_component_type(0) != int_t ||
      zeros.vec_component_type(1) != int_t ||
      zeros.vec_literal_value(0).s64 != 0 ||
      zeros.vec_literal_value(1).s64 != 0)
    fail("{0, 0} did not retain two typed zero literals");

  const operand_info &mixed = stores[1]->src1();
  if (mixed.vec_is_literal(0) || !mixed.vec_is_literal(1) ||
      mixed.vec_symbol(0)->name() != "%r1" ||
      mixed.vec_literal_value(1).s64 != 0)
    fail("mixed register/literal vector metadata is incorrect");

  const operand_info &literal_first = stores[2]->src1();
  if (!literal_first.vec_is_literal(0) || literal_first.vec_is_literal(1) ||
      literal_first.vec_literal_value(0).s64 != 0 ||
      literal_first.vec_symbol(1)->name() != "%r1")
    fail("literal/register vector metadata is incorrect");

  const operand_info &registers = stores[3]->src1();
  if (registers.vector_has_literal() ||
      registers.vec_symbol(0)->name() != "%r1" ||
      registers.vec_symbol(1)->name() != "%r2")
    fail("identifier-only vector regressed");

  const operand_info &middle = stores[4]->src1();
  if (middle.get_vect_nelem() != 4 || !middle.vec_is_literal(0) ||
      middle.vec_is_literal(1) || !middle.vec_is_literal(2) ||
      middle.vec_is_literal(3))
    fail("v4 literal-middle vector metadata is incorrect");

  const symbol *r1 = mixed.vec_symbol(0);
  const symbol *r2 = registers.vec_symbol(1);
  expect_input_registers(stores[1], {r1});
  expect_input_registers(stores[2], {r1});
  expect_input_registers(stores[3], {r1, r2});
  expect_input_registers(stores[4], {r1, r2});
  expect_scoreboard_raw_hazard(r1, stores[1]);
  expect_scoreboard_raw_hazard(r1, stores[2]);
  expect_scoreboard_raw_hazard(r2, stores[3]);
  expect_scoreboard_raw_hazard(r2, stores[4]);

  const operand_info &f32 = stores[5]->src1();
  if (f32.vec_is_literal(0) || !f32.vec_is_literal(1) ||
      f32.vec_component_type(1) != float_op_t ||
      std::fabs(f32.vec_literal_value(1).f32 - 2.0f) > 0.0f)
    fail("contextual f32 vector literal conversion failed");

  const operand_info &f64 = stores[6]->src1();
  if (!f64.vec_is_literal(0) || !f64.vec_is_literal(1) ||
      f64.vec_component_type(0) != double_op_t ||
      f64.vec_component_type(1) != double_op_t ||
      f64.vec_literal_value(0).f64 != 1.0 ||
      f64.vec_literal_value(1).f64 != 2.0)
    fail("f64 vector literal bits were not preserved");

  std::printf("parsed and validated %s\n", argv[1]);
  return 0;
}
