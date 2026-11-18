#include <gtest/gtest.h>

#include <unistd.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "../../../libcuda/gpgpu_context.h"
#include "cuda-sim/memory.h"
#include "cuda-sim/ptx_ir.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/scoreboard.h"

struct CUctx_st;
CUctx_st *GPGPUSim_Context(gpgpu_context *ctx);

namespace {
function_info *load_function(gpgpu_context *ctx, const std::string &name,
                             const std::string &body) {
  char path[] = "/tmp/flashgpu-vector-regression-XXXXXX";
  const int fd = mkstemp(path);
  if (fd < 0) {
    ADD_FAILURE() << "could not create PTX fixture";
    return nullptr;
  }
  close(fd);
  {
    std::ofstream fixture(path);
    fixture << ".version 8.8\n.target sm_120a\n.address_size 64\n"
            << ".entry " << name << "() { " << body << " ret; }\n";
    if (!fixture.good()) {
      ADD_FAILURE() << "could not write PTX fixture";
      std::remove(path);
      return nullptr;
    }
  }
  auto *symbols = ctx->gpgpu_ptx_sim_load_ptx_from_filename(path);
  std::remove(path);
  if (symbols == nullptr) {
    ADD_FAILURE() << "could not parse PTX fixture";
    return nullptr;
  }
  auto *function = symbols->lookup_function(name.c_str());
  if (function != nullptr) {
    function->do_pdom();
    function->set_pdom();
  }
  return function;
}

class ObservedMemory : public memory_space_impl<8192> {
 public:
  ObservedMemory() : memory_space_impl<8192>("vector-test", 64) {}
  void read(mem_addr_t addr, size_t size, void *data) const override {
    reads.emplace_back(addr, size);
    memory_space_impl<8192>::read(addr, size, data);
  }
  void write(mem_addr_t addr, size_t size, const void *data,
             ptx_thread_info *thread, const ptx_instruction *inst) override {
    writes.emplace_back(addr, size);
    memory_space_impl<8192>::write(addr, size, data, thread, inst);
  }
  mutable std::vector<std::pair<mem_addr_t, size_t>> reads;
  std::vector<std::pair<mem_addr_t, size_t>> writes;
};
class ObservedGpu : public gpgpu_t {
 public:
  ObservedGpu(const gpgpu_functional_sim_config &config, gpgpu_context *ctx)
      : gpgpu_t(config, ctx) {
    delete m_global_mem;
    m_global_mem = &memory;
  }
  ~ObservedGpu() override {
    delete m_tex_mem;
    delete m_surf_mem;
  }
  ObservedMemory memory;
};
class VectorCore : public core_t {
 public:
  VectorCore(gpgpu_sim *gpu, kernel_info_t *kernel)
      : core_t(gpu, kernel, 32, 32) {}
  ~VectorCore() override { deleteSIMTStack(); }
  void warp_exit(unsigned) override {}
  bool warp_waiting_at_barrier(unsigned) const override { return false; }
  void checkExecutionStatusAndUpdate(warp_inst_t &, unsigned,
                                     unsigned) override {}
};

TEST(BlackwellVectorLdstIntegrationTest, StoreCollectsEveryDataRegister) {
  auto *ctx = GPGPU_Context();
  GPGPUSim_Context(ctx);
  for (unsigned count : {2u, 4u, 8u}) {
    SCOPED_TRACE(count);
    std::string registers = "{";
    for (unsigned i = 0; i < count; ++i) {
      if (i != 0) registers += ',';
      registers += "%r" + std::to_string(i);
    }
    registers += '}';
    const std::string name = "vector_store_collector_" + std::to_string(count);
    auto *function = load_function(
        ctx, name, ".reg .b64 %rd99; .reg .b32 %r<8>; st.global.v" +
                  std::to_string(count) + ".b32 [%rd99], " + registers + ";");
    ASSERT_NE(function, nullptr);
    const auto *instruction = function->get_dyn_inst(function->get_start_PC());
    ASSERT_NE(instruction, nullptr);
    const auto *address = function->get_symtab()->lookup("%rd99");
    ASSERT_EQ(instruction->ar1, address->reg_num());
    EXPECT_NE(std::find(std::begin(instruction->arch_reg.src),
                        std::end(instruction->arch_reg.src),
                        static_cast<int>(address->arch_reg_num())),
              std::end(instruction->arch_reg.src));
    for (unsigned i = 0; i < count; ++i) {
      const auto *reg = function->get_symtab()->lookup(
          ("%r" + std::to_string(i)).c_str());
      EXPECT_NE(std::find(std::begin(instruction->arch_reg.src),
                          std::end(instruction->arch_reg.src),
                          static_cast<int>(reg->arch_reg_num())),
                std::end(instruction->arch_reg.src))
          << "missing collector source " << i;
    }
    EXPECT_EQ(std::count_if(std::begin(instruction->arch_reg.src),
                            std::end(instruction->arch_reg.src),
                            [](int reg) { return reg >= 0; }),
              count + 1);
  }
}

TEST(BlackwellVectorLdstIntegrationTest, SinkLoadsCheckEveryOutputHazard) {
  auto *ctx = GPGPU_Context();
  GPGPUSim_Context(ctx);
  auto *gpu = ctx->the_gpgpusim->g_the_gpu;
  for (unsigned bits : {32u, 64u}) {
    const unsigned count = 256 / bits;
    for (unsigned sink_mask : {0u, 1u, 2u, 1u << (count - 1), 5u}) {
      SCOPED_TRACE(::testing::Message() << bits << ':' << sink_mask);
      std::string registers = "{";
      for (unsigned i = 0; i < count; ++i) {
        if (i != 0) registers += ',';
        registers += sink_mask & (1u << i) ? "_" : "%r" + std::to_string(i);
      }
      registers += '}';
      const std::string name = "vector_load_hazards_" + std::to_string(bits) +
                               "_" + std::to_string(sink_mask);
      auto *function = load_function(
          ctx, name, ".reg .b64 %rd99; .reg .b" + std::to_string(bits) +
                    " %r<8>; ld.global.v" + std::to_string(count) + ".b" +
                    std::to_string(bits) + " " + registers + ", [%rd99];");
      ASSERT_NE(function, nullptr);
      const auto *instruction = function->get_dyn_inst(function->get_start_PC());
      ASSERT_NE(instruction, nullptr);
      warp_inst_t producer(gpu->getShaderCoreConfig());
      static_cast<inst_t &>(producer) =
          static_cast<const inst_t &>(*instruction);
      std::fill_n(producer.out, MAX_OUTPUT_VALUES, 0);
      producer.outcount = 1;
      active_mask_t active;
      active.set(0);
      producer.issue(active, 0, 0, 0, 0, 0);
      for (unsigned i = 0; i < count; ++i) {
        if (sink_mask & (1u << i)) continue;
        SCOPED_TRACE(i);
        const auto *reg = function->get_symtab()->lookup(
            ("%r" + std::to_string(i)).c_str());
        producer.out[0] = reg->reg_num();
        Scoreboard scoreboard(0, 1, nullptr);
        EXPECT_FALSE(scoreboard.checkCollision(0, instruction));
        scoreboard.reserveRegistersForWarp(&producer, 0);
        EXPECT_TRUE(scoreboard.checkCollision(0, instruction));
        EXPECT_EQ(scoreboard.getCollisionType(0, instruction), PROD_MEM_GLOBAL);
        scoreboard.releaseRegistersForWarp(&producer, 0);
        EXPECT_FALSE(scoreboard.checkCollision(0, instruction));
      }
    }
  }
}

TEST(BlackwellVectorLdstIntegrationTest, LiteralVectorStoresWriteEveryElement) {
  auto *ctx = GPGPU_Context();
  GPGPUSim_Context(ctx);
  auto *gpu = ctx->the_gpgpusim->g_the_gpu;
  ObservedGpu observed(gpu->get_config(), ctx);
  for (unsigned bits : {32u, 64u}) {
    const unsigned count = 256 / bits;
    const unsigned bytes = bits / 8;
    for (unsigned literal_mask : {0u, 2u, (1u << count) - 1}) {
      SCOPED_TRACE(::testing::Message() << bits << ':' << literal_mask);
      std::string values = "{";
      for (unsigned i = 0; i < count; ++i) {
        if (i != 0) values += ',';
        values += literal_mask & (1u << i) ? std::to_string(1000 + i)
                                          : "%r" + std::to_string(i);
      }
      values += '}';
      const std::string name = "vector_store_literals_" + std::to_string(bits) +
                               "_" + std::to_string(literal_mask);
      auto *function = load_function(
          ctx, name, ".reg .b64 %rd99; .reg .b" + std::to_string(bits) +
                    " %r<8>; st.global.v" + std::to_string(count) + ".b" +
                    std::to_string(bits) + " [%rd99], " + values + ";");
      ASSERT_NE(function, nullptr);
      kernel_info_t kernel(dim3(1), dim3(32), function, 0ull);
      VectorCore core(gpu, &kernel);
      ptx_thread_info thread(kernel);
      thread.init(&observed, &core, 0, 0, 0, 0, true);
      thread.set_tid(dim3(0, 0, 0));
      thread.set_ntid(dim3(32, 1, 1));
      thread.set_ctaid(dim3(0, 0, 0));
      thread.set_info(function);
      for (unsigned i = 0; i < count; ++i) {
        ptx_reg_t value;
        value.u64 = 100 + i;
        thread.set_reg(function->get_symtab()->lookup(
                           ("%r" + std::to_string(i)).c_str()), value);
      }
      constexpr uint64_t base = 0x10000;
      ptx_reg_t address;
      address.u64 = base;
      thread.set_reg(function->get_symtab()->lookup("%rd99"), address);
      const auto *instruction = function->get_dyn_inst(function->get_start_PC());
      ASSERT_NE(instruction, nullptr);
      warp_inst_t dynamic(gpu->getShaderCoreConfig());
      static_cast<inst_t &>(dynamic) =
          static_cast<const inst_t &>(*instruction);
      active_mask_t active;
      active.set(0);
      dynamic.issue(active, 0, 0, 0, 0, 0);
      observed.memory.reads.clear();
      observed.memory.writes.clear();
      thread.ptx_exec_inst(dynamic, 0);
      std::vector<std::pair<mem_addr_t, size_t>> expected_writes;
      for (unsigned i = 0; i < count; ++i) {
        expected_writes.emplace_back(base + i * bytes, bytes);
        uint64_t actual = 0;
        observed.memory.memory_space_impl<8192>::read(base + i * bytes, bytes,
                                                       &actual);
        EXPECT_EQ(actual, literal_mask & (1u << i) ? 1000 + i : 100 + i);
      }
      EXPECT_EQ(observed.memory.writes, expected_writes);
      EXPECT_TRUE(observed.memory.reads.empty());
      dynamic.generate_mem_accesses();
      ASSERT_EQ(dynamic.accessq_count(), 1);
      EXPECT_EQ(dynamic.accessq_back().get_byte_mask().count(), 32);
    }
  }
}

TEST(BlackwellVectorLdstIntegrationTest, SinkElementsDoNotAccessMemory) {
  auto *ctx = GPGPU_Context();
  GPGPUSim_Context(ctx);
  auto *gpu = ctx->the_gpgpusim->g_the_gpu;
  ObservedGpu observed(gpu->get_config(), ctx);
  // First, middle and last sinks, plus an all-register control. Also cover
  // the other 256-bit form for which PTX 8.8 defines sink memory semantics.
  for (unsigned bits : {32u, 64u}) {
    const unsigned count = 256 / bits, bytes = bits / 8;
    for (unsigned sink_mask : {0u, 1u, 2u, 1u << (count - 1), 5u}) {
      for (bool store : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << bits << ":" << sink_mask << ":" << store);
        const std::string name = "vector_sinks_" + std::to_string(bits) + "_" +
                                 std::to_string(sink_mask) +
                                 (store ? "st" : "ld");
        std::string regs = "{";
        for (unsigned i = 0; i < count; ++i) {
          if (i) regs += ',';
          regs += sink_mask & (1u << i) ? "_" : "%r" + std::to_string(i);
        }
        regs += '}';
        char path[] = "/tmp/flashgpu-vector-sinks-XXXXXX";
        const int fd = mkstemp(path);
        ASSERT_GE(fd, 0);
        close(fd);
        std::ofstream(path)
            << ".version 8.8\n.target sm_120a\n.address_size 64\n"
            << ".entry " << name << "() { .reg .b64 %rd; .reg .b" << bits
            << " %r<8>; " << (store ? "st" : "ld") << ".global.v" << count
            << ".b" << bits << " "
            << (store ? "[%rd]," + regs : regs + ",[%rd]") << "; ret; }\n";
        auto *symbols = ctx->gpgpu_ptx_sim_load_ptx_from_filename(path);
        std::remove(path);
        ASSERT_NE(symbols, nullptr);
        auto *func = symbols->lookup_function(name.c_str());
        ASSERT_NE(func, nullptr);
        func->do_pdom();
        func->set_pdom();
        kernel_info_t kernel(dim3(1), dim3(32), func, 0ull);
        VectorCore core(gpu, &kernel);
        ptx_thread_info thread(kernel);
        thread.init(&observed, &core, 0, 0, 0, 0, true);
        thread.set_tid(dim3(0, 0, 0));
        thread.set_ntid(dim3(32, 1, 1));
        thread.set_ctaid(dim3(0, 0, 0));
        thread.set_info(func);
        constexpr uint64_t base = 0x10000;
        for (unsigned i = 0; i < count; ++i) {
          ptx_reg_t reg;
          reg.u64 = 100 + i;
          thread.set_reg(
              func->get_symtab()->lookup(("%r" + std::to_string(i)).c_str()),
              reg);
          const uint64_t value = 1000 + i;
          observed.memory.memory_space_impl<8192>::write(
              base + i * bytes, bytes, &value, nullptr, nullptr);
        }
        observed.memory.reads.clear();
        observed.memory.writes.clear();
        ptx_reg_t address;
        address.u64 = base;
        thread.set_reg(func->get_symtab()->lookup("%rd"), address);
        const auto *inst = func->get_dyn_inst(func->get_start_PC());
        warp_inst_t dynamic(gpu->getShaderCoreConfig());
        static_cast<inst_t &>(dynamic) = static_cast<const inst_t &>(*inst);
        active_mask_t active;
        active.set(0);
        dynamic.issue(active, 0, 0, 0, 0, 0);
        thread.ptx_exec_inst(dynamic, 0);
        std::vector<std::pair<mem_addr_t, size_t>> expected;
        for (unsigned i = 0; i < count; ++i) {
          const bool sink = sink_mask & (1u << i);
          if (!sink) expected.emplace_back(base + i * bytes, bytes);
          uint64_t value = 0;
          observed.memory.memory_space_impl<8192>::read(base + i * bytes, bytes,
                                                        &value);
          EXPECT_EQ(value, store && !sink ? 100 + i : 1000 + i);
          if (!store && !sink)
            EXPECT_EQ(thread
                          .get_reg(func->get_symtab()->lookup(
                              ("%r" + std::to_string(i)).c_str()))
                          .u64,
                      1000 + i);
        }
        EXPECT_EQ(store ? observed.memory.writes : observed.memory.reads,
                  expected);
        EXPECT_TRUE(store ? observed.memory.reads.empty()
                          : observed.memory.writes.empty());
        dynamic.generate_mem_accesses();
        ASSERT_FALSE(dynamic.accessq_empty());
        const auto mask = dynamic.accessq_back().get_byte_mask();
        for (unsigned i = 0; i < 32; ++i)
          EXPECT_EQ(mask.test(i), !(sink_mask & (1u << (i / bytes))));
        EXPECT_EQ(mask.count(), expected.size() * bytes);
        dynamic.accessq_pop_back();
        EXPECT_TRUE(dynamic.accessq_empty());
      }
    }
  }
}
}  // namespace
