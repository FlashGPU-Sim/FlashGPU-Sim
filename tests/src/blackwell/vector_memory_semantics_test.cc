#include <gtest/gtest.h>

#include <unistd.h>
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

struct CUctx_st;
CUctx_st *GPGPUSim_Context(gpgpu_context *ctx);

namespace {
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
