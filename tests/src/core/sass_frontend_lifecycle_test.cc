#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <new>
#include <string>

#include "gpgpu-sim/flash/frontend/execution_frontend.h"
#include "gpgpu-sim/flash/sass/runtime/runtime_adapter.h"
#include "gpgpu-sim/flash/tma.h"
#include "gpgpu-sim/flash/wgmma/tensor_wgmma.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "gpgpu-sim/mem_fetch.h"
#include "kernel_info.h"
#include "libcuda/gpgpu_context.h"
#include "option_parser.h"

namespace {
class BarrierIssueCore : public exec_shader_core_ctx {
 public:
  static void issue(exec_shader_core_ctx &core, const warp_inst_t *inst,
                    const active_mask_t &mask, unsigned warp) {
    register_set pipe(1, "native-barrier-test");
    // Native issue must not depend on any PTX control-flow state. Make an
    // accidental stack access deterministic rather than reading an empty deque.
    auto &stack = (core.*(&BarrierIssueCore::m_simt_stack))[warp];
    auto *saved = stack;
    stack = nullptr;
    (core.*(&BarrierIssueCore::issue_warp))(pipe, inst, mask, warp, 0);
    stack = saved;
  }
  static void cycle_barriers(exec_shader_core_ctx &core) {
    (core.*(&BarrierIssueCore::m_barriers)).cycle();
  }
};

class TestGpu : public exec_gpgpu_sim {
 public:
  using exec_gpgpu_sim::exec_gpgpu_sim;
  exec_shader_core_ctx *core() {
    return static_cast<exec_shader_core_ctx *>(m_cluster[0]->get_core(0));
  }
};

void CheckNativeBarrierIssue(bool masked) {
  gpgpu_context ctx;
  gpgpu_sim_config config(&ctx);
  auto *options = option_parser_create();
  ctx.ptx_reg_options(options);
  ctx.instruction_timing.reg_options(options);
  flash_gpgpu_sim::sass::register_runtime_options(options);
  icnt_reg_options(options);
  config.reg_options(options);
  option_parser_cfgfile(options, "gpgpusim.config");
  const char *args[] = {"barrier-issue-test",
                        "-gpgpu_n_clusters",
                        "1",
                        "-gpgpu_n_cores_per_cluster",
                        "1",
                        "-network_mode",
                        "2",
                        "-gpgpu_named_barrier_arrive_latency",
                        "7",
                        "-gpgpu_named_barrier_arrive_visibility_latency",
                        "0"};
  option_parser_cmdline(options, sizeof(args) / sizeof(args[0]), args);
  config.init();
  TestGpu gpu(config, &ctx);
  gpu.init();
  char path[] = "/tmp/flashgpu-native-barrier-XXXXXX";
  const int fd = mkstemp(path);
  if (fd < 0) std::exit(2);
  close(fd);
  std::ofstream f(path);
  f << "SASSIR\t3\nDECODER\tnvdisasm\tversion\t1.0.0\n"
       "KERNEL\tbarrier_issue\tsm120\t16\nRESOURCES\t16\t0\t0\t0\n"
       "PARAMBANK\t0x380\t0x0\n"
       "INST\t0\t0\t0\t1\tnvdisasm\t"
    << (masked ? "@!PT" : "")
    << "\tBAR.SYNC.DEFER_BLOCKING\t0x0,0x40\t0\t1\t0\t0\t0\t0\nENDINST\n"
       "INST\t16\t0\t0\t1\tnvdisasm\t\tBAR.SYNC.DEFER_BLOCKING\t0x0,"
       "0x40\t0\t1\t0\t0\t0\t0\n"
       "ENDINST\nENDKERNEL\n";
  f.close();
  setenv("FLASHGPU_SASS_IR", path, 1);
  setenv("FLASHGPU_SASS_TIMING", "1", 1);
  kernel_info_t kernel(dim3(1), dim3(64), "barrier_issue", &ctx, 0, 98764);
  kernel.set_resource_usage({16, 0, 0, 0});
  auto *core = gpu.core();
  core->set_kernel(&kernel);
  core->issue_block2core(kernel);
  for (unsigned warp = 0; warp < 2; ++warp) {
    const auto *inst = core->get_next_inst(warp, 0);
    const auto mask = core->get_active_mask(warp, inst);
    BarrierIssueCore::issue(*core, inst, mask, warp);
    EXPECT_EQ(core->named_arrive_warp_ready(warp), masked);
    if (warp == 0) EXPECT_EQ(core->frontend_instruction_can_issue(0), masked);
  }
  for (unsigned tick = 0; tick < 256; ++tick) {
    ++gpu.gpu_sim_cycle;
    BarrierIssueCore::cycle_barriers(*core);
  }
  EXPECT_FALSE(core->warp_waiting_at_barrier(0));
  EXPECT_FALSE(core->warp_waiting_at_barrier(1));
  EXPECT_TRUE(core->frontend_instruction_can_issue(0));
  EXPECT_TRUE(core->frontend_instruction_can_issue(1));
  unsigned pc, rpc;
  core->get_pdom_stack_top_info(0, nullptr, &pc, &rpc);
  EXPECT_EQ(pc, 16u);
  std::remove(path);
  std::exit(testing::Test::HasFailure() ? 1 : 0);
}

void CheckClusterSizeQuery(bool native, bool timing, unsigned sms) {
  auto *ctx = GPGPU_Context();
  gpgpu_sim_config config(ctx);
  auto *options = option_parser_create();
  ctx->ptx_reg_options(options);
  ctx->instruction_timing.reg_options(options);
  flash_gpgpu_sim::sass::register_runtime_options(options);
  icnt_reg_options(options);
  config.reg_options(options);
  option_parser_cfgfile(options, "gpgpusim.config");
  const std::string count = std::to_string(sms);
  const char *args[] = {"cluster-query-test",
                        "-gpgpu_n_clusters",
                        "1",
                        "-gpgpu_n_cores_per_cluster",
                        count.c_str(),
                        "-network_mode",
                        "2"};
  option_parser_cmdline(options, 7, args);
  config.init();
  TestGpu gpu(config, ctx);
  // Bind the API to this subprocess's GPU without starting a simulation thread.
  auto *state = ctx->the_gpgpusim;
  state->g_the_gpu = &gpu;
  state->g_the_gpu_config = &config;
  state->the_cude_device = new _cuda_device_id(&gpu);
  state->the_context = new CUctx_st(state->the_cude_device);
  state->g_sim_done = false;
  char host_function;
  function_info function(1, ctx);
  if (native) {
    setenv("FLASHGPU_SASS_AUTO", "1", 1);
    setenv("FLASHGPU_SASS_TIMING", timing ? "1" : "", 1);
    state->the_context->register_sass_function(98765, &host_function,
                                               "cluster_query");
  } else {
    unsetenv("FLASHGPU_SASS_IR");
    unsetenv("FLASHGPU_SASS_AUTO");
    unsetenv("FLASHGPU_SASS_TIMING");
    state->the_context->register_hostFun_function(&host_function, &function);
  }
  cudaLaunchConfig_t launch{};
  launch.gridDim = dim3(32);
  launch.blockDim = dim3(32);
  int size = -1;
  if (native) {
    EXPECT_EQ(
        cudaOccupancyMaxPotentialClusterSize(&size, &host_function, &launch),
        cudaErrorNotSupported);
    EXPECT_EQ(size, -1);
  } else {
    for (bool opt_in : {false, true}) {
      function.set_nonportable_cluster_size_allowed(opt_in);
      EXPECT_EQ(
          cudaOccupancyMaxPotentialClusterSize(&size, &host_function, &launch),
          cudaSuccess);
      EXPECT_EQ(size, static_cast<int>(std::min(sms, opt_in ? 16u : 8u)));
    }
  }
  EXPECT_EQ(
      cudaOccupancyMaxPotentialClusterSize(nullptr, &host_function, &launch),
      cudaErrorInvalidValue);
  std::exit(testing::Test::HasFailure() ? 1 : 0);
}

TEST(SassFrontendLifecycleTest, NativeBarrierIssueDoesNotReadPtxStack) {
  ASSERT_EXIT(CheckNativeBarrierIssue(false), testing::ExitedWithCode(0), "");
}
TEST(SassFrontendLifecycleTest, MaskedNativeBarrierDoesNotParkWarp) {
  ASSERT_EXIT(CheckNativeBarrierIssue(true), testing::ExitedWithCode(0), "");
}
TEST(SassFrontendLifecycleTest, FunctionalClusterQueryReportsUnsupported) {
  ASSERT_EXIT(CheckClusterSizeQuery(true, false, 1), testing::ExitedWithCode(0),
              "");
}
TEST(SassFrontendLifecycleTest, TimingClusterQueryReportsUnsupported) {
  ASSERT_EXIT(CheckClusterSizeQuery(true, true, 1), testing::ExitedWithCode(0),
              "");
}
TEST(SassFrontendLifecycleTest,
     PtxClusterQueryPreservesPortableAndOptInLimits) {
  ASSERT_EXIT(CheckClusterSizeQuery(false, false, 18),
              testing::ExitedWithCode(0), "");
}
TEST(SassFrontendLifecycleTest, PtxClusterQueryRespectsPhysicalCapacity) {
  ASSERT_EXIT(CheckClusterSizeQuery(false, false, 1),
              testing::ExitedWithCode(0), "");
}

// Exercise the actual legacy bypass response path without issuing a second
// functional atomic or requiring a complete memory-system simulation.
class AtomicResponseLsu : public ldst_unit {
 public:
  AtomicResponseLsu(shader_core_ctx *core, const shader_core_config *config,
                    const memory_config *memory)
      : ldst_unit(nullptr, nullptr, core, nullptr, nullptr, config, memory,
                  nullptr, 0, 0, static_cast<l1_cache *>(nullptr)) {}
  void retire(mem_fetch *response) {
    m_next_global = response;
    writeback();
    m_next_wb.clear();
  }
};

enum class Scenario { kReusedAddress, kReset, kDistinctAddress, kSameLaunch,
                      kModuleGlobals, kStaticGlobalAllocation, kSharedGlobal, kGlobalSizeConflict,
                      kGlobalModuleConflict, kIdentity, kAtomic, kPredicatedAtomic,
                      kPtxNamedBarrier, kNativeNamedBarrier,
                      kPtxExitBarrier, kNativeExitBarrier,
                      kNativeTma, kPtxTma, kMaskedTma, kFastTma };

void CheckFrontend(Scenario scenario, unsigned lanes = 1) {
  gpgpu_context ctx;
  gpgpu_sim_config config(&ctx);
  auto *options = option_parser_create();
  ctx.ptx_reg_options(options);
  ctx.instruction_timing.reg_options(options);
  flash_gpgpu_sim::sass::register_runtime_options(options);
  icnt_reg_options(options);
  config.reg_options(options);
  option_parser_cfgfile(options, "gpgpusim.config");
  const char *args[] = {"frontend-test", "-gpgpu_n_clusters", "1",
                        "-gpgpu_n_cores_per_cluster", "1", "-network_mode", "2"};
  option_parser_cmdline(options, 7, args);
  const bool named_barrier = scenario == Scenario::kPtxNamedBarrier ||
      scenario == Scenario::kNativeNamedBarrier ||
      scenario == Scenario::kPtxExitBarrier || scenario == Scenario::kNativeExitBarrier;
  if (named_barrier) {
    const char *latency_args[] = {"barrier-test", "-gpgpu_barrier_release_latency",
                                  "7", "-gpgpu_cta_barrier_release_latency", "3"};
    option_parser_cmdline(options, 5, latency_args);
  }
  config.init();
  TestGpu gpu(config, &ctx);
  gpu.init();
  auto *core = gpu.core();
  if (named_barrier) {
    const bool native = scenario == Scenario::kNativeNamedBarrier ||
                        scenario == Scenario::kNativeExitBarrier;
    const bool exit = scenario == Scenario::kNativeExitBarrier ||
                      scenario == Scenario::kPtxExitBarrier;
    barrier_set_t barriers(core, 64, 32, 16, 32);
    warp_set_t warps; warps.set(0); warps.set(1);
    barriers.allocate_barrier(0, warps);
    warp_inst_t inst(gpu.getShaderCoreConfig());
    inst.op = BARRIER_OP;
    inst.bar_type = SYNC;
    inst.set_bar_id(0);
    inst.set_bar_count(static_cast<unsigned>(-1));
    if (native) inst.set_dependency_control({});
    barriers.warp_reaches_barrier(0, 0, &inst);
    if (exit) barriers.warp_exit(1);
    else barriers.warp_reaches_barrier(0, 1, &inst);
    const unsigned latency = native ? 3 : 7;
    bool ok = barriers.warp_waiting_at_barrier(0);
    for (unsigned i = 1; i <= latency; ++i) {
      barriers.cycle();
      ok &= barriers.warp_waiting_at_barrier(0) == (i < latency);
    }
    std::exit(ok ? 0 : 1);
  }
  if (scenario == Scenario::kStaticGlobalAllocation) {
    using namespace flash_gpgpu_sim::sass;
    const auto before = reinterpret_cast<uint64_t>(gpu.gpu_malloc(256));
    uint64_t first = 0, second = 0;
    register_runtime_device_global(98780, &first, "first", 8, &gpu);
    register_runtime_device_global(98781, &second, "second", 512, &gpu);
    const auto a = runtime_device_global_address(&first, 8, 0);
    const auto b = runtime_device_global_address(&second, 512, 0);
    const auto after = reinterpret_cast<uint64_t>(gpu.gpu_malloc(256));
    // Native static symbols belong to global static memory, not cudaMalloc's
    // heap. Header-only globals must not shift every workload tensor address.
    const bool ok = after == before + 256 && a < STATIC_ALLOC_LIMIT &&
        b < STATIC_ALLOC_LIMIT && a >= (1ull << 32) && b >= (1ull << 32) &&
        a % 256 == 0 && b % 256 == 0 && (a + 8 <= b || b + 512 <= a);
    std::exit(ok ? 0 : 1);
  }
  if (scenario == Scenario::kGlobalSizeConflict ||
      scenario == Scenario::kGlobalModuleConflict) {
    using namespace flash_gpgpu_sim::sass;
    uint64_t host_symbol = 0, unrelated_symbol = 0;
    register_runtime_device_global(98770, &host_symbol, "shared", 8, &gpu);
    if (scenario == Scenario::kGlobalModuleConflict)
      register_runtime_device_global(98771, &unrelated_symbol, "shared", 8, &gpu);
    register_runtime_device_global(98771, &host_symbol, "shared",
        scenario == Scenario::kGlobalSizeConflict ? 16 : 8, &gpu);
    std::exit(0);
  }
  if (scenario == Scenario::kSharedGlobal) {
    using namespace flash_gpgpu_sim::sass;
    // Header-defined inline/weak CUDA variables have one host identity but
    // registrations in every translation unit's fatbin.
    uint64_t host_symbol = 0, unrelated_symbol = 0;
    register_runtime_device_global(98770, &host_symbol, "shared", 8, &gpu);
    const auto address = runtime_device_global_address(&host_symbol, 8, 0);
    uint64_t value = 123;
    gpu.get_global_memory()->write(address, 8, &value, nullptr, nullptr);
    register_runtime_device_global(98771, &host_symbol, "other_TU_shared", 8, &gpu);
    register_runtime_device_global(98771, &host_symbol, "other_TU_shared", 8, &gpu);
    register_runtime_device_global(98772, &unrelated_symbol, "shared", 8, &gpu);
    bool ok = runtime_device_global_address(&host_symbol, 8, 0) == address;
    const auto unrelated = runtime_device_global_address(&unrelated_symbol, 8, 0);
    ok &= unrelated != address;
    gpu.get_global_memory()->read(address, 8, &value);
    ok &= value == 123;
    gpu.get_global_memory()->read(unrelated, 8, &value);
    ok &= value == 0;
    for (unsigned module : {98770u, 98771u}) {
      char path[] = "/tmp/flashgpu-shared-global-XXXXXX";
      int fd = mkstemp(path);
      if (fd < 0) std::exit(2);
      close(fd);
      setenv("FLASHGPU_SASS_IR", path, 1);
      ok &= runtime_sassir_path("reader", module) == path;
      kernel image;
      image.constant_banks.push_back({4, std::vector<uint8_t>(8)});
      image.constant_globals.push_back({4, 0, 8, 0,
          module == 98770 ? "shared" : "other_TU_shared", {}});
      relocate_runtime_kernel_globals(image, path);
      uint64_t pointer = 0;
      for (unsigned i = 0; i < 8; ++i)
        pointer |= uint64_t(image.constant_banks[0].data[i]) << (8 * i);
      ok &= pointer == address;
      std::remove(path);
    }
    std::exit(ok ? 0 : 1);
  }
  if (scenario == Scenario::kModuleGlobals) {
    using namespace flash_gpgpu_sim::sass;
    constexpr unsigned module = 98763;
    uint64_t host_symbol = 0;
    bind_runtime_device(module, &gpu);
    register_runtime_device_global(module, &host_symbol, "state", 16, &gpu);
    const auto address = runtime_device_global_address(&host_symbol, 8, 8);
    uint64_t value = 99;
    gpu.get_global_memory()->read(address, sizeof(value), &value);
    bool ok = value == 0;
    char path[] = "/tmp/flashgpu-global-reloc-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) std::exit(2);
    close(fd);
    setenv("FLASHGPU_SASS_IR", path, 1);
    ok &= runtime_sassir_path("first", module) == path;
    kernel image;
    image.constant_banks.push_back({4, std::vector<uint8_t>(16)});
    image.constant_globals.push_back({4, 0, 16, 8, "state", {}});
    image.constant_globals.push_back({4, 8, 4, 0, "local", {42, 0, 0, 0}});
    relocate_runtime_kernel_globals(image, path);
    auto pointer = [&](unsigned offset) {
      uint64_t p = 0;
      for (unsigned i = 0; i < 8; ++i)
        p |= uint64_t(image.constant_banks[0].data[offset + i]) << (8 * i);
      return p;
    };
    ok &= pointer(0) == address;
    const auto local = pointer(8);
    uint32_t initialized = 0;
    gpu.get_global_memory()->read(local, 4, &initialized);
    ok &= initialized == 42;
    initialized = 71;
    gpu.get_global_memory()->write(local, 4, &initialized, nullptr, nullptr);
    image.name = "second";
    relocate_runtime_kernel_globals(image, path);
    initialized = 0;
    gpu.get_global_memory()->read(local, 4, &initialized);
    ok &= pointer(0) == address && pointer(8) == local && initialized == 71;
    std::remove(path);
    std::exit(ok ? 0 : 1);
  }

  if (scenario == Scenario::kNativeTma || scenario == Scenario::kPtxTma ||
      scenario == Scenario::kMaskedTma || scenario == Scenario::kFastTma) {
    kernel_info_t launch(dim3(1), dim3(32), "tma_issue", &ctx, 0, 98762);
    core->set_kernel(&launch);
    barrier_set_t barriers(core, 64, 32, 16, 32);
    warp_set_t warps; warps.set(0);
    barriers.allocate_barrier(0, warps);
    flash_gpgpu_sim::tma_unit_t tma(core, &barriers, nullptr, nullptr);
    active_mask_t active;
    if (scenario != Scenario::kMaskedTma) active.set(0);
    warp_inst_t inst(gpu.getShaderCoreConfig());
    inst.issue(active, 0, 0, 0, 0, 0);
    inst.latency = scenario == Scenario::kFastTma ? 1 : 44;
    if (scenario == Scenario::kNativeTma) inst.set_dependency_control({});
    inst_t::tma_static_info_t info;
    info.tma_type = inst_t::tma_static_info_t::TMA_NORMAL;
    info.src_space = inst_t::tma_static_info_t::TMA_GLOBAL;
    info.dst_space = inst_t::tma_static_info_t::TMA_SHARED_CTA;
    inst.set_tma_static_info(info);
    if (active.any()) {
      inst_t::tma_dyn_info_t dynamic;
      dynamic.src_addr = 0x200000;
      dynamic.dst_addr = 0x400;
      dynamic.mbar_addr = 0x80;
      dynamic.size_in_bytes = 128;
      inst.set_tma_dyn_info(0, dynamic);
    }
    // Admit a real transaction, but do not advance the asynchronous engine:
    // neither frontend adds an execution-latency hold at this boundary.
    // The caller accounts for the configured issue-to-next-instruction delay;
    // this asynchronous submission must not charge that delay again.
    tma.warp_reaches_tma(0, 0, &inst);
    const bool expected = false;
    const bool held = barriers.warp_waiting_at_barrier(0);
    if (held != expected) fprintf(stderr, "TMA held=%d expected=%d\n", held, expected);
    std::exit(held == expected ? 0 : 1);
  }
  const uint64_t address = reinterpret_cast<uint64_t>(gpu.gpu_malloc(4));
  const uint32_t initial = 7;
  gpu.get_global_memory()->write(address, 4, &initial, nullptr, nullptr);
  char path[] = "/tmp/flashgpu-frontend-lifecycle-XXXXXX";
  const int fd = mkstemp(path);
  if (fd < 0) std::exit(2);
  close(fd);
  std::ofstream f(path);
  f << "SASSIR\t3\nDECODER\tnvdisasm\tversion\t1.0.0\n";
  auto begin = [&](const char *name) {
    f << "KERNEL\t" << name << "\tsm120\t16\nRESOURCES\t16\t0\t0\t0\n"
         "PARAMBANK\t0x380\t0x0\n";
  };
  auto inst = [&](unsigned pc, const char *op, const std::string &operands,
                  const char *guard = "") {
    f << "INST\t" << pc << "\t0\t0\t1\tnvdisasm\t" << guard << '\t'
      << op << '\t' << operands << "\t0\t1\t0\t0\t0\t0\nENDINST\n";
  };
  begin("first");
  inst(0, "MOV", "R2," + std::to_string(uint32_t(address)));
  inst(16, "MOV", "R3," + std::to_string(uint32_t(address >> 32)));
  inst(32, "MOV", "R4,42");
  inst(48, "ATOMG.E.EXCH.STRONG.GPU", "PT,RZ,desc[UR8][R2.64],R4",
       scenario == Scenario::kPredicatedAtomic ? "@!PT" : "");
  f << "ENDKERNEL\n";
  begin("second");
  inst(64, "NOP", "");
  f << "ENDKERNEL\n";
  f.close();
  setenv("FLASHGPU_SASS_IR", path, 1);
  setenv("FLASHGPU_SASS_TIMING", "1", 1);
  auto frontend = flash_gpgpu_sim::make_sass_frontend(*core);
  alignas(kernel_info_t) unsigned char storage[sizeof(kernel_info_t)];
  alignas(kernel_info_t) unsigned char alternate[sizeof(kernel_info_t)];
  auto *kernel = new (storage) kernel_info_t(dim3(64), dim3(lanes), "first",
                                            &ctx, 0, 98761);
  frontend->start_cta(0, 0, lanes, 37, lanes, *kernel);
  bool ok = true;
  if (scenario == Scenario::kAtomic || scenario == Scenario::kPredicatedAtomic) {
    // This fixture starts the adapter directly, bypassing normal CTA admission.
    // execute() still needs the owning kernel for cluster/DSM interpretation.
    core->set_kernel(kernel);
    active_mask_t mask;
    for (unsigned lane = 0; lane < lanes; ++lane) mask.set(lane);
    warp_inst_t issued(gpu.getShaderCoreConfig());
    for (unsigned pc = 0; pc <= 48; pc += 16) {
      warp_inst_t dynamic = *frontend->fetch(0, pc);
      dynamic.issue(mask, 0, pc, 0, 0, 0);
      frontend->execute(dynamic);
      if (pc == 48) issued = dynamic;
    }
    uint32_t value = 0;
    gpu.get_global_memory()->read(address, 4, &value);
    const unsigned expected = scenario == Scenario::kAtomic ? lanes : 0;
    const unsigned actual = core->get_shd_warp(0)->get_n_atomic();
    ok = actual == expected && value == (expected ? 42u : initial);
    if (!ok) fprintf(stderr, "atomic count=%u expected=%u value=%u\n",
                     actual, expected, value);
    if (ok) {
      AtomicResponseLsu lsu(core, gpu.getShaderCoreConfig(), gpu.getMemoryConfig());
      while (!issued.accessq_empty()) {
        auto *response = new mem_fetch(issued.accessq_back(), &issued, 0, 8,
                                      0, 0, 0, gpu.getMemoryConfig(), 0);
        issued.accessq_pop_back();
        lsu.retire(response);
      }
      ok &= core->get_shd_warp(0)->get_n_atomic() == 0;
      gpu.get_global_memory()->read(address, 4, &value);
      ok &= value == (expected ? 42u : initial);
      if (!ok) fprintf(stderr, "atomic response left count=%u value=%u\n",
                       core->get_shd_warp(0)->get_n_atomic(), value);
    }
  } else if (scenario == Scenario::kIdentity) {
    core->get_shd_warp(0)->set_done_exit();
    ok &= frontend->logical_cta_id(0) == 37 && frontend->cta_warp_id(0) == 0;
    frontend->release_cta(0);
    ok &= frontend->logical_cta_id(0) == -1 && frontend->cta_warp_id(0) == -1;
    ok &= frontend->logical_cta_id(63) == -1;
    frontend->start_cta(0, 0, lanes, 52, lanes, *kernel);
    ok &= frontend->logical_cta_id(0) == 52;
    frontend->reset();
    ok &= frontend->logical_cta_id(0) == -1;
  } else if (scenario == Scenario::kSameLaunch) {
    frontend->start_cta(1, 32, 32 + lanes, 38, lanes, *kernel);
    ok &= frontend->logical_cta_id(0) == 37 && frontend->logical_cta_id(1) == 38;
    frontend->release_cta(0);
    ok &= frontend->logical_cta_id(1) == 38;
  } else {
    frontend->release_cta(0);
    const auto previous_uid = kernel->get_uid();
    kernel->~kernel_info_t();
    if (scenario == Scenario::kReset) frontend->reset();
    void *next = scenario == Scenario::kDistinctAddress ? alternate : storage;
    kernel = new (next) kernel_info_t(dim3(64), dim3(lanes), "second", &ctx, 0, 98761);
    ok &= kernel->get_uid() != previous_uid;
    frontend->start_cta(0, 0, lanes, 0, lanes, *kernel);
    unsigned pc, rpc;
    frontend->control_flow_top(0, nullptr, &pc, &rpc);
    ok &= pc == 64;
    if (pc != 64) fprintf(stderr, "rebound launch PC=%u expected=64\n", pc);
  }
  std::remove(path);
  std::exit(ok ? 0 : 1);
}

TEST(SassFrontendLifecycleTest, ReusedKernelAddressLoadsNewImage) {
  ASSERT_EXIT(CheckFrontend(Scenario::kReusedAddress), testing::ExitedWithCode(0), "");
}
TEST(SassFrontendLifecycleTest, ResetBeforeKernelAddressReuse) {
  ASSERT_EXIT(CheckFrontend(Scenario::kReset), testing::ExitedWithCode(0), "");
}
TEST(SassFrontendLifecycleTest, DifferentKernelAddressLoadsNewImage) {
  ASSERT_EXIT(CheckFrontend(Scenario::kDistinctAddress), testing::ExitedWithCode(0), "");
}
TEST(SassFrontendLifecycleTest, SameLaunchKeepsResidentCtas) {
  ASSERT_EXIT(CheckFrontend(Scenario::kSameLaunch), testing::ExitedWithCode(0), "");
}
TEST(SassFrontendLifecycleTest, IdentityLivesUntilCtaRelease) {
  ASSERT_EXIT(CheckFrontend(Scenario::kIdentity), testing::ExitedWithCode(0), "");
}
TEST(SassFrontendLifecycleTest, AtomicSingleLaneAccounting) {
  ASSERT_EXIT(CheckFrontend(Scenario::kAtomic), testing::ExitedWithCode(0), "");
}
TEST(SassFrontendLifecycleTest, AtomicPartialWarpAccounting) {
  ASSERT_EXIT(CheckFrontend(Scenario::kAtomic, 5), testing::ExitedWithCode(0), "");
}
TEST(SassFrontendLifecycleTest, AtomicFullWarpAccounting) {
  ASSERT_EXIT(CheckFrontend(Scenario::kAtomic, 32), testing::ExitedWithCode(0), "");
}
TEST(SassFrontendLifecycleTest, PredicatedOffAtomicAccounting) {
  ASSERT_EXIT(CheckFrontend(Scenario::kPredicatedAtomic), testing::ExitedWithCode(0), "");
}
}  // namespace

TEST(SassFrontendLifecycleTest, NativeTmaDoesNotAddExecutionLatencyHold) {
  ASSERT_EXIT(CheckFrontend(Scenario::kNativeTma), testing::ExitedWithCode(0), "");
}
TEST(SassFrontendLifecycleTest, PtxTmaPreservesAsynchronousSubmission) {
  ASSERT_EXIT(CheckFrontend(Scenario::kPtxTma), testing::ExitedWithCode(0), "");
}
TEST(SassFrontendLifecycleTest, MaskedTmaDoesNotHoldWarp) {
  ASSERT_EXIT(CheckFrontend(Scenario::kMaskedTma), testing::ExitedWithCode(0), "");
}
TEST(SassFrontendLifecycleTest, SingleCycleTmaDoesNotHoldWarp) {
  ASSERT_EXIT(CheckFrontend(Scenario::kFastTma), testing::ExitedWithCode(0), "");
}

TEST(SassFrontendLifecycleTest, ModuleGlobalsRelocateAndInitializeOnlyOnce) {
  ASSERT_EXIT(CheckFrontend(Scenario::kModuleGlobals), testing::ExitedWithCode(0), "");
}

TEST(SassFrontendLifecycleTest, SharedHostGlobalRelocatesAcrossFatbins) {
  ASSERT_EXIT(CheckFrontend(Scenario::kSharedGlobal), testing::ExitedWithCode(0), "");
}

TEST(SassFrontendLifecycleTest, SharedHostGlobalRejectsSizeConflict) {
  ASSERT_DEATH(CheckFrontend(Scenario::kGlobalSizeConflict),
               "conflicting device-global host registration");
}

TEST(SassFrontendLifecycleTest, SharedHostGlobalRejectsModuleConflict) {
  ASSERT_DEATH(CheckFrontend(Scenario::kGlobalModuleConflict),
               "conflicting device-global module registration");
}

TEST(SassFrontendLifecycleTest, NativeWgmmaWaitUsesProjectedMetadata) {
  warp_inst_t inst;
  inst_t::wgmma_static_info_t info;
  info.operation = inst_t::wgmma_static_info_t::WGMMA_WAIT_GROUP;
  info.wait_group_num = 1;
  inst.set_wgmma_static_info(info);
  EXPECT_EQ(flash_gpgpu_sim::wgmma_wait_group_num_from_inst(&inst), 1u);
}

TEST(SassFrontendLifecycleTest, NativeWgmmaTrafficUsesProjectedMetadata) {
  shader_core_config config(nullptr);
  config.warp_size = 32;
  config.gpgpu_wgmma_rf_traffic_enable = true;
  config.gpgpu_wgmma_rf_traffic_assume_accumulate = false;
  config.gpgpu_wgmma_rf_traffic_include_rs_a = false;
  warp_inst_t inst;
  inst_t::wgmma_static_info_t info;
  info.operation = inst_t::wgmma_static_info_t::WGMMA_MMA_ASYNC;
  info.accumulator_bytes_per_thread = 128;
  info.register_a_registers_per_thread = 4;
  inst.set_wgmma_static_info(info);
  EXPECT_EQ(flash_gpgpu_sim::wgmma_collector_token_bytes_from_inst(&inst, &config),
            16384u);
  config.gpgpu_wgmma_rf_traffic_assume_accumulate = true;
  config.gpgpu_wgmma_rf_traffic_include_rs_a = true;
  EXPECT_EQ(flash_gpgpu_sim::wgmma_collector_token_bytes_from_inst(&inst, &config),
            34816u);
  config.gpgpu_wgmma_rf_traffic_enable = false;
  EXPECT_EQ(flash_gpgpu_sim::wgmma_collector_token_bytes_from_inst(&inst, &config),
            0u);
}

TEST(SassFrontendLifecycleTest, PtxNamedBarrierKeepsUpstreamReleaseLatency) {
  ASSERT_EXIT(CheckFrontend(Scenario::kPtxNamedBarrier), testing::ExitedWithCode(0), "");
}
TEST(SassFrontendLifecycleTest, NativeNamedBarrierKeepsCalibratedReleaseLatency) {
  ASSERT_EXIT(CheckFrontend(Scenario::kNativeNamedBarrier), testing::ExitedWithCode(0), "");
}
TEST(SassFrontendLifecycleTest, PtxWarpExitKeepsUpstreamBarrierReleaseLatency) {
  ASSERT_EXIT(CheckFrontend(Scenario::kPtxExitBarrier), testing::ExitedWithCode(0), "");
}
TEST(SassFrontendLifecycleTest, NativeWarpExitKeepsCalibratedBarrierReleaseLatency) {
  ASSERT_EXIT(CheckFrontend(Scenario::kNativeExitBarrier), testing::ExitedWithCode(0), "");
}

TEST(SassFrontendLifecycleTest, NativeGlobalsDoNotConsumeDynamicHeap) {
  ASSERT_EXIT(CheckFrontend(Scenario::kStaticGlobalAllocation), testing::ExitedWithCode(0), "");
}

namespace {
void CheckNativeBarrierEffects(bool per_lane, bool parity, bool imported) {
  gpgpu_context ctx;
  gpgpu_sim_config config(&ctx);
  auto *options = option_parser_create();
  ctx.ptx_reg_options(options);
  ctx.instruction_timing.reg_options(options);
  icnt_reg_options(options);
  config.reg_options(options);
  option_parser_cfgfile(options, "gpgpusim.config");
  const char *args[] = {"barrier-test",
                        "-gpgpu_n_clusters",
                        "1",
                        "-gpgpu_n_cores_per_cluster",
                        "1",
                        "-network_mode",
                        "2"};
  option_parser_cmdline(options, 7, args);
  config.init();
  TestGpu gpu(config, &ctx);
  gpu.init();
  auto *core = gpu.core();
  active_mask_t mask;
  mask.set(0);
  mask.set(1);
  core->get_shd_warp(0)->init(0, 0, 0, mask, 0, 0);
  barrier_set_t barriers(core, 64, 32, 16, 32);
  warp_set_t warps;
  warps.set(0);
  barriers.allocate_barrier(0, warps);
  warp_inst_t inst(gpu.getShaderCoreConfig());
  inst.op = MBARRIER_OP;
  inst.set_dependency_control({});
  inst.issue(mask, 0, 0, 0, 0, 0);
  inst_t::mbarrier_static_info_t op;
  inst_t::mbarrier_info_t info;
  info.bar_id = 0x100;
  info.bar_count = 1;
  info.bar_parity = parity;
  if (imported) {
    barriers.initialize_native_stored_barrier(0, 0, 0x100, 1, parity);
  } else {
    op.operation = inst_t::mbarrier_static_info_t::MBARRIER_INIT;
    inst.set_mbarrier_static_info(op);
    inst.set_mbarrier_info(0, info);
    barriers.warp_reaches_mbarrier(0, 0, inst, mask, nullptr, 0);
  }
  op.operation = inst_t::mbarrier_static_info_t::MBARRIER_ARRIVE_EXPECT_TX;
  op.arrive = true;
  op.expect_tx = true;
  inst.set_mbarrier_static_info(op);
  info.bar_count = per_lane ? 12 : 6;
  inst.set_mbarrier_info(0, info);
  barriers.warp_reaches_mbarrier(0, 0, inst, mask, nullptr, 0);
  // Try-wait must park on the requested initial phase until both lane
  // completions arrive. The legacy uniform form contributes only once.
  op.operation = inst_t::mbarrier_static_info_t::MBARRIER_TRY_WAIT;
  inst.set_mbarrier_static_info(op);
  barriers.warp_reaches_mbarrier(0, 0, inst, mask, nullptr, 0);
  if (!barriers.warp_waiting_at_barrier(0)) std::exit(3);
  op.operation = inst_t::mbarrier_static_info_t::MBARRIER_COMPLETE_TX;
  op.per_lane_complete_tx = per_lane;
  inst.set_mbarrier_static_info(op);
  info.bar_count = 6;
  inst.set_mbarrier_info(0, info);
  inst.set_mbarrier_info(1, info);
  barriers.warp_reaches_mbarrier(0, 0, inst, mask, nullptr, 0);
  for (unsigned i = 0; i < 256; ++i) {
    barriers.cycle();
    ++gpu.gpu_sim_cycle;
  }
  std::exit(barriers.warp_waiting_at_barrier(0) ? 4 : 0);
}
TEST(SassBarrierTimingTest, PerLaneCompletionReleasesWaiter) {
  ASSERT_EXIT(CheckNativeBarrierEffects(true, false, false),
              testing::ExitedWithCode(0), "");
}
TEST(SassBarrierTimingTest, LegacyUniformCompletionStillCountsOnce) {
  ASSERT_EXIT(CheckNativeBarrierEffects(false, false, false),
              testing::ExitedWithCode(0), "");
}
TEST(SassBarrierTimingTest, ExplicitInitialParityReachesTiming) {
  ASSERT_EXIT(CheckNativeBarrierEffects(true, true, false),
              testing::ExitedWithCode(0), "");
}
TEST(SassBarrierTimingTest, StoredInitialParityReachesTiming) {
  ASSERT_EXIT(CheckNativeBarrierEffects(true, true, true),
              testing::ExitedWithCode(0), "");
}
}  // namespace
