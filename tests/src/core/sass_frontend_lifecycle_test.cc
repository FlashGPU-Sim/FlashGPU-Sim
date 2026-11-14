#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <new>
#include <string>
#include <unistd.h>

#include "libcuda/gpgpu_context.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "gpgpu-sim/flash/frontend/execution_frontend.h"
#include "gpgpu-sim/flash/sass/runtime/runtime_adapter.h"
#include "kernel_info.h"
#include "gpgpu-sim/flash/tma.h"
#include "option_parser.h"

namespace {
class TestGpu : public exec_gpgpu_sim {
 public:
  using exec_gpgpu_sim::exec_gpgpu_sim;
  exec_shader_core_ctx *core() {
    return static_cast<exec_shader_core_ctx *>(m_cluster[0]->get_core(0));
  }
};

enum class Scenario { kReusedAddress, kReset, kDistinctAddress, kSameLaunch,
                      kModuleGlobals, kIdentity, kAtomic, kPredicatedAtomic, kNativeTma, kPtxTma, kMaskedTma, kFastTma };

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
  config.init();
  TestGpu gpu(config, &ctx);
  gpu.init();
  auto *core = gpu.core();
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
    // the issue-side warp hold is independent of data completion.
    tma.warp_reaches_tma(0, 0, &inst);
    const bool expected = scenario == Scenario::kPtxTma;
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
    active_mask_t mask;
    for (unsigned lane = 0; lane < lanes; ++lane) mask.set(lane);
    for (unsigned pc = 0; pc <= 48; pc += 16) {
      warp_inst_t dynamic = *frontend->fetch(0, pc);
      dynamic.issue(mask, 0, pc, 0, 0, 0);
      frontend->execute(dynamic);
    }
    uint32_t value = 0;
    gpu.get_global_memory()->read(address, 4, &value);
    const unsigned expected = scenario == Scenario::kAtomic ? lanes : 0;
    const unsigned actual = core->get_shd_warp(0)->get_n_atomic();
    ok = actual == expected && value == (expected ? 42u : initial);
    if (!ok) fprintf(stderr, "atomic count=%u expected=%u value=%u\n",
                     actual, expected, value);
    if (ok) core->decrement_atomic_count(0, expected);
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

TEST(SassFrontendLifecycleTest, NativeTmaAvoidsPtxProxyHold) {
  ASSERT_EXIT(CheckFrontend(Scenario::kNativeTma), testing::ExitedWithCode(0), "");
}
TEST(SassFrontendLifecycleTest, PtxTmaKeepsProxyHold) {
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
