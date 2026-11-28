#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <unistd.h>
#include "kernel_info.h"
#include "gpgpu-sim/flash/sass/runtime/runtime_adapter.h"
#include <limits>
#include <string>
#include <vector>
#include "gpgpu-sim/flash/cta_dispatch_budget.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "gpgpu-sim/l2cache.h"
#include "libcuda/gpgpu_context.h"
#include "option_parser.h"

namespace {
class TimingGpu : public exec_gpgpu_sim {
 public:
  using exec_gpgpu_sim::exec_gpgpu_sim;
  using gpgpu_sim::issue_block2core;
  unsigned launched() const { return m_total_cta_launched; }
  shader_core_ctx *core() { return m_cluster[0]->get_core(0); }
  memory_partition_unit *partition() { return m_memory_partition_unit[0]; }
};
template <typename Check>
void Run(const std::string &extra, Check check) {
  gpgpu_context ctx;
  gpgpu_sim_config config(&ctx);
  auto *options = option_parser_create();
  ctx.ptx_reg_options(options);
  ctx.instruction_timing.reg_options(options);
  flash_gpgpu_sim::sass::register_runtime_options(options);
  icnt_reg_options(options);
  config.reg_options(options);
  option_parser_cfgfile(options, "gpgpusim.config");
  const std::string args =
      "-gpgpu_n_clusters 1 -gpgpu_n_cores_per_cluster 2 -network_mode 2 "
      "-gpgpu_kernel_launch_latency 0 -gpgpu_TB_launch_latency 0 "
      "-gpgpu_memlatency_stat 0 -gpgpu_simple_dram_model 1 "
      "-gpgpu_l2_rop_latency 0 -gpgpu_l2_multi_issue_port_model 0 "
      "-gpgpu_l2_rop_delay_output_sectors_per_cycle 1 "
      "-gpgpu_l2_tma_request_coalescing 0 " + extra;
  option_parser_delimited_string(options, args.c_str(), " ");
  config.init();
  TimingGpu gpu(config, &ctx);
  gpu.init();
  check(gpu, ctx);
  std::exit(testing::Test::HasFailure() ? 1 : 0);
}
mem_fetch *Request(TimingGpu &gpu, gpgpu_context &ctx, unsigned sid = 0) {
  mem_access_byte_mask_t bytes;
  for (unsigned i = 0; i < 32; ++i) bytes.set(i);
  mem_access_sector_mask_t sector; sector.set(0);
  mem_access_t access(GLOBAL_ACC_R, 0, 32, false, active_mask_t().set(),
                      bytes, sector, &ctx);
  return new mem_fetch(access, nullptr, 0, 8, 0, sid, 0,
                       gpu.getMemoryConfig(), 0);
}
void CheckRelease(unsigned width, unsigned capacity, unsigned data_width = 0) {
  Run("-gpgpu_cache:dl2 S:1:128:4,L:B:m:L:L,A:8:16,32:0,32 "
      "-gpgpu_dram_partition_queues 32:32:32:" + std::to_string(capacity) +
      " -gpgpu_l2_mshr_release_requests_per_cycle " + std::to_string(width) +
      (data_width ? " -gpgpu_l2_multi_issue_port_model 1 -gpgpu_l2_shared_data_array 1 -gpgpu_l2_data_port_sectors_per_cycle " + std::to_string(data_width) : ""),
      [&](TimingGpu &gpu, gpgpu_context &ctx) {
    auto *sub = gpu.partition()->get_sub_partition(0);
    std::vector<mem_fetch *> requests;
    for (unsigned i = 0; i < 8; ++i) {
      auto *mf = Request(gpu, ctx);
      requests.push_back(mf);
      sub->push(mf, 0);
    }
    for (unsigned t = 0; t < 40; ++t) {
      gpu.gpu_sim_cycle = t;
      sub->cache_cycle(t);
    }
    ASSERT_FALSE(sub->L2_dram_queue_empty());
    auto *fill = sub->L2_dram_queue_top();
    sub->L2_dram_queue_pop();
    ASSERT_TRUE(sub->L2_dram_queue_empty()); // Eight waiters, one DRAM miss.
    fill->set_reply();
    sub->dram_L2_queue_push(fill);
    unsigned count = 0, maximum = 0;
    for (unsigned t = 40; t < 100; ++t) {
      gpu.gpu_sim_cycle = t;
      sub->cache_cycle(t);
      unsigned released = 0;
      while (sub->top()) {
        auto *mf = sub->pop();
        ASSERT_LT(count, requests.size());
        EXPECT_EQ(mf, requests[count++]);
        ++released;
        delete mf;
      }
      maximum = std::max(maximum, released);
      EXPECT_LE(released, std::min(width, capacity));
    }
    EXPECT_EQ(count, 8u);
    EXPECT_EQ(maximum, std::min(std::min(width, capacity), data_width ? data_width : width));
  });
}
TEST(B200SystemTimingTest, MshrReleaseOne) {
  ASSERT_EXIT(CheckRelease(1, 16), testing::ExitedWithCode(0), "");
}
TEST(B200SystemTimingTest, MshrReleaseTwo) {
  ASSERT_EXIT(CheckRelease(2, 16), testing::ExitedWithCode(0), "");
}
TEST(B200SystemTimingTest, MshrReleaseHonorsBackpressure) {
  ASSERT_EXIT(CheckRelease(4, 1), testing::ExitedWithCode(0), "");
}
void CheckLocality(bool enabled) {
  Run("-gpgpu_cache:dl2 none -dram_latency 50 "
      "-gpgpu_simple_dram_local_read_latency 10 "
      "-gpgpu_simple_dram_remote_read_latency " + std::to_string(enabled ? 30 : 0),
      [&](TimingGpu &gpu, gpgpu_context &ctx) {
    auto *partition = gpu.partition();
    auto *sub = partition->get_sub_partition(0);
    auto *remote = Request(gpu, ctx, 1);
    auto *local = Request(gpu, ctx, 0);
    sub->push(remote, 0);
    sub->push(local, 0);
    std::vector<mem_fetch *> returned;
    std::vector<unsigned> ticks;
    for (unsigned t = 0; t < 100; ++t) {
      gpu.gpu_sim_cycle = t;
      sub->cache_cycle(t);
      partition->simple_dram_model_cycle();
      while (sub->top()) {
        returned.push_back(sub->pop());
        ticks.push_back(t);
      }
    }
    ASSERT_EQ(returned.size(), 2u);
    EXPECT_EQ(returned[0], enabled ? local : remote);
    EXPECT_EQ(returned[1], enabled ? remote : local);
    if (enabled) {
      EXPECT_LT(ticks[0], 25u);
      EXPECT_GE(ticks[1], 40u);
    } else {
      EXPECT_GE(ticks[0], 50u);
    }
    delete remote;
    delete local;
  });
}
TEST(B200SystemTimingTest, RemoteReadDoesNotBlockLaterLocalRead) {
  ASSERT_EXIT(CheckLocality(true), testing::ExitedWithCode(0), "");
}
TEST(B200SystemTimingTest, DisabledLocalityKeepsFifoAndUniformLatency) {
  ASSERT_EXIT(CheckLocality(false), testing::ExitedWithCode(0), "");
}
TEST(B200SystemTimingTest, DispatchBudgetSharedAcrossLaunchKinds) {
  using flash_gpgpu_sim::cta_dispatch_budget;
  cta_dispatch_budget budget(3, 2, 2);
  EXPECT_EQ(budget.available(0), 2u);
  budget.consume(0, 1); // An open thread-block cluster.
  EXPECT_EQ(budget.available(0), 1u); // Ordinary launch shares its GPC budget.
  budget.consume(0, 1);
  EXPECT_EQ(budget.available(0), 0u);
  EXPECT_EQ(budget.available(1), 1u); // Also shares device budget.
  budget.consume(1, 1);
  EXPECT_EQ(budget.available(1), 0u);
  cta_dispatch_budget next_tick(3, 2, 2);
  EXPECT_EQ(next_tick.available(0), 2u);
  cta_dispatch_budget unlimited(0, 0, 2);
  unlimited.consume(0, 8);
  EXPECT_EQ(unlimited.available(0), std::numeric_limits<unsigned>::max());
}

void CheckDispatch(bool clustered, unsigned interval = 0) {
  Run("-gpgpu_n_clusters 2 -gpgpu_cta_load_balance 0 "
      "-gpgpu_cta_dispatch_width 3 -gpgpu_cta_dispatch_per_gpc " +
      std::to_string(clustered ? 1 : 2) +
      (interval ? " -gpgpu_n_clusters 1 -gpgpu_n_cores_per_cluster 1 -gpgpu_cta_dispatch_interval " + std::to_string(interval) : ""),
      [&](TimingGpu &gpu, gpgpu_context &ctx) {
    char path[] = "/tmp/flashgpu-dispatch-XXXXXX";
    int fd = mkstemp(path);
    ASSERT_GE(fd, 0); close(fd);
    std::ofstream file(path);
    file << "SASSIR\t3\nDECODER\tnvdisasm\tversion\t1.0.0\n"
            "KERNEL\tdispatch_probe\tsm120\t16\nRESOURCES\t16\t0\t0\t0\n"
            "PARAMBANK\t0x380\t0x0\n"
            "INST\t0\t0\t0\t1\tnvdisasm\t\tEXIT\t\t0\t1\t0\t0\t0\t0\n"
            "ENDINST\nENDKERNEL\n";
    file.close();
    setenv("FLASHGPU_SASS_IR", path, 1);
    setenv("FLASHGPU_SASS_TIMING", "1", 1);
    kernel_info_t kernel(dim3(8), dim3(32), "dispatch_probe", &ctx, 0, 98799);
    kernel.set_resource_usage({16, 0, 0, 0});
    if (clustered) kernel.set_cluster_launch(dim3(2, 1, 1));
    gpu.launch(&kernel);
    gpu.issue_block2core();
    EXPECT_EQ(gpu.launched(), clustered || interval ? 1u : 3u);
    ++gpu.gpu_sim_cycle;
    gpu.issue_block2core();
    EXPECT_EQ(gpu.launched(), interval ? 1u : clustered ? 2u : 6u);
    if (interval) {
      gpu.gpu_sim_cycle = interval;
      gpu.issue_block2core();
      EXPECT_EQ(gpu.launched(), 2u);
    }
    std::remove(path);
  });
}
TEST(B200SystemTimingTest, OrdinaryLaunchRespectsGlobalAndGpcWidth) {
  ASSERT_EXIT(CheckDispatch(false), testing::ExitedWithCode(0), "");
}
TEST(B200SystemTimingTest, PartialClusterContinuesOnNextTick) {
  ASSERT_EXIT(CheckDispatch(true), testing::ExitedWithCode(0), "");
}
} // namespace

TEST(B200SystemTimingTest, MshrReleaseHonorsDataArrayBandwidth) {
  ASSERT_EXIT(CheckRelease(4, 16, 1), testing::ExitedWithCode(0), "");
}

TEST(B200SystemTimingTest, PerSmAdmissionInterval) {
  ASSERT_EXIT(CheckDispatch(false, 5), testing::ExitedWithCode(0), "");
}
