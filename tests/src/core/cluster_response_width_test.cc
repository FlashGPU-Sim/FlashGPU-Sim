#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "libcuda/gpgpu_context.h"
#include "gpgpu-sim/gpu-sim.h"
#include "gpgpu-sim/icnt_wrapper.h"
#include "option_parser.h"

namespace {
class ResponseTestGpu : public exec_gpgpu_sim {
 public:
  using exec_gpgpu_sim::exec_gpgpu_sim;
  simt_core_cluster *cluster() { return m_cluster[0]; }
};

struct ResponseCase {
  const char *name;
  bool ingress;
  unsigned width, capacity, ticks;
  std::vector<mem_access_type> types;
  std::vector<unsigned> sectors;
  unsigned accepted;
};

// Exercise the actual cluster loop, real per-SM queues and real LSU/TMA
// consumers. Only the interconnect producer is replaced with a deterministic
// queue. Each subprocess creates two SMs, so per-SM rather than per-GPC budget
// accounting is also checked. No simulated core cycles consume the replies.
void CheckResponse(const ResponseCase &test) {
  gpgpu_context ctx;
  gpgpu_sim_config config(&ctx);
  auto *options = option_parser_create();
  ctx.ptx_reg_options(options);
  ctx.func_sim->ptx_opcocde_latency_options(options);
  icnt_reg_options(options);
  config.reg_options(options);
  option_parser_cfgfile(options, "gpgpusim.config");
  const std::string overrides =
      "-gpgpu_n_clusters 1 -gpgpu_n_cores_per_cluster 2 -network_mode 2 "
      "-gpgpu_memlatency_stat 0 -gpgpu_tma_response_width 1 "
      "-gpgpu_cp_async_response_width 1 "
      "-gpgpu_cluster_response_ingress_sectors_per_cycle " +
      std::to_string(test.ingress ? test.width : 0) +
      " -gpgpu_cluster_response_dispatch_sectors_per_cycle " +
      std::to_string(test.ingress ? 0 : test.width) +
      " -gpgpu_n_cluster_ejection_buffer_size " +
      std::to_string(test.ingress ? test.capacity : 32) +
      " -gpgpu_n_ldst_response_buffer_size " +
      std::to_string(test.ingress ? 32 : test.capacity);
  option_parser_delimited_string(options, overrides.c_str(), " ");
  config.init();
  ResponseTestGpu gpu(config, &ctx);
  gpu.init();
  const auto *shader = gpu.getShaderCoreConfig();
  const auto *memory = gpu.getMemoryConfig();
  std::vector<std::deque<mem_fetch *>> incoming(2);
  std::vector<std::vector<mem_fetch *>> packets(2);
  for (unsigned sid = 0; sid < 2; ++sid) {
    for (unsigned i = 0; i < test.types.size(); ++i) {
      const auto type = test.types[i];
      const bool write = type == GLOBAL_ACC_W;
      const unsigned bytes = test.sectors[i] * SECTOR_SIZE;
      mem_access_t access(type, 0x1000 + i * 128, bytes, write, &ctx);
      auto *mf = new mem_fetch(access, nullptr, 0, 8, 0, sid, 0, memory, 0);
      mf->set_reply();
      mf->set_return_timestamp(0);
      packets[sid].push_back(mf);
      if (test.ingress) incoming[sid].push_back(mf);
      else {
        mf->set_status(IN_CLUSTER_TO_SHADER_QUEUE, 0);
        gpu.cluster()->push_response_fifo(mf);
      }
    }
  }
  icnt_top = [&](unsigned node) -> void * {
    for (unsigned sid = 0; sid < 2; ++sid)
      if (node == shader->topology().global_sm_node_id(sid))
        return incoming[sid].empty() ? nullptr : incoming[sid].front();
    return nullptr;
  };
  icnt_pop = [&](unsigned node) -> void * {
    auto *mf = static_cast<mem_fetch *>(icnt_top(node));
    if (mf) incoming[mf->get_sid()].pop_front();
    return mf;
  };
  for (unsigned tick = 0; tick < test.ticks; ++tick) {
    gpu.gpu_sim_cycle = tick + 1;
    gpu.cluster()->icnt_cycle();
  }
  for (unsigned sid = 0; sid < 2; ++sid) {
    unsigned accepted = 0;
    for (unsigned i = 0; i < packets[sid].size(); ++i) {
      const auto state = packets[sid][i]->get_status();
      const bool moved = test.ingress
          ? std::find(incoming[sid].begin(), incoming[sid].end(), packets[sid][i]) == incoming[sid].end()
          : state == IN_SHADER_LDST_RESPONSE_FIFO || state == IN_TMA_RESPONSE_FIFO;
      EXPECT_EQ(moved, i < test.accepted) << test.name << " SM=" << sid << " packet=" << i;
      accepted += moved;
    }
    EXPECT_EQ(accepted, test.accepted) << test.name << " SM=" << sid;
    if (test.ingress)
      EXPECT_EQ(incoming[sid].size(), packets[sid].size() - test.accepted);
  }
  // Subprocess ownership ends here; queued replies intentionally remain in
  // their real consumer queues. No kernel or global simulator thread started.
  std::exit(testing::Test::HasFailure() ? 1 : 0);
}

void PrintTo(const ResponseCase &test, std::ostream *out) {
  *out << test.name << " width=" << test.width << " capacity=" << test.capacity
       << " ticks=" << test.ticks << " expected=" << test.accepted;
}

class ClusterResponseWidthTest : public testing::TestWithParam<ResponseCase> {};
TEST_P(ClusterResponseWidthTest, HonorsBudgetAndBackpressure) {
  ASSERT_EXIT(CheckResponse(GetParam()), testing::ExitedWithCode(0), "");
}

std::vector<ResponseCase> Cases() {
  std::vector<ResponseCase> cases;
  for (bool ingress : {false, true}) {
    const char *prefix = ingress ? "Ingress" : "Dispatch";
    // Names are generated by the parameter suffix below.
    for (unsigned width : {0u, 1u, 2u, 4u})
      cases.push_back({prefix, ingress, width, 16, 1,
                       std::vector<mem_access_type>(6, GLOBAL_ACC_R),
                       std::vector<unsigned>(6, 1), width ? width : 1});
    cases.push_back({prefix, ingress, 4, 1, 1,
                     {GLOBAL_ACC_R, GLOBAL_ACC_R}, {1, 1}, 1});
    cases.push_back({prefix, ingress, 4, 0, 1,
                     {GLOBAL_ACC_R, GLOBAL_ACC_R}, {1, 1}, 0});
    cases.push_back({prefix, ingress, 4, 16, 1,
                     {GLOBAL_ACC_R, TMA_ACC_R, CP_ASYNC_ACC_R, GLOBAL_ACC_R, TMA_ACC_R},
                     {1, 1, 1, 1, 1}, 4});
    cases.push_back({prefix, ingress, 1, 16, 4, {GLOBAL_ACC_R}, {4}, 1});
    cases.push_back({prefix, ingress, 1, 16, 3, {GLOBAL_ACC_R}, {4}, 0});
    cases.push_back({prefix, ingress, 4, 16, 1,
                     std::vector<mem_access_type>(6, GLOBAL_ACC_W),
                     std::vector<unsigned>(6, 1), 4});
  }
  // Instruction replies are transport packets too; this only tests ingress,
  // not the independent one-fill-per-tick instruction-cache consumer.
  cases.push_back({"InstructionIngress", true, 4, 16, 1,
                   {INST_ACC_R, GLOBAL_ACC_R, INST_ACC_R, GLOBAL_ACC_R},
                   {1, 1, 1, 1}, 4});
  // Legacy async packet widths remain separate consumer limits. Explicit
  // ingress budgets govern transport; dispatch still obeys the TMA port cap.
  cases.push_back({"AsyncLegacy", true, 0, 16, 1,
                   {TMA_ACC_R, TMA_ACC_R}, {1, 1}, 1});
  cases.push_back({"AsyncIngress", true, 4, 16, 1,
                   {TMA_ACC_R, TMA_ACC_R}, {1, 1}, 2});
  cases.push_back({"AsyncDispatch", false, 4, 16, 1,
                   {TMA_ACC_R, TMA_ACC_R, GLOBAL_ACC_R}, {1, 1, 1}, 1});
  return cases;
}
INSTANTIATE_TEST_SUITE_P(PerSm, ClusterResponseWidthTest,
                         testing::ValuesIn(Cases()));
}  // namespace
