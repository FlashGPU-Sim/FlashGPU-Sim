#pragma once
#include "../../../cuda-sim/ptx_ir.h"
#include "../../../cuda-sim/ptx_sim.h"
#include "../tb_cluster.h"
#include "../tcgen05.h"
namespace flash_gpgpu_sim {
std::vector<tb_cluster_target_t> tcgen05_ptx_targets(const ptx_instruction *,
                                                     ptx_thread_info *);
void tcgen05_assert_cta_group1(const ptx_instruction *);
bool tcgen05_is_warp_leader(const ptx_thread_info *);
tcgen05_tmem_scope_t tcgen05_tmem_scope(const ptx_instruction *,
                                        const ptx_thread_info *);
ptx_reg_t tcgen05_read_operand(const ptx_instruction *, ptx_thread_info *,
                               unsigned);
ptx_reg_t tcgen05_read_u64_operand(const ptx_instruction *, ptx_thread_info *,
                                   unsigned);
std::vector<uint32_t> tcgen05_read_vector_words(const operand_info &,
                                                ptx_thread_info *);
mem_addr_t tcgen05_eval_address(const operand_info &, ptx_thread_info *);
} // namespace flash_gpgpu_sim
