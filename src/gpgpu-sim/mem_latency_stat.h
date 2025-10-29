// Copyright (c) 2009-2011, Tor M. Aamodt
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef MEM_LATENCY_STAT_H
#define MEM_LATENCY_STAT_H

#include <stdio.h>
#include <zlib.h>
#include <map>
#include <array>
#include <vector>

#include "../parallel_sim_util.h"

class memory_config;
class memory_stats_t {
 public:
  memory_stats_t(unsigned n_shader,
                 const class shader_core_config *shader_config,
                 const memory_config *mem_config, const class gpgpu_sim *gpu);

  memory_stats_t(const memory_stats_t &other)
      : memory_stats_t(other.m_n_shader, other.m_shader_config,
                       other.m_memory_config, other.m_gpu) {}

  void clear_accumulator();
  void aggregate(const memory_stats_t &other);

  unsigned memlatstat_done(class mem_fetch *mf);
  void memlatstat_read_done(class mem_fetch *mf);
  void memlatstat_dram_access(class mem_fetch *mf);
  void memlatstat_icnt2mem_pop(class mem_fetch *mf);
  void memlatstat_lat_pw();
  void memlatstat_print(unsigned n_mem, unsigned gpu_mem_n_bk);

  void visualizer_print(gzFile visualizer_file);

  // Reset local L2 stats that are aggregated each sampling window
  void clear_L2_stats_pw();

  unsigned m_n_shader;

  const shader_core_config *m_shader_config;
  const memory_config *m_memory_config;
  const class gpgpu_sim *m_gpu;

  unsigned max_mrq_latency;
  unsigned max_dq_latency;
  unsigned max_mf_latency;
  unsigned max_icnt2mem_latency;
  unsigned max_icnt2sh_latency;
  unsigned long long int tot_icnt2mem_latency;
  unsigned long long int tot_icnt2sh_latency;
  unsigned long long int tot_mrq_latency;
  unsigned long long int tot_mrq_num;
  std::array<unsigned, 32> mrq_lat_table;
  std::array<unsigned, 32> dq_lat_table;
  std::array<unsigned, 32> mf_lat_table;
  std::array<unsigned, 24> icnt2mem_lat_table;
  std::array<unsigned, 24> icnt2sh_lat_table;
  std::array<unsigned, 32> mf_lat_pw_table;  // table storing values of mf latency Per
                                              // Window
  // total latency summed up per window. divide by
  // mf_num_lat_pw to obtain average latency Per Window
  unsigned mf_num_lat_pw;
  unsigned max_warps;
  unsigned mf_tot_lat_pw;

  unsigned long long int mf_total_lat;

  using table_t = std::vector<std::vector<unsigned int>>;

  unsigned long long int *
      *mf_total_lat_table;      // mf latency sums[dram chip id][bank id]
  // [dram chip id][bank id]
  table_t mf_max_lat_table;
  unsigned num_mfs;
  unsigned int ***bankwrites;  // bankwrites[shader id][dram chip id][bank id]
  unsigned int ***bankreads;   // bankreads[shader id][dram chip id][bank id]
  unsigned int **totalbankwrites;    // bankwrites[dram chip id][bank id]
  unsigned int **totalbankreads;     // bankreads[dram chip id][bank id]
  unsigned int **totalbankaccesses;  // bankaccesses[dram chip id][bank id]
  unsigned int
      *num_MCBs_accessed;  // tracks how many memory controllers are accessed
                           // whenever any thread in a warp misses in cache
  unsigned int *position_of_mrq_chosen;  // position of mrq in m_queue chosen

  unsigned ***mem_access_type_stats;  // dram access type classification

  // AerialVision L2 stats
  unsigned L2_read_miss;
  unsigned L2_write_miss;
  unsigned L2_read_hit;
  unsigned L2_write_hit;

  // L2 cache stats
  unsigned int *L2_cbtoL2length;
  unsigned int *L2_cbtoL2writelength;
  unsigned int *L2_L2tocblength;
  unsigned int *L2_dramtoL2length;
  unsigned int *L2_dramtoL2writelength;
  unsigned int *L2_L2todramlength;

  // DRAM access row locality stats
  // [dram chip id][bank id]
  table_t concurrent_row_access;
  table_t num_activates;
  table_t row_access;
  table_t max_conc_access2samerow;
  table_t max_servicetime2samerow;

  // Power stats
  unsigned total_n_access;
  unsigned total_n_reads;
  unsigned total_n_writes;
};

// Provide a thread-safe manager for memory_stats_t objects.
class memory_stats_manager_t {
public:
  using stats_ptr_t = std::shared_ptr<memory_stats_t>;
  memory_stats_manager_t(unsigned n_shader, const memory_config *mem_config,
                         const shader_core_config *shader_config,
                         const gpgpu_sim *gpu);

  stats_ptr_t get_stats();
  stats_ptr_t get_aggregate_stats() { return m_aggregate_stats; }

  void aggregate_stats();

private:
  stats_ptr_t m_aggregate_stats;
#ifdef FLASH_GPGPU_SIM_OMP
  std::array<stats_ptr_t, FLASH_GPGPU_SIM_OMP_MAX_THREADS> m_thread_local_stats;
#endif
};

#endif /*MEM_LATENCY_STAT_H*/
