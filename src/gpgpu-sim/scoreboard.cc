// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh
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

#include "scoreboard.h"
#include "../cuda-sim/ptx_sim.h"
#include "shader.h"
#include "shader_trace.h"

// Constructor
Scoreboard::Scoreboard(unsigned sid, unsigned n_warps, class gpgpu_t* gpu)
    : longopregs() {
  m_sid = sid;
  // Initialize size of table
  reg_table.resize(n_warps);
  longopregs.resize(n_warps);
  reg_producer.resize(n_warps);

  m_gpu = gpu;
}

// Print scoreboard contents
void Scoreboard::printContents() const {
  printf("scoreboard contents (sid=%d): \n", m_sid);
  for (unsigned i = 0; i < reg_table.size(); i++) {
    if (reg_table[i].size() == 0) continue;
    printf("  wid = %2d: ", i);
    std::set<unsigned>::const_iterator it;
    for (it = reg_table[i].begin(); it != reg_table[i].end(); it++)
      printf("%u ", *it);
    printf("\n");
  }
}

void Scoreboard::reserveRegister(unsigned wid, unsigned regnum) {
  if (!(reg_table[wid].find(regnum) == reg_table[wid].end())) {
    printf(
        "Error: trying to reserve an already reserved register (sid=%d, "
        "wid=%d, regnum=%d).",
        m_sid, wid, regnum);
    abort();
  }
  SHADER_GPPRINTF(SCOREBOARD, "Reserved Register - warp:%d, reg: %d\n", wid,
                 regnum);
  reg_table[wid].insert(regnum);
}

// Unmark register as write-pending
void Scoreboard::releaseRegister(unsigned wid, unsigned regnum) {
  if (!(reg_table[wid].find(regnum) != reg_table[wid].end())) return;
  SHADER_GPPRINTF(SCOREBOARD, "Release register - warp:%d, reg: %d\n", wid,
                 regnum);
  reg_table[wid].erase(regnum);
}

const bool Scoreboard::islongop(unsigned warp_id, unsigned regnum) {
  return longopregs[warp_id].find(regnum) != longopregs[warp_id].end();
}

void Scoreboard::reserveRegisters(const class warp_inst_t* inst) {
  reserveRegistersForWarp(inst, inst->warp_id());
}

void Scoreboard::reserveRegistersForWarp(const class warp_inst_t* inst,
                                         unsigned warp_id) {
  // Classify producer type for NCU-style stall tracking
  reg_producer_t prod = PROD_OTHER;
  unsigned op = inst->op;
  if (op == TENSOR_CORE_OP) {
    prod = PROD_TENSOR_CORE;
  } else if (op == TENSOR_MEMORY_ACCELERATOR_OP) {
    prod = PROD_TMA;
  } else if (op == TENSOR_MAP_OP) {
    prod = PROD_TENSOR_MAP;
  } else if (op == SFU_OP || op == ALU_SFU_OP || op == DP_OP) {
    prod = PROD_SFU;
  } else if (op == LOAD_OP || op == STORE_OP || op == MEMORY_BARRIER_OP ||
             op == TENSOR_CORE_LOAD_OP || op == TENSOR_CORE_STORE_OP ||
             op == MBARRIER_OP) {
    if (inst->space.get_type() == shared_space) {
      prod = PROD_MEM_SHARED;
    } else {
      prod = PROD_MEM_GLOBAL;
    }
  } else if (op == SP_OP || op == INTP_OP || op == ALU_OP) {
    prod = PROD_SP_INT;
  }

  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      reserveRegister(warp_id, inst->out[r]);
      reg_producer[warp_id][inst->out[r]] = prod;
      SHADER_GPPRINTF(SCOREBOARD, "Reserved register - warp:%d, reg: %d\n",
                     warp_id, inst->out[r]);
    }
  }

  // Keep track of long operations
  if (inst->is_load() && (inst->space.get_type() == global_space ||
                          inst->space.get_type() == local_space ||
                          inst->space.get_type() == param_space_kernel ||
                          inst->space.get_type() == param_space_local ||
                          inst->space.get_type() == param_space_unclassified ||
                          inst->space.get_type() == tex_space)) {
    for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
      if (inst->out[r] > 0) {
        SHADER_GPPRINTF(SCOREBOARD, "New longopreg marked - warp:%d, reg: %d\n",
                       warp_id, inst->out[r]);
        longopregs[warp_id].insert(inst->out[r]);
      }
    }
  }
}

void Scoreboard::reclassifyShared(const class warp_inst_t* inst) {
  if (!inst) return;
  unsigned warp_id = inst->warp_id();
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0 &&
        reg_table[warp_id].find(inst->out[r]) != reg_table[warp_id].end()) {
      reg_producer[warp_id][inst->out[r]] = PROD_MEM_SHARED;
      longopregs[warp_id].erase(inst->out[r]);
    }
  }
}

// Release registers for an instruction
void Scoreboard::releaseRegisters(const class warp_inst_t* inst) {
  releaseRegistersForWarp(inst, inst->warp_id());
}

void Scoreboard::releaseRegistersForWarp(const class warp_inst_t* inst,
                                         unsigned warp_id) {
  for (unsigned r = 0; r < MAX_OUTPUT_VALUES; r++) {
    if (inst->out[r] > 0) {
      SHADER_GPPRINTF(SCOREBOARD, "Register Released - warp:%d, reg: %d\n",
                     warp_id, inst->out[r]);
      releaseRegister(warp_id, inst->out[r]);
      longopregs[warp_id].erase(inst->out[r]);
      reg_producer[warp_id].erase(inst->out[r]);
    }
  }
}

// Return the dominant (longest-latency) producer type among all colliding
// registers.  Priority: GLOBAL > TMA > TENSOR_CORE > SFU > SHARED > SP_INT > OTHER
static int producer_severity(reg_producer_t p) {
  switch (p) {
    case PROD_MEM_GLOBAL:  return 6;
    case PROD_TMA:         return 5;
    case PROD_TENSOR_CORE: return 4;
    case PROD_SFU:         return 3;
    case PROD_MEM_SHARED:  return 2;
    case PROD_SP_INT:      return 1;
    default:               return 0;
  }
}

reg_producer_t Scoreboard::getCollisionType(unsigned wid,
                                            const class inst_t* inst) const {
  std::set<int> inst_regs;
  for (unsigned i = 0; i < inst->outcount; i++) inst_regs.insert(inst->out[i]);
  for (unsigned i = 0; i < inst->incount; i++) inst_regs.insert(inst->in[i]);
  if (inst->pred > 0) inst_regs.insert(inst->pred);
  if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  if (inst->ar2 > 0) inst_regs.insert(inst->ar2);

  reg_producer_t worst = PROD_OTHER;
  for (auto it = inst_regs.begin(); it != inst_regs.end(); ++it) {
    if (reg_table[wid].find(*it) != reg_table[wid].end()) {
      auto prod_it = reg_producer[wid].find(*it);
      reg_producer_t p = (prod_it != reg_producer[wid].end())
                             ? prod_it->second
                             : PROD_OTHER;
      if (producer_severity(p) > producer_severity(worst)) worst = p;
    }
  }
  return worst;
}

/**
 * Checks to see if registers used by an instruction are reserved in the
 *scoreboard
 *
 * @return
 * true if WAW or RAW hazard (no WAR since in-order issue)
 **/
bool Scoreboard::checkCollision(unsigned wid, const class inst_t* inst) const {
  // Get list of all input and output registers
  std::set<int> inst_regs;

  for (unsigned iii = 0; iii < inst->outcount; iii++)
    inst_regs.insert(inst->out[iii]);

  for (unsigned jjj = 0; jjj < inst->incount; jjj++)
    inst_regs.insert(inst->in[jjj]);

  if (inst->pred > 0) inst_regs.insert(inst->pred);
  if (inst->ar1 > 0) inst_regs.insert(inst->ar1);
  if (inst->ar2 > 0) inst_regs.insert(inst->ar2);

  // Check for collision, get the intersection of reserved registers and
  // instruction registers
  std::set<int>::const_iterator it2;
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
    if (reg_table[wid].find(*it2) != reg_table[wid].end()) {
      return true;
    }
  return false;
}

bool Scoreboard::pendingWrites(unsigned wid) const {
  return !reg_table[wid].empty();
}
