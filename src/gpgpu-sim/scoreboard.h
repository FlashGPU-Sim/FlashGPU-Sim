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

#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <vector>
#include "assert.h"

#ifndef SCOREBOARD_H_
#define SCOREBOARD_H_

#include "../abstract_hardware_model.h"

// Producer type for scoreboard collision classification (NCU-style stall tracking)
enum reg_producer_t {
  PROD_MEM_GLOBAL,   // global/local/param/tex load → NCU "Long Scoreboard"
  PROD_MEM_SHARED,   // shared memory load → NCU "Short Scoreboard"
  PROD_TENSOR_CORE,  // TENSOR_CORE_OP result
  PROD_SP_INT,       // SP/INT/ALU result
  PROD_SFU,          // SFU/DP result
  PROD_TMA,          // TMA operation result
  PROD_TENSOR_MAP,   // tensormap descriptor/proxy operation result
  PROD_OTHER,
};

class Scoreboard {
 public:
  Scoreboard(unsigned sid, unsigned n_warps, class gpgpu_t *gpu,
             bool alu_result_forwarding = false);

  void reserveRegisters(const warp_inst_t *inst);
  void releaseRegisters(const warp_inst_t *inst);
  void reserveRegistersForWarp(const warp_inst_t *inst, unsigned warp_id);
  void releaseRegistersForWarp(const warp_inst_t *inst, unsigned warp_id);
  void releaseRegister(unsigned wid, unsigned regnum, unsigned owner_uid);

  bool checkCollision(unsigned wid, const inst_t *inst) const;
  reg_producer_t getCollisionType(unsigned wid, const inst_t *inst) const;
  bool pendingWrites(unsigned wid) const;
  void printContents() const;
  const bool islongop(unsigned warp_id, unsigned regnum);

 private:
  void reserveRegister(unsigned wid, unsigned regnum, unsigned owner_uid,
                       bool forwardable,
                       unsigned long long forward_ready_cycle);
  void clearRegister(unsigned wid, unsigned regnum);
  bool registerCollision(unsigned wid, unsigned regnum) const;
  unsigned long long currentCycle() const;
  int get_sid() const { return m_sid; }

  unsigned m_sid;

  // keeps track of pending writes to registers
  // indexed by warp id, reg_id => pending write count
  std::vector<std::set<unsigned> > reg_table;
  // Register that depend on a long operation (global, local or tex memory)
  std::vector<std::set<unsigned> > longopregs;
  // Producer type for each pending register (NCU-style stall classification)
  std::vector<std::map<unsigned, reg_producer_t> > reg_producer;
  // Dynamic producer ownership prevents a late writeback from clearing a
  // younger producer that reused a forwarded destination register.
  std::vector<std::map<unsigned, unsigned> > reg_owner;
  std::vector<std::map<unsigned, unsigned long long> > reg_forward_ready;

  class gpgpu_t *m_gpu;
  bool m_alu_result_forwarding;
};

#endif /* SCOREBOARD_H_ */
