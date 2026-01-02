// Copyright (c) 2009-2011, Tor M. Aamodt, Ali Bakhoda
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

#ifndef opcodes_h_included
#define opcodes_h_included

enum opcode_t {
#define OP_DEF(OP, FUNC, STR, DST, CLASSIFICATION) OP,
#define OP_W_DEF(OP, FUNC, STR, DST, CLASSIFICATION) OP,
#include "opcodes.def"
  NUM_OPCODES
#undef OP_DEF
#undef OP_W_DEF
};

enum special_regs {
  CLOCK_REG,
  HALFCLOCK_ID,
  CLOCK64_REG,
  CTAID_REG,
  ENVREG_REG,
  GRIDID_REG,
  LANEID_REG,
  LANEMASK_EQ_REG,
  LANEMASK_LE_REG,
  LANEMASK_LT_REG,
  LANEMASK_GE_REG,
  LANEMASK_GT_REG,
  NCTAID_REG,
  NTID_REG,
  NSMID_REG,
  NWARPID_REG,
  PM_REG,
  SMID_REG,
  TID_REG,
  WARPID_REG,
  WARPSZ_REG
};
enum wmma_type {
  LOAD_A,
  LOAD_B,
  LOAD_C,
  STORE_D,
  MMA,
  ROW,
  COL,
  M16N16K16,
  M32N8K16,
  M8N32K16

};

// MMA instruction shape identifiers (for lexer/parser token values)
// These are separate from flash_gpgpu_sim::mma_shape_type to avoid enum value conflicts
// Values start at 100 to avoid conflicts with wmma_type enum (which ends at ~10)
enum {
  MMA_M16N8K8 = 100,
  MMA_M16N8K4 = 101,
  MMA_M16N8K16 = 102,
  MMA_M16N8K32 = 103,
  MMA_M16N8K64 = 104,
  MMA_M8N8K4 = 105
};

// MMA layout mode identifiers
enum {
  MMA_ROW_COL = 200,
  MMA_ROW_ROW = 201,
  MMA_COL_ROW = 202,
  MMA_COL_COL = 203
};

#endif
