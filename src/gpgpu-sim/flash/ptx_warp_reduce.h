#pragma once
class ptx_instruction;
class core_t;
class warp_inst_t;
namespace flash_gpgpu_sim {
void execute_ptx_redux(const ptx_instruction *, core_t *, warp_inst_t &);
}
