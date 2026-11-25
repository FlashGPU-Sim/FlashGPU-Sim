#pragma once
class ptx_instruction;
class ptx_thread_info;
namespace flash_gpgpu_sim {
bool execute_ptx_bfloat_minmax(const ptx_instruction *, ptx_thread_info *);
}
