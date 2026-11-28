#pragma once
class ptx_instruction;
class ptx_thread_info;
namespace flash_gpgpu_sim {
void execute_ptx_global_reduce(const ptx_instruction *, ptx_thread_info *);
}
