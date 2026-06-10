#ifndef FLASH_GPGPU_SIM_ELECT_H
#define FLASH_GPGPU_SIM_ELECT_H

// Forward declarations
class ptx_instruction;
class ptx_thread_info;
class core_t;
class warp_inst_t;

/**
 * Handle elect.sync instruction.
 *
 * elect.sync d|p, membermask
 *
 * Elects a leader thread from a set of threads specified by membermask.
 * The laneid of the elected thread is returned in d.
 * The predicate p is set to True for the leader thread, False for all others.
 *
 * Election is deterministic: the lowest numbered lane in membermask is elected.
 */
void handle_elect_inst(const ptx_instruction *pI, core_t *core,
                       const warp_inst_t &inst);

#endif // FLASH_GPGPU_SIM_ELECT_H
