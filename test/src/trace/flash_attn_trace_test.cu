// GPT-2 Flash Attention Kernel Trace Test (Data-Driven)
//
// Kernel: flash_attn_kernel
// Grid: (1, 12, 1), Block: (128, 1, 1), Shared mem: 8736 bytes
// Args: arg0 [12,10,64] fp16 (Q), arg1 [12,10,64] fp16 (K),
//       arg2 [12,10,64] fp16 (V), arg3 [12,10,64] fp16 (output),
//       arg4=12 (num_heads), arg5=10 (seq_len), arg6=64 (head_dim),
//       arg7=0.125 (scale)
// Output: arg3
// Validates against captured GPU output from real hardware.

#include "trace_test_utils.h"

int main() {
    printf("=== GPT-2 Flash Attention Trace Test ===\n");

    CUcontext ctx;
    if (init_cuda(&ctx)) { fprintf(stderr, "CUDA init failed\n"); return 1; }

    char exe_dir[1024];
    get_exe_dir(exe_dir, sizeof(exe_dir));
    char path[2048];

    // Load fatbin
    build_path(path, sizeof(path), exe_dir, "flash_attn_kernel.fatbin");
    CUmodule module = load_fatbin(path);
    if (!module) return 1;
    CUfunction func = get_kernel(module, "flash_attn_kernel");
    if (!func) return 1;
    set_shared_mem(func, 8736);

    // Load input data (all 15360 bytes = 12 * 10 * 64 * 2)
    build_path(path, sizeof(path), exe_dir, "data/flash_attn_kernel_launch4_arg0.bin");
    void* d_arg0 = load_bin_to_gpu(path, 15360);
    if (!d_arg0) return 1;

    build_path(path, sizeof(path), exe_dir, "data/flash_attn_kernel_launch4_arg1.bin");
    void* d_arg1 = load_bin_to_gpu(path, 15360);
    if (!d_arg1) return 1;

    build_path(path, sizeof(path), exe_dir, "data/flash_attn_kernel_launch4_arg2.bin");
    void* d_arg2 = load_bin_to_gpu(path, 15360);
    if (!d_arg2) return 1;

    build_path(path, sizeof(path), exe_dir, "data/flash_attn_kernel_launch4_arg3.bin");
    void* d_arg3 = load_bin_to_gpu(path, 15360);
    if (!d_arg3) return 1;

    int32_t arg4 = 12;
    int32_t arg5 = 10;
    int32_t arg6 = 64;
    float arg7 = 0.125f;

    // Triton scratch buffers
    void* global_scratch = alloc_gpu(6144);
    void* profile_scratch = NULL;

    void* args[] = {
        &d_arg0, &d_arg1, &d_arg2, &d_arg3,
        &arg4, &arg5, &arg6, &arg7,
        &global_scratch, &profile_scratch
    };

    printf("Launching flash_attn_kernel: grid=(1,12,1) block=(128,1,1) smem=8736\n");
    if (launch_kernel(func, 1, 12, 1, 128, 1, 1, 8736, args)) return 1;
    printf("Kernel completed\n");

    // Validate output
    build_path(path, sizeof(path), exe_dir, "data/flash_attn_kernel_launch4_arg3_output.bin");
    int rc = validate_fp16_from_file(d_arg3, path, 15360);

    cudaFree(d_arg0); cudaFree(d_arg1); cudaFree(d_arg2); cudaFree(d_arg3);
    cudaFree(global_scratch);
    cuModuleUnload(module);
    cuCtxDestroy(ctx);

    printf(rc == 0 ? "PASSED\n" : "FAILED\n");
    return rc;
}
