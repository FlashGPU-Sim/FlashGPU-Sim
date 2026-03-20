// GPT-2 Linear Kernel Trace Test (Data-Driven)
//
// Kernel: linear_kernel (matrix multiply + bias)
// Grid: (1, 12, 1), Block: (128, 1, 1), Shared mem: 16400 bytes
// Args: arg0 [10,768] fp16 (input), arg1 [768,768] fp16 (weight),
//       arg2 [768] fp16 (bias), arg3 [10,768] fp16 (output),
//       arg4=10 (M), arg5=768 (N), arg6=768 (K), arg7=768 (stride)
// Output: arg3
// Validates against captured GPU output from real hardware.

#include "trace_test_utils.h"

int main() {
    printf("=== GPT-2 Linear Trace Test ===\n");

    CUcontext ctx;
    if (init_cuda(&ctx)) { fprintf(stderr, "CUDA init failed\n"); return 1; }

    char exe_dir[1024];
    get_exe_dir(exe_dir, sizeof(exe_dir));
    char path[2048];

    // Load fatbin
    build_path(path, sizeof(path), exe_dir, "linear_kernel.fatbin");
    CUmodule module = load_fatbin(path);
    if (!module) return 1;
    CUfunction func = get_kernel(module, "linear_kernel");
    if (!func) return 1;
    set_shared_mem(func, 16400);

    // Load input data
    build_path(path, sizeof(path), exe_dir, "data/linear_kernel_launch5_arg0.bin");
    void* d_arg0 = load_bin_to_gpu(path, 15360);
    if (!d_arg0) return 1;

    build_path(path, sizeof(path), exe_dir, "data/linear_kernel_launch5_arg1.bin");
    void* d_arg1 = load_bin_to_gpu(path, 1179648);
    if (!d_arg1) return 1;

    build_path(path, sizeof(path), exe_dir, "data/linear_kernel_launch5_arg2.bin");
    void* d_arg2 = load_bin_to_gpu(path, 1536);
    if (!d_arg2) return 1;

    build_path(path, sizeof(path), exe_dir, "data/linear_kernel_launch5_arg3.bin");
    void* d_arg3 = load_bin_to_gpu(path, 15360);
    if (!d_arg3) return 1;

    int32_t arg4 = 10;
    int32_t arg5 = 768;
    int32_t arg6 = 768;
    int32_t arg7 = 768;

    // Triton scratch buffers
    void* global_scratch = alloc_gpu(6144);
    void* profile_scratch = NULL;

    void* args[] = {
        &d_arg0, &d_arg1, &d_arg2, &d_arg3,
        &arg4, &arg5, &arg6, &arg7,
        &global_scratch, &profile_scratch
    };

    printf("Launching linear_kernel: grid=(1,12,1) block=(128,1,1) smem=16400\n");
    if (launch_kernel(func, 1, 12, 1, 128, 1, 1, 16400, args)) return 1;
    printf("Kernel completed\n");

    // Validate output
    build_path(path, sizeof(path), exe_dir, "data/linear_kernel_launch5_arg3_output.bin");
    int rc = validate_fp16_from_file(d_arg3, path, 15360);

    cudaFree(d_arg0); cudaFree(d_arg1); cudaFree(d_arg2); cudaFree(d_arg3);
    cudaFree(global_scratch);
    cuModuleUnload(module);
    cuCtxDestroy(ctx);

    printf(rc == 0 ? "PASSED\n" : "FAILED\n");
    return rc;
}
