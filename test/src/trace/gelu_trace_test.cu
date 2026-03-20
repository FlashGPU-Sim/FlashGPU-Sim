// GPT-2 GELU Kernel Trace Test (Data-Driven)
//
// Kernel: gelu_kernel (approximate GELU activation)
// Grid: (30, 1, 1), Block: (128, 1, 1), Shared mem: 2056 bytes
// Args: arg0 [30720] fp16 (input), arg1 [30720] fp16 (output), arg2=30720 (int32)
// Output: arg1
// Validates against captured GPU output from real hardware.

#include "trace_test_utils.h"

int main() {
    printf("=== GPT-2 GELU Trace Test ===\n");

    CUcontext ctx;
    if (init_cuda(&ctx)) { fprintf(stderr, "CUDA init failed\n"); return 1; }

    char exe_dir[1024];
    get_exe_dir(exe_dir, sizeof(exe_dir));
    char path[2048];

    // Load fatbin
    build_path(path, sizeof(path), exe_dir, "gelu_kernel.fatbin");
    CUmodule module = load_fatbin(path);
    if (!module) return 1;
    CUfunction func = get_kernel(module, "gelu_kernel");
    if (!func) return 1;
    set_shared_mem(func, 2056);

    // Load input data
    build_path(path, sizeof(path), exe_dir, "data/gelu_kernel_launch9_arg0.bin");
    void* d_arg0 = load_bin_to_gpu(path, 61440);
    if (!d_arg0) return 1;

    build_path(path, sizeof(path), exe_dir, "data/gelu_kernel_launch9_arg1.bin");
    void* d_arg1 = load_bin_to_gpu(path, 61440);
    if (!d_arg1) return 1;

    int32_t arg2 = 30720;

    // Triton scratch buffers
    void* global_scratch = alloc_gpu(7680);
    void* profile_scratch = NULL;

    void* args[] = { &d_arg0, &d_arg1, &arg2, &global_scratch, &profile_scratch };

    printf("Launching gelu_kernel: grid=(30,1,1) block=(128,1,1) smem=2056\n");
    if (launch_kernel(func, 30, 1, 1, 128, 1, 1, 2056, args)) return 1;
    printf("Kernel completed\n");

    // Validate output
    build_path(path, sizeof(path), exe_dir, "data/gelu_kernel_launch9_arg1_output.bin");
    int rc = validate_fp16_from_file(d_arg1, path, 61440);

    cudaFree(d_arg0); cudaFree(d_arg1); cudaFree(global_scratch);
    cuModuleUnload(module);
    cuCtxDestroy(ctx);

    printf(rc == 0 ? "PASSED\n" : "FAILED\n");
    return rc;
}
