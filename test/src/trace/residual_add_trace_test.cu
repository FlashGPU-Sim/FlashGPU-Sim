// GPT-2 Residual Add Kernel Trace Test (Data-Driven)
//
// Kernel: residual_add_kernel (element-wise add, in-place on arg0)
// Grid: (8, 1, 1), Block: (128, 1, 1), Shared mem: 2056 bytes
// Args: arg0 [10,768] fp16 (input/output), arg1 [10,768] fp16 (residual),
//       arg2=7680 (num_elements)
// Output: arg0 (in-place)
// Validates against captured GPU output from real hardware.

#include "trace_test_utils.h"

int main() {
    printf("=== GPT-2 Residual Add Trace Test ===\n");

    CUcontext ctx;
    if (init_cuda(&ctx)) { fprintf(stderr, "CUDA init failed\n"); return 1; }

    char exe_dir[1024];
    get_exe_dir(exe_dir, sizeof(exe_dir));
    char path[2048];

    // Load fatbin
    build_path(path, sizeof(path), exe_dir, "residual_add_kernel.fatbin");
    CUmodule module = load_fatbin(path);
    if (!module) return 1;
    CUfunction func = get_kernel(module, "residual_add_kernel");
    if (!func) return 1;
    set_shared_mem(func, 2056);

    // Load input data
    build_path(path, sizeof(path), exe_dir, "data/residual_add_kernel_launch6_arg0.bin");
    void* d_arg0 = load_bin_to_gpu(path, 15360);
    if (!d_arg0) return 1;

    build_path(path, sizeof(path), exe_dir, "data/residual_add_kernel_launch6_arg1.bin");
    void* d_arg1 = load_bin_to_gpu(path, 15360);
    if (!d_arg1) return 1;

    int32_t arg2 = 7680;

    // Triton scratch buffers
    void* global_scratch = alloc_gpu(2048);
    void* profile_scratch = NULL;

    void* args[] = { &d_arg0, &d_arg1, &arg2, &global_scratch, &profile_scratch };

    printf("Launching residual_add_kernel: grid=(8,1,1) block=(128,1,1) smem=2056\n");
    if (launch_kernel(func, 8, 1, 1, 128, 1, 1, 2056, args)) return 1;
    printf("Kernel completed\n");

    // Validate output (in-place on arg0)
    build_path(path, sizeof(path), exe_dir, "data/residual_add_kernel_launch6_arg0_output.bin");
    int rc = validate_fp16_from_file(d_arg0, path, 15360);

    cudaFree(d_arg0); cudaFree(d_arg1); cudaFree(global_scratch);
    cuModuleUnload(module);
    cuCtxDestroy(ctx);

    printf(rc == 0 ? "PASSED\n" : "FAILED\n");
    return rc;
}
