// GPT-2 Data-Driven Trace Tests (unified runner)
//
// Runs Triton kernel PTX against captured GPU data and validates output.
// Usage: data_driven_trace_test <test_name>
//   test_name: gelu | flash_attn | layernorm | residual_add | linear
//
// Each test loads input .bin files, launches the kernel, and compares
// GPU output against expected .bin captured from real hardware.

#include "trace_test_utils.h"

// Scalar argument: either int32 or float, stored in a union
enum scalar_type { SCALAR_I32, SCALAR_F32 };

struct scalar_arg {
    enum scalar_type type;
    union {
        int32_t i32;
        float f32;
    };
};

// Pointer argument: file path suffix and byte size
struct ptr_arg {
    const char* file_suffix;  // e.g., "arg0.bin"
    size_t size;
};

// Full test configuration
struct trace_test_config {
    const char* name;           // display name
    const char* kernel_name;    // PTX function name
    const char* fatbin_name;    // fatbin filename
    const char* data_prefix;    // data file prefix, e.g., "gelu_kernel_launch9"

    unsigned grid[3];
    unsigned block[3];
    unsigned smem;

    int num_ptr_args;
    struct ptr_arg ptr_args[8];

    int num_scalar_args;
    struct scalar_arg scalar_args[8];

    int output_arg_idx;         // which ptr_arg is the output
    const char* output_suffix;  // expected output file suffix
    size_t output_size;
};

// ─── Test Configurations ────────────────────────────────────────────────

static const struct trace_test_config TESTS[] = {
    {
        .name = "gelu",
        .kernel_name = "gelu_kernel",
        .fatbin_name = "gelu_kernel.fatbin",
        .data_prefix = "gelu_kernel_launch9",
        .grid = {30, 1, 1}, .block = {128, 1, 1}, .smem = 2056,
        .num_ptr_args = 2,
        .ptr_args = {
            {"arg0.bin", 61440},
            {"arg1.bin", 61440},
        },
        .num_scalar_args = 1,
        .scalar_args = {{SCALAR_I32, {.i32 = 30720}}},
        .output_arg_idx = 1,
        .output_suffix = "arg1_output.bin",
        .output_size = 61440,
    },
    {
        .name = "flash_attn",
        .kernel_name = "flash_attn_kernel",
        .fatbin_name = "flash_attn_kernel.fatbin",
        .data_prefix = "flash_attn_kernel_launch4",
        .grid = {1, 12, 1}, .block = {128, 1, 1}, .smem = 8736,
        .num_ptr_args = 4,
        .ptr_args = {
            {"arg0.bin", 15360},
            {"arg1.bin", 15360},
            {"arg2.bin", 15360},
            {"arg3.bin", 15360},
        },
        .num_scalar_args = 4,
        .scalar_args = {
            {SCALAR_I32, {.i32 = 12}},
            {SCALAR_I32, {.i32 = 10}},
            {SCALAR_I32, {.i32 = 64}},
            {SCALAR_F32, {.f32 = 0.125f}},
        },
        .output_arg_idx = 3,
        .output_suffix = "arg3_output.bin",
        .output_size = 15360,
    },
    {
        .name = "layernorm",
        .kernel_name = "layernorm_kernel",
        .fatbin_name = "layernorm_kernel.fatbin",
        .data_prefix = "layernorm_kernel_launch2",
        .grid = {10, 1, 1}, .block = {128, 1, 1}, .smem = 2056,
        .num_ptr_args = 4,
        .ptr_args = {
            {"arg0.bin", 15360},
            {"arg1.bin", 1536},
            {"arg2.bin", 1536},
            {"arg3.bin", 15360},
        },
        .num_scalar_args = 2,
        .scalar_args = {
            {SCALAR_I32, {.i32 = 10}},
            {SCALAR_I32, {.i32 = 768}},
        },
        .output_arg_idx = 3,
        .output_suffix = "arg3_output.bin",
        .output_size = 15360,
    },
    {
        .name = "residual_add",
        .kernel_name = "residual_add_kernel",
        .fatbin_name = "residual_add_kernel.fatbin",
        .data_prefix = "residual_add_kernel_launch6",
        .grid = {8, 1, 1}, .block = {128, 1, 1}, .smem = 2056,
        .num_ptr_args = 2,
        .ptr_args = {
            {"arg0.bin", 15360},
            {"arg1.bin", 15360},
        },
        .num_scalar_args = 1,
        .scalar_args = {{SCALAR_I32, {.i32 = 7680}}},
        .output_arg_idx = 0,
        .output_suffix = "arg0_output.bin",
        .output_size = 15360,
    },
    {
        .name = "linear",
        .kernel_name = "linear_kernel",
        .fatbin_name = "linear_kernel.fatbin",
        .data_prefix = "linear_kernel_launch5",
        .grid = {1, 12, 1}, .block = {128, 1, 1}, .smem = 16400,
        .num_ptr_args = 4,
        .ptr_args = {
            {"arg0.bin", 15360},
            {"arg1.bin", 1179648},
            {"arg2.bin", 1536},
            {"arg3.bin", 15360},
        },
        .num_scalar_args = 4,
        .scalar_args = {
            {SCALAR_I32, {.i32 = 10}},
            {SCALAR_I32, {.i32 = 768}},
            {SCALAR_I32, {.i32 = 768}},
            {SCALAR_I32, {.i32 = 768}},
        },
        .output_arg_idx = 3,
        .output_suffix = "arg3_output.bin",
        .output_size = 15360,
    },
};

static const int NUM_TESTS = sizeof(TESTS) / sizeof(TESTS[0]);

// ─── Generic Test Runner ────────────────────────────────────────────────

static int run_test(const struct trace_test_config* t) {
    printf("=== GPT-2 %s Trace Test ===\n", t->name);

    CUcontext ctx;
    if (init_cuda(&ctx)) { fprintf(stderr, "CUDA init failed\n"); return 1; }

    char exe_dir[1024];
    get_exe_dir(exe_dir, sizeof(exe_dir));
    char path[2048];

    // Load fatbin and get kernel
    build_path(path, sizeof(path), exe_dir, t->fatbin_name);
    CUmodule module = load_fatbin(path);
    if (!module) return 1;
    CUfunction func = get_kernel(module, t->kernel_name);
    if (!func) return 1;
    set_shared_mem(func, t->smem);

    // Load pointer args from .bin files
    void* d_ptrs[8] = {};
    for (int i = 0; i < t->num_ptr_args; i++) {
        char data_file[256];
        snprintf(data_file, sizeof(data_file), "data/%s_%s",
                 t->data_prefix, t->ptr_args[i].file_suffix);
        build_path(path, sizeof(path), exe_dir, data_file);
        d_ptrs[i] = load_bin_to_gpu(path, t->ptr_args[i].size);
        if (!d_ptrs[i]) return 1;
    }

    // Triton scratch buffers
    void* global_scratch = alloc_gpu(8192);
    void* profile_scratch = NULL;

    // Build args array: [ptr_args..., scalar_args..., global_scratch, profile_scratch]
    // Max: 8 ptrs + 8 scalars + 2 scratch = 18
    void* args[18];
    int arg_idx = 0;

    for (int i = 0; i < t->num_ptr_args; i++)
        args[arg_idx++] = &d_ptrs[i];

    for (int i = 0; i < t->num_scalar_args; i++)
        args[arg_idx++] = (void*)&t->scalar_args[i].i32;  // union, same address

    args[arg_idx++] = &global_scratch;
    args[arg_idx++] = &profile_scratch;

    // Launch
    printf("Launching %s: grid=(%u,%u,%u) block=(%u,%u,%u) smem=%u\n",
           t->kernel_name, t->grid[0], t->grid[1], t->grid[2],
           t->block[0], t->block[1], t->block[2], t->smem);
    if (launch_kernel(func, t->grid[0], t->grid[1], t->grid[2],
                      t->block[0], t->block[1], t->block[2], t->smem, args))
        return 1;
    printf("Kernel completed\n");

    // Validate output
    char output_file[256];
    snprintf(output_file, sizeof(output_file), "data/%s_%s",
             t->data_prefix, t->output_suffix);
    build_path(path, sizeof(path), exe_dir, output_file);
    int rc = validate_fp16_from_file(d_ptrs[t->output_arg_idx], path, t->output_size);

    // Cleanup
    for (int i = 0; i < t->num_ptr_args; i++)
        cudaFree(d_ptrs[i]);
    cudaFree(global_scratch);
    cuModuleUnload(module);
    cuCtxDestroy(ctx);

    printf(rc == 0 ? "PASSED\n" : "FAILED\n");
    return rc;
}

// ─── Main ───────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <test_name>\n", argv[0]);
        fprintf(stderr, "Available tests:");
        for (int i = 0; i < NUM_TESTS; i++)
            fprintf(stderr, " %s", TESTS[i].name);
        fprintf(stderr, "\n");
        return 1;
    }

    const char* test_name = argv[1];
    for (int i = 0; i < NUM_TESTS; i++) {
        if (strcmp(TESTS[i].name, test_name) == 0)
            return run_test(&TESTS[i]);
    }

    fprintf(stderr, "Unknown test: %s\n", test_name);
    return 1;
}
