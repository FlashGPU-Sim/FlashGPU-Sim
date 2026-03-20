// Shared utilities for GPT-2 trace tests
// Each test loads a PTX kernel (via fatbin), allocates GPU memory,
// loads/generates data, launches the kernel, and validates output.

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>

// FP16 conversion helpers
static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    if (exponent == 0) {
        if (mantissa == 0) return sign ? -0.0f : 0.0f;
        float val = mantissa / 1024.0f / 16384.0f;
        return sign ? -val : val;
    } else if (exponent == 31) {
        return mantissa == 0 ? (sign ? -INFINITY : INFINITY) : NAN;
    }
    float val = (1.0f + mantissa / 1024.0f) * powf(2.0f, (float)((int)exponent - 15));
    return sign ? -val : val;
}

static inline uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exponent = ((bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = bits & 0x7FFFFF;
    if (exponent > 15) return (sign << 15) | 0x7C00;  // inf
    if (exponent < -14) return (sign << 15);           // zero/denorm
    uint16_t h = (sign << 15) | ((exponent + 15) << 10) | (mantissa >> 13);
    return h;
}

// Initialize CUDA driver API context
static int init_cuda(CUcontext* ctx) {
    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    return cuCtxCreate(ctx, 0, dev) == CUDA_SUCCESS ? 0 : 1;
}

// Get directory of the executable
static void get_exe_dir(char* buf, size_t bufsize) {
    ssize_t len = readlink("/proc/self/exe", buf, bufsize - 1);
    if (len == -1) { buf[0] = '.'; buf[1] = '\0'; return; }
    buf[len] = '\0';
    char* slash = strrchr(buf, '/');
    if (slash) *slash = '\0';
}

// Load fatbin module from file
static CUmodule load_fatbin(const char* fatbin_path) {
    CUmodule module;
    CUresult r = cuModuleLoad(&module, fatbin_path);
    if (r != CUDA_SUCCESS) {
        const char* err;
        cuGetErrorString(r, &err);
        fprintf(stderr, "Failed to load %s: %s\n", fatbin_path, err);
        return NULL;
    }
    return module;
}

// Get kernel function from module
static CUfunction get_kernel(CUmodule module, const char* name) {
    CUfunction func;
    CUresult r = cuModuleGetFunction(&func, module, name);
    if (r != CUDA_SUCCESS) {
        const char* err;
        cuGetErrorString(r, &err);
        fprintf(stderr, "Failed to get function '%s': %s\n", name, err);
        return NULL;
    }
    return func;
}

// Set max dynamic shared memory for a kernel
static void set_shared_mem(CUfunction func, int bytes) {
    CUresult r = cuFuncSetAttribute(
        func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes);
    if (r != CUDA_SUCCESS) {
        const char* err;
        cuGetErrorString(r, &err);
        fprintf(stderr, "Warning: Failed to set shared memory size: %s\n", err);
    }
}

// Allocate GPU memory and zero it
static void* alloc_gpu(size_t size) {
    void* ptr;
    cudaMalloc(&ptr, size);
    cudaMemset(ptr, 0, size);
    return ptr;
}

// Load binary data from file into a GPU buffer
static void* load_bin_to_gpu(const char* path, size_t size) {
    void* d_ptr;
    cudaMalloc(&d_ptr, size);

    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open %s\n", path);
        cudaFree(d_ptr);
        return NULL;
    }
    void* h_ptr = malloc(size);
    size_t read = fread(h_ptr, 1, size, fp);
    fclose(fp);
    if (read != size) {
        fprintf(stderr, "Short read from %s: got %zu, expected %zu\n", path, read, size);
    }
    cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
    free(h_ptr);
    return d_ptr;
}

// Launch kernel and synchronize
static int launch_kernel(CUfunction func,
                         unsigned gx, unsigned gy, unsigned gz,
                         unsigned bx, unsigned by, unsigned bz,
                         unsigned smem, void** args) {
    CUresult r = cuLaunchKernel(func, gx, gy, gz, bx, by, bz, smem, 0, args, 0);
    if (r != CUDA_SUCCESS) {
        const char* err;
        cuGetErrorString(r, &err);
        fprintf(stderr, "Kernel launch failed: %s\n", err);
        return 1;
    }
    cuCtxSynchronize();
    return 0;
}

// Validate FP16 GPU output against expected binary file
// Returns 0 on pass, 1 on failure
static int validate_fp16_from_file(void* d_actual, const char* expected_path, size_t size) {
    FILE* fp = fopen(expected_path, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open expected output: %s\n", expected_path);
        return 1;
    }

    size_t num_elements = size / 2;
    uint16_t* h_expected = (uint16_t*)malloc(size);
    uint16_t* h_actual = (uint16_t*)malloc(size);
    fread(h_expected, 1, size, fp);
    fclose(fp);
    cudaMemcpy(h_actual, d_actual, size, cudaMemcpyDeviceToHost);

    int mismatches = 0;
    float rel_tol = 1e-3f;
    float abs_tol = 1e-2f;

    for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
        float exp_f = fp16_to_fp32(h_expected[i]);
        float act_f = fp16_to_fp32(h_actual[i]);
        float diff = fabsf(exp_f - act_f);
        float max_abs = fmaxf(fabsf(exp_f), fabsf(act_f));
        float rel_err = (max_abs > 0) ? (diff / max_abs) : 0;
        int pass = (rel_err <= rel_tol) || (diff <= abs_tol);
        if (!pass) {
            if (mismatches == 0) printf("  VALIDATION FAILED:\n");
            printf("    [%zu] expected=%.6f actual=%.6f diff=%.6e rel_err=%.6e\n",
                   i, exp_f, act_f, diff, rel_err);
            mismatches++;
        }
    }

    free(h_expected);
    free(h_actual);

    if (mismatches == 0) {
        printf("  Validation PASSED (%zu fp16 elements)\n", num_elements);
        return 0;
    }
    printf("  %d mismatches (showing first 10)\n", mismatches);
    return 1;
}

// Validate FP16 GPU output against a host-side expected buffer
// Returns 0 on pass, 1 on failure
static int validate_fp16_from_host(void* d_actual, const uint16_t* h_expected, size_t num_elements) {
    size_t size = num_elements * 2;
    uint16_t* h_actual = (uint16_t*)malloc(size);
    cudaMemcpy(h_actual, d_actual, size, cudaMemcpyDeviceToHost);

    int mismatches = 0;
    float rel_tol = 1e-3f;
    float abs_tol = 1e-2f;

    for (size_t i = 0; i < num_elements && mismatches < 10; i++) {
        float exp_f = fp16_to_fp32(h_expected[i]);
        float act_f = fp16_to_fp32(h_actual[i]);
        float diff = fabsf(exp_f - act_f);
        float max_abs = fmaxf(fabsf(exp_f), fabsf(act_f));
        float rel_err = (max_abs > 0) ? (diff / max_abs) : 0;
        int pass = (rel_err <= rel_tol) || (diff <= abs_tol);
        if (!pass) {
            if (mismatches == 0) printf("  VALIDATION FAILED:\n");
            printf("    [%zu] expected=%.6f actual=%.6f diff=%.6e rel_err=%.6e\n",
                   i, exp_f, act_f, diff, rel_err);
            mismatches++;
        }
    }

    free(h_actual);

    if (mismatches == 0) {
        printf("  Validation PASSED (%zu fp16 elements)\n", num_elements);
        return 0;
    }
    printf("  %d mismatches (showing first 10)\n", mismatches);
    return 1;
}

// Build a path relative to the executable directory
static void build_path(char* out, size_t outsize, const char* exe_dir, const char* rel) {
    snprintf(out, outsize, "%s/%s", exe_dir, rel);
}
