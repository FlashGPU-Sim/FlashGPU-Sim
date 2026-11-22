// Helper function to load tensor from binary file
void* load_tensor_arg(const char* exe_path, const char* rel_path, size_t size, int arg_idx,
                      const char* dtype, const char* shape) {
    // Handle zero-size tensors (e.g., dummy bias when HAS_BIAS=False)
    if (size == 0) {
        printf("  Loaded arg[%d]: tensor shape=%s, dtype=%s, size=0 bytes (zero-size, using NULL)\n", arg_idx, shape, dtype);
        return NULL;
    }

    void* d_ptr;
    cudaMalloc(&d_ptr, size);

    char data_path[2048];
    snprintf(data_path, sizeof(data_path), "%s/%s", exe_path, rel_path);
    FILE* fp = fopen(data_path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s\n", data_path);
        return NULL;
    }

    void* h_ptr = malloc(size);
    fread(h_ptr, 1, size, fp);
    fclose(fp);
    cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
    free(h_ptr);

    printf("  Loaded arg[%d]: tensor shape=%s, dtype=%s, size=%zu bytes\n", arg_idx, shape, dtype, size);
    return d_ptr;
}

// Optionally copy one tensor argument back as uint64_t values.  This keeps
// profiling buffers usable in both native launches and simulator replays
// without specializing generated harnesses for a particular kernel.
void maybe_dump_u64_arg(void* d_ptr, size_t size, int arg_idx) {
    const char* selected = getenv("TRITON_TRACE_DUMP_U64_ARG");
    if (!selected || atoi(selected) != arg_idx || !d_ptr || size == 0) {
        return;
    }

    size_t count = size / sizeof(uint64_t);
    const char* requested_count = getenv("TRITON_TRACE_DUMP_U64_COUNT");
    if (requested_count) {
        const unsigned long requested = strtoul(requested_count, NULL, 10);
        if (requested < count) count = requested;
    }

    uint64_t* values = (uint64_t*)malloc(count * sizeof(uint64_t));
    if (!values) {
        fprintf(stderr, "Error: Could not allocate uint64 dump buffer\n");
        return;
    }
    const cudaError_t copy_result = cudaMemcpy(
        values, d_ptr, count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    if (copy_result != cudaSuccess) {
        fprintf(stderr, "Error: Could not dump arg[%d]: %s\n", arg_idx,
                cudaGetErrorString(copy_result));
        free(values);
        return;
    }

    printf("TRITON_TRACE_U64_ARG[%d] count=%zu", arg_idx, count);
    for (size_t index = 0; index < count; ++index) {
        printf(" %llu", (unsigned long long)values[index]);
    }
    printf("\n");
    free(values);
}
