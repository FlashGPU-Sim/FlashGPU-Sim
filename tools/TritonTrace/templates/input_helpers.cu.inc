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
