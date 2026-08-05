# Makefile for {{KERNEL_NAME}} launch {{LAUNCH_ID}}
# Packages Triton's generated binary into a fatbinary loaded at runtime.
# PTX is kept alongside it for inspection and trace tooling.

NVCC = nvcc
FATBINARY = fatbinary
NVCC_BIN := $(shell command -v $(NVCC))
CUDA_LIB_DIR := $(abspath $(dir $(NVCC_BIN))/../lib)
CUDART_FLAG := $(shell if [ -f "$(CUDA_LIB_DIR)/libcudart.so.13" ]; then echo "-l:libcudart.so.13"; else echo "-lcudart"; fi)
CUDA_FLAGS = -L$(CUDA_LIB_DIR) -Xlinker -rpath -Xlinker $(CUDA_LIB_DIR) $(CUDART_FLAG) -lcuda
TARGET = {{TARGET_NAME}}
PTX_FILE = {{PTX_FILENAME}}
CUBIN_FILE = {{CUBIN_FILENAME}}
FATBIN_FILE = {{FATBIN_FILENAME}}
FATBIN_DEPS = {{FATBIN_DEPS}}
FATBINARY_IMAGES_LEGACY = {{FATBINARY_IMAGES_LEGACY}}
FATBINARY_IMAGES_IMAGE3 = {{FATBINARY_IMAGES_IMAGE3}}

# Auto-detect architecture from PTX .target directive. CUBIN comes from the
# same Triton compile, so this profile also matches the CUBIN payload.
ARCH := $(shell grep '^\.target' $(PTX_FILE) | head -1 | awk '{print $$2}')
ARCH_NUM := $(patsubst sm_%,%,$(ARCH))
PTX_PROFILE := $(patsubst sm_%,compute_%,$(ARCH))

all: $(TARGET) $(FATBIN_FILE)

# Package Triton's CUBIN when available. Falling back to PTX-only avoids ptxas,
# but may require a driver that understands the PTX version emitted by Triton.
$(FATBIN_FILE): $(FATBIN_DEPS)
	@echo "Detected architecture: $(ARCH)"
	@if $(FATBINARY) --help 2>&1 | grep -q -- '--image3'; then \
		$(FATBINARY) --create=$(FATBIN_FILE) $(FATBINARY_IMAGES_IMAGE3); \
	else \
		$(FATBINARY) --create=$(FATBIN_FILE) $(FATBINARY_IMAGES_LEGACY); \
	fi

$(TARGET): {{HARNESS_FILENAME}}
	$(NVCC) -arch=$(ARCH) -o $(TARGET) {{HARNESS_FILENAME}} $(CUDA_FLAGS)

clean:
	rm -f $(TARGET) $(FATBIN_FILE)

.PHONY: all clean
