CUDA_INSTALL_PATH ?= /usr/local/cuda
NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc

TARGET := vectorAdd
CUDA_ARCH ?= 120a

NVCC_FLAGS ?= -O2 -std=c++17
GENCODE_FLAGS := \
	-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
	-gencode arch=compute_$(CUDA_ARCH),code=compute_$(CUDA_ARCH)

.PHONY: all clean

all: $(TARGET)

$(TARGET): vectorAdd.cu
	$(NVCC) $(NVCC_FLAGS) -cudart shared $(GENCODE_FLAGS) $< -o $@

clean:
	$(RM) $(TARGET)
