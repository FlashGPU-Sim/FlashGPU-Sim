# Makefile for {{KERNEL_NAME}} launch {{LAUNCH_ID}}

NVCC = nvcc
CUDA_FLAGS = -lcuda -lcudart
TARGET = {{TARGET_NAME}}

all: $(TARGET)

$(TARGET): {{HARNESS_FILENAME}}
	$(NVCC) -o $(TARGET) {{HARNESS_FILENAME}} $(CUDA_FLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
