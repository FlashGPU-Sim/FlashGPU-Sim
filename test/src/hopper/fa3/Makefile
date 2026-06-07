CUDA_ARCH ?= sm_90a

.PHONY: all test clean

all: test

test:
	$(MAKE) -C ../../.. HOPPER_CUDA_ARCH=$(CUDA_ARCH) hopper-gtest

clean:
	$(MAKE) -C ../../.. clean
