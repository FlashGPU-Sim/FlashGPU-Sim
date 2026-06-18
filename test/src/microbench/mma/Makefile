SRC ?= mma_accept_queue_bench.cu
TARGET_NAME ?= mma_accept_queue_bench_cuda128_sm120a
ARCH ?= sm_120a
PTX_PROFILE ?= compute_120a
include ../common.mk
BIN_DIR ?= $(STANDALONE_BIN_ROOT)/mma
TARGET ?= $(BIN_DIR)/$(TARGET_NAME)

SMOKE_ARGS ?= --samples 5 --warmup 1 --n 0,1,2,4,8,16 --modes mma,fma,empty
SWEEP_ARGS ?= --samples 101 --warmup 10 --n 0,1,2,3,4,5,6,8,10,12,16,20,24,32,40,48,56,64 --modes mma,fma,empty

.PHONY: all sm120a sm90a saturation-sm120a saturation-sm90a clean run-smoke run-sweep sass ptx sass-sm90a sass-sm120a sass-saturation-sm90a sass-saturation-sm120a

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

sm120a:
	$(MAKE) ARCH=sm_120a PTX_PROFILE=compute_120a TARGET_NAME=mma_accept_queue_bench_cuda128_sm120a all

sm90a:
	$(MAKE) ARCH=sm_90a PTX_PROFILE=compute_90a TARGET_NAME=mma_accept_queue_bench_cuda128_sm90a all

saturation-sm120a:
	$(MAKE) ARCH=sm_120a PTX_PROFILE=compute_120a SRC=mma_saturation_bench.cu TARGET_NAME=mma_saturation_bench_cuda128_sm120a all

saturation-sm90a:
	$(MAKE) ARCH=sm_90a PTX_PROFILE=compute_90a SRC=mma_saturation_bench.cu TARGET_NAME=mma_saturation_bench_cuda128_sm90a all

run-smoke: $(TARGET)
	./$(TARGET) $(SMOKE_ARGS)

run-sweep: $(TARGET)
	./$(TARGET) $(SWEEP_ARGS)

sass: $(TARGET)
	$(CUOBJDUMP) -sass $(TARGET)

ptx: $(TARGET)
	$(CUOBJDUMP) -ptx $(TARGET)

sass-sm90a: sm90a
	$(CUOBJDUMP) -sass $(BIN_DIR)/mma_accept_queue_bench_cuda128_sm90a

sass-sm120a: sm120a
	$(CUOBJDUMP) -sass $(BIN_DIR)/mma_accept_queue_bench_cuda128_sm120a

sass-saturation-sm90a: saturation-sm90a
	$(CUOBJDUMP) -sass $(BIN_DIR)/mma_saturation_bench_cuda128_sm90a

sass-saturation-sm120a: saturation-sm120a
	$(CUOBJDUMP) -sass $(BIN_DIR)/mma_saturation_bench_cuda128_sm120a

clean:
	rm -f $(BIN_DIR)/mma_accept_queue_bench_cuda128_sm120a $(BIN_DIR)/mma_accept_queue_bench_cuda128_sm90a
	rm -f $(BIN_DIR)/mma_saturation_bench_cuda128_sm120a $(BIN_DIR)/mma_saturation_bench_cuda128_sm90a
	rm -f *.ptx *.sass *.cubin
