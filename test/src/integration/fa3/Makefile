NVCC ?= nvcc

TARGET := fa3_fwd_hdim128_fp16_sm90_driver
SRC := $(TARGET).cu
PTX := $(TARGET).ptx

COMMON_FLAGS := -O3 -std=c++17 --ftemplate-backtrace-limit=0 --use_fast_math \
	-lineinfo --expt-relaxed-constexpr \
	-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED \
	-DCUTLASS_ENABLE_GDC_FOR_SM90 \
	-DCUTLASS_DEBUG_TRACE_LEVEL=0 \
	-DNDEBUG \
	-Iflash-attention/hopper \
	-Iflash-attention/csrc/cutlass/include

.PHONY: all ptx bin clean

all: ptx bin

ptx: $(PTX)

$(PTX): $(SRC)
	$(NVCC) -ptx $< -o $@ $(COMMON_FLAGS) -arch=compute_90a

bin: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $< -o $@ $(COMMON_FLAGS) -gencode arch=compute_90a,code=sm_90a

clean:
	rm -f $(TARGET)
