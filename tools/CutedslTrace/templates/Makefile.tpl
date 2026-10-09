CXX ?= g++
CUDA_HOME ?= $(if $(CUDA_INSTALL_PATH),$(CUDA_INSTALL_PATH),/usr/local/cuda)
TVM_FFI_ROOT ?= @TVM_ROOT@
CUTEDSL_LIB ?= @DSL_LIB@
CXXFLAGS ?= -O2 -std=c++17

all: replay

# Resolve CUDA entry points on use: CuTe's archive also contains unused APIs.
replay: harness.cc Makefile $(wildcard modules/*.o)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I"$(CUDA_HOME)/include" -I"$(TVM_FFI_ROOT)/include" -I"$(TVM_FFI_ROOT)/3rdparty/dlpack/include" harness.cc modules/*.o $(LDFLAGS) "$(CUTEDSL_LIB)/libcuda_dialect_runtime_static.a" -L"$(CUDA_HOME)/lib64" -Wl,-z,lazy,--enable-new-dtags,-rpath,"$(CUDA_HOME)/lib64" -lcudart -L"$(TVM_FFI_ROOT)/lib" -Wl,-rpath,"$(TVM_FFI_ROOT)/lib" -ltvm_ffi -ldl -pthread $(LDLIBS) -o $@

clean:
	rm -f replay

.PHONY: all clean
