# Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
# Timothy G. Rogers
# The University of British Columbia
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
# Neither the name of The University of British Columbia nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# GPGPU-Sim Makefile

DEBUG?=0
TRACE?=0

ifeq ($(DEBUG),1)
	CXXFLAGS = -Wall -DDEBUG -fopenmp $(EXTRA_FLAGS)
else
	CXXFLAGS = -Wall -fopenmp $(EXTRA_FLAGS)
endif

ifeq ($(TRACE),1)
	CXXFLAGS += -DTRACING_ON=1
endif

include ../../version_detection.mk

ifeq ($(GNUC_CPP0X), 1)
    CXXFLAGS += -std=c++17
endif

ifneq ($(DEBUG),1)
	OPTFLAGS += -O3
else
	CXXFLAGS += 
endif

# Include cuda-sim to get ptx.tab.h for mbarrier.cc
CXXFLAGS += -I$(CUDA_INSTALL_PATH)/include \
-I$(SIM_OBJ_FILES_DIR)/cuda-sim \


POWER_FLAGS=
ifneq ($(GPGPUSIM_POWER_MODEL),)
	POWER_FLAGS = -I$(GPGPUSIM_POWER_MODEL) -DGPGPUSIM_POWER_MODEL
endif

OPTFLAGS += -g3 -fPIC
OPTFLAGS += -DCUDART_VERSION=$(CUDART_VERSION)

CPP = g++ $(SNOW)
OEXT = o

OUTPUT_DIR=$(SIM_OBJ_FILES_DIR)/gpgpu-sim

SRCS = $(shell ls *.cc) $(shell ls flash/*.cc) $(shell ls flash/mma/*.cc) $(shell ls flash/wgmma/*.cc) $(shell ls flash/ptx_sched/*.cc)

ifeq ($(FLASH_GEM_FORGE),1)
SRCS += $(shell ls flash/gem5/*.cc)
endif

EXCLUDES = 

ifeq ($(GPGPUSIM_POWER_MODEL), )
EXCLUDES += power_interface.cc
endif

CSRCS = $(filter-out $(EXCLUDES), $(SRCS))

OBJS = $(CSRCS:%.cc=$(OUTPUT_DIR)/%.$(OEXT))
 
libgpu_uarch_sim.a:$(OBJS)
	ar rcs  $(OUTPUT_DIR)/libgpu_uarch_sim.a $(OBJS)

# Create flash directory if it doesn't exist
$(OUTPUT_DIR)/flash:
	mkdir -p $(OUTPUT_DIR)/flash

$(OUTPUT_DIR)/flash/mma:
	mkdir -p $(OUTPUT_DIR)/flash/mma

$(OUTPUT_DIR)/flash/wgmma:
	mkdir -p $(OUTPUT_DIR)/flash/wgmma

$(OUTPUT_DIR)/flash/ptx_sched:
	mkdir -p $(OUTPUT_DIR)/flash/ptx_sched

$(OUTPUT_DIR)/flash/gem5:
	mkdir -p $(OUTPUT_DIR)/flash/gem5

$(OUTPUT_DIR)/Makefile.makedepend: depend

depend:
	touch $(OUTPUT_DIR)/Makefile.makedepend
	makedepend -f$(OUTPUT_DIR)/Makefile.makedepend -p$(OUTPUT_DIR)/ $(CSRCS) 2> /dev/null

$(OUTPUT_DIR)/%.$(OEXT): %.cc
	@mkdir -p $(dir $@)
	$(CPP) $(OPTFLAGS) $(CXXFLAGS) $(POWER_FLAGS) -o $@ -c $<

# Specific rule for flash subdirectory
$(OUTPUT_DIR)/flash/%.$(OEXT): flash/%.cc | $(OUTPUT_DIR)/flash
	$(CPP) $(OPTFLAGS) $(CXXFLAGS) $(POWER_FLAGS) -o $@ -c $<

# Specific rule for flash/mma subdirectory
$(OUTPUT_DIR)/flash/mma/%.$(OEXT): flash/mma/%.cc | $(OUTPUT_DIR)/flash/mma
	$(CPP) $(OPTFLAGS) $(CXXFLAGS) $(POWER_FLAGS) -o $@ -c $<

# Specific rule for flash/wgmma subdirectory
$(OUTPUT_DIR)/flash/wgmma/%.$(OEXT): flash/wgmma/%.cc | $(OUTPUT_DIR)/flash/wgmma
	$(CPP) $(OPTFLAGS) $(CXXFLAGS) $(POWER_FLAGS) -o $@ -c $<

# Specific rule for flash/ptx_sched subdirectory
$(OUTPUT_DIR)/flash/ptx_sched/%.$(OEXT): flash/ptx_sched/%.cc | $(OUTPUT_DIR)/flash/ptx_sched
	$(CPP) $(OPTFLAGS) $(CXXFLAGS) $(POWER_FLAGS) -o $@ -c $<

# Specific rule for flash/gem5 subdirectory
$(OUTPUT_DIR)/flash/gem5/%.$(OEXT): flash/gem5/%.cc | $(OUTPUT_DIR)/flash/gem5
	$(CPP) $(OPTFLAGS) $(CXXFLAGS) $(POWER_FLAGS) -o $@ -c $<

clean:
	rm -f *.o core *~ *.a 
	rm -f Makefile.makedepend Makefile.makedepend.bak

$(OUTPUT_DIR)/option_parser.$(OEXT): option_parser.h

$(OBJS): $(OUTPUT_DIR)/../cuda-sim/ptx.tab.h

$(OUTPUT_DIR)/../cuda-sim/ptx.tab.h:
	make -C ../cuda-sim/ $(OUTPUT_DIR)/../cuda-sim/ptx.tab.c

include $(OUTPUT_DIR)/Makefile.makedepend
