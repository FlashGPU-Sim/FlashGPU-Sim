# GoogleTest checkout and library build rules.

GTEST_MK := $(lastword $(MAKEFILE_LIST))

.PHONY: setup-gtest

setup-gtest:
	@if [ -f "$(GTEST_DIR)/include/gtest/gtest.h" ]; then \
		echo "Google Test already exists at $(GTEST_DIR)."; \
	elif [ "$(abspath $(GTEST_DIR))" != "$(abspath $(LOCAL_GTEST_DIR))" ]; then \
		echo "ERROR: configured GTEST_DIR is incomplete: $(GTEST_DIR)" >&2; \
		exit 1; \
	elif [ -e "$(LOCAL_GTEST_CLONE_DIR)" ]; then \
		echo "ERROR: incomplete Google Test checkout: $(LOCAL_GTEST_CLONE_DIR)" >&2; \
		exit 1; \
	else \
		echo "Downloading Google Test..."; \
		git clone --depth 1 --branch release-1.12.1 \
			https://github.com/google/googletest.git $(LOCAL_GTEST_CLONE_DIR); \
	fi

$(OBJ_DIR)/gtest-all.o: $(GTEST_SRCS_) $(TOP_MAKEFILE) $(GTEST_MK) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(GTEST_DIR) $(INCLUDES) -c \
		$(GTEST_DIR)/src/gtest-all.cc -o $@

$(OBJ_DIR)/gtest_main.o: $(GTEST_SRCS_) $(TOP_MAKEFILE) $(GTEST_MK) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -I$(GTEST_DIR) $(INCLUDES) -c \
		$(GTEST_DIR)/src/gtest_main.cc -o $@

$(OBJ_DIR)/gtest.a: $(OBJ_DIR)/gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

$(OBJ_DIR)/gtest_main.a: $(OBJ_DIR)/gtest-all.o $(OBJ_DIR)/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^
