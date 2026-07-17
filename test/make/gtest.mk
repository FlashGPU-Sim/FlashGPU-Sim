# GoogleTest setup and host-side support objects.

GTEST_MK := $(lastword $(MAKEFILE_LIST))

.PHONY: setup-gtest

setup-gtest:
	@if [ ! -d "$(GTEST_DIR)" ]; then \
		echo "Downloading Google Test..."; \
		git clone https://github.com/google/googletest.git $(GTEST_CLONE_DIR); \
		cd $(GTEST_DIR) && git checkout release-1.12.1; \
	else \
		echo "Google Test already exists."; \
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
