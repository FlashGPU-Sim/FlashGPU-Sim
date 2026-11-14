# One source-derived list for compilation and final shared-library linking.
# Never enumerate build-tree *.o files: renamed/moved sources leave stale
# objects behind. Standalone tools are deliberately excluded.
SASS_SOURCE_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
SASS_SRCS := $(wildcard $(addprefix $(SASS_SOURCE_DIR),\
    *.cc decode/*.cc functional/*.cc runtime/*.cc timing/*.cc))
