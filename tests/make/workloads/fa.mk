# Shared FlashAttention checkout and include paths for FA2 and FA3 workloads.

FA_MK := $(lastword $(MAKEFILE_LIST))

FLASH_ATTENTION_PACKAGE_DIR := third_party/flash-attention
FLASH_ATTENTION_DIR ?= $(FLASH_ATTENTION_PACKAGE_DIR)/checkout
FLASH_ATTENTION_PREPARE_SCRIPT := $(FLASH_ATTENTION_PACKAGE_DIR)/prepare.sh
FLASH_ATTENTION_PATCHES := \
	$(wildcard $(FLASH_ATTENTION_PACKAGE_DIR)/patches/*.patch)
FLASH_ATTENTION_PREPARED_STAMP := $(FLASH_ATTENTION_DIR)/.gpgpusim-prepared

FLASH_ATTENTION_FA2_INCLUDES = \
	-I$(FLASH_ATTENTION_DIR)/csrc/flash_attn/src \
	-I$(FLASH_ATTENTION_DIR)/csrc/cutlass/include
FLASH_ATTENTION_FA3_INCLUDES = \
	-I$(FLASH_ATTENTION_DIR)/hopper \
	-I$(FLASH_ATTENTION_DIR)/csrc/cutlass/include

# This file target is reached through FA2/FA3 object prerequisites. Normal
# builds trigger it automatically; it is not a standalone user workflow.
$(FLASH_ATTENTION_PREPARED_STAMP): \
$(FA_MK) $(FLASH_ATTENTION_PREPARE_SCRIPT) $(FLASH_ATTENTION_PATCHES)
	FLASH_ATTENTION_DIR="$(abspath $(FLASH_ATTENTION_DIR))" \
		$(FLASH_ATTENTION_PREPARE_SCRIPT)
	@touch $@
