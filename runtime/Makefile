# SPDX-License-Identifier: Apache-2.0
# Minimal cross-compile Makefile for libcompgen_runtime.a.
#
# The CMakeLists.txt in this directory is the canonical build for host
# use (and FireSim host tooling). This Makefile is a stripped-down
# alternative aimed at cross-compiling the runtime against embedded
# toolchains (Zephyr SDK's riscv64-zephyr-elf-gcc, bare-metal
# riscv64-unknown-elf-gcc) without pulling CMake into every bundle.
#
# Callers override CC/AR/CFLAGS; the CompGen Python emitter invokes
# this Makefile with the appropriate flags for the target it is
# generating a bundle for.
#
# Example:
#     make -C runtime CC=riscv64-zephyr-elf-gcc AR=riscv64-zephyr-elf-ar \
#         CFLAGS='-O2 -ffreestanding -fno-builtin -mabi=lp64d \
#                 -march=rv64imafdc_zicsr_zifencei -mcmodel=medany' \
#         OUT_DIR=/tmp/compgen_runtime_build
#
# Emits ``$(OUT_DIR)/libcompgen_runtime.a``.

CC      ?= cc
AR      ?= ar
CFLAGS  ?= -O2 -std=c17 -Wall -Wextra
INCDIR  := include
OUT_DIR ?= .

# Embedded slice of the runtime: arena + op primitives. The
# task-graph engine (src/engine.c etc.) targets hosted environments
# with pthreads and is built through CMakeLists.txt; embedded bundles
# only need the AOT-inference subset.
SRCS := src/arena.c src/ops.c
OBJS := $(addprefix $(OUT_DIR)/,$(SRCS:.c=.o))
TARGET := $(OUT_DIR)/libcompgen_runtime.a

all: $(TARGET)

$(TARGET): $(OBJS)
	$(AR) rcs $@ $^

$(OUT_DIR)/%.o: %.c $(INCDIR)/compgen/arena.h $(INCDIR)/compgen/ops.h $(INCDIR)/compgen/types.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

clean:
	rm -rf $(OUT_DIR)/src $(TARGET)

.PHONY: all clean
