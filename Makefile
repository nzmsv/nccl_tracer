.PHONY: all

BPFTOOL := $(shell which bpftool)
LIBBPF_INC := /usr/local/include
LIBBPF_LIB := /usr/local/lib
KERNEL_HEADER_DIR := /usr/src/linux-headers-$(shell uname -r)

all: nccl_trace

nccl_trace.bpf.o: nccl_trace.bpf.c
	clang -g -O2 -target bpf -D__TARGET_ARCH_x86 -D__EXPORTED_HEADERS__ \
		-I. \
		-I$(KERNEL_HEADER_DIR)/arch/x86/include/generated/uapi \
		-I$(KERNEL_HEADER_DIR)/arch/x86/include/generated \
		-I$(KERNEL_HEADER_DIR)/arch/x86/include/uapi \
		-I$(KERNEL_HEADER_DIR)/include/uapi \
		-I$(KERNEL_HEADER_DIR)/include \
		-I$(LIBBPF_INC) \
		-c $< -o $@

nccl_trace.skel.h: nccl_trace.bpf.o
	$(BPFTOOL) gen skeleton $< > $@

nccl_trace: nccl_trace.cc nccl_trace.skel.h
	clang++ -g -O2 -D__TARGET_ARCH_x86 -D__EXPORTED_HEADERS__ \
		-I. \
		-I$(KERNEL_HEADER_DIR)/arch/x86/include/generated/uapi \
		-I$(KERNEL_HEADER_DIR)/arch/x86/include/uapi \
		-I$(LIBBPF_INC) \
		-o $@ $< -L$(LIBBPF_LIB) -lbpf -lelf -lz
