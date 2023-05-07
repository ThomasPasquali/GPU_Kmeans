CC=g++
CFLAGS=-Wall -Wextra

CUDA_FLAGS=--compiler-options -Wall --compiler-options -Wextra 
CUDAC=nvcc

BIN_DIR=bin
SRC_DIR=src

ALL_TARGETS=main
CUDA_LIBS=utils.cu

$(BIN_DIR)/$(SRC_DIR)/main: $(SRC_DIR)/main.cu $(SRC_DIR)/kmeans.cu $(addprefix lib/cuda/,$(CUDA_LIBS))
	@ mkdir -p $(BIN_DIR)/$(SRC_DIR)
	@ $(CUDAC) $(CUDA_FLAGS) $^ -o $@

all: $(addprefix $(BIN_DIR)/$(SRC_DIR)/,$(ALL_TARGETS))

clean: 
	@ rm -f $(BIN_DIR)/$(SRC_DIR)/*
