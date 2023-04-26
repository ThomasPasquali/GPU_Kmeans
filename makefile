CC=g++
CFLAGS=-Wall -Wextra

CUDA_FLAGS=--compiler-options -Wall --compiler-options -Wextra 
CUDAC=nvcc

BIN_DIR=bin
SRC_DIR=src

ALL_TARGETS=main
HPP_TARGETS=input_parser.hpp point.hpp errors.hpp
CUDA_LIBS=utils.cu kmeans.cu

$(BIN_DIR)/$(SRC_DIR)/main: $(SRC_DIR)/main.cu $(addprefix lib/cuda/,$(CUDA_LIBS)) #$(addprefix include/,$(HPP_TARGETS))
	@ mkdir -p $(BIN_DIR)/$(SRC_DIR)
	@ $(CUDAC) $(CUDA_FLAGS) $^ -o $@

all: $(addprefix $(BIN_DIR)/$(SRC_DIR)/,$(ALL_TARGETS))

clean: 
	@ rm -f $(BIN_DIR)/$(SRC_DIR)/*
