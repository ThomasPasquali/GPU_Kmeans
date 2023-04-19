CC=g++
CFLAGS=-Wall -Wextra 

BIN_DIR=bin
SRC_DIR=src

ALL_TARGETS=main
HPP_TARGETS=input_parser.hpp point.hpp

$(BIN_DIR)/$(SRC_DIR)/main: $(SRC_DIR)/main.cpp $(addprefix include/,$(HPP_TARGETS))
	@mkdir -p $(BIN_DIR)/$(SRC_DIR)
	@$(CC) $(CFLAGS) $< -o $@

all: $(addprefix $(BIN_DIR)/$(SRC_DIR)/,$(ALL_TARGETS))

clean: 
	@rm -f $(BIN_DIR)/$(SRC_DIR)/*
