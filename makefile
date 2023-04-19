CC=g++
CFLAGS=-Wall -Wextra 

BIN_DIR=bin
SRC_DIR=src

$(BIN_DIR)/$(SRC_DIR)/main: $(SRC_DIR)/main.cpp
	@mkdir -p $(BIN_DIR)/$(SRC_DIR)
	@$(CC) $(CFLAGS) $< -o $@

all: $(addprefix $(BIN_DIR)/$(SRC_DIR)/,$(ALL_TARGETS))

clean: 
	@rm -f $(BIN_DIR)/$(SRC_DIR)/*
