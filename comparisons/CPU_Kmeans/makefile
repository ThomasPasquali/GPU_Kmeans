CC=g++
CFLAGS=-Wall -Wextra

BIN_DIR=bin
SRC_DIR=src

ALL_TARGETS=kmeans

$(BIN_DIR)/kmeans: $(SRC_DIR)/main.cpp $(SRC_DIR)/kmeans.cpp 
	@ mkdir -p $(BIN_DIR)
	@ $(CC) $(CFLAGS) $^ -o $@

all: $(addprefix $(BIN_DIR)/,$(ALL_TARGETS))

clean: 
	@ rm -f $(BIN_DIR)/$(SRC_DIR)/*

