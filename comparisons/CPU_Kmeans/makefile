CC=g++
CFLAGS=-Wall -Wextra

BIN_DIR=bin
SRC_DIR=src

INCLUDES_DIR=-I../../src/include

ALL_TARGETS=kmeans

all: $(addprefix $(BIN_DIR)/,$(ALL_TARGETS))

$(BIN_DIR)/kmeans: $(SRC_DIR)/main.cpp $(SRC_DIR)/kmeans.cpp
	@ mkdir -p $(BIN_DIR)
	@ $(CC) $(CFLAGS) $^ $(INCLUDES_DIR) -o $@

clean:
	@ rm -f $(BIN_DIR)/*

