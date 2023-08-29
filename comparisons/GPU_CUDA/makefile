CC=g++
CFLAGS=-Wall -Wextra

SRC_DIR=./
BIN_DIR=$(SRC_DIR)bin/
KMCUDA_DIR=$(SRC_DIR)kmcuda/src/

INCLUDES_DIR=-I$(SRC_DIR)../../src/include -I$(KMCUDA_DIR)
LIB_DIR=-L$(BIN_DIR)

ALL_TARGETS=libKMCUDA.so kmcuda

all: $(addprefix $(BIN_DIR),$(ALL_TARGETS))

$(BIN_DIR)libKMCUDA.so:
	@ mkdir -p $(BIN_DIR)
	@ cd $(KMCUDA_DIR) && cmake -DCMAKE_BUILD_TYPE=Release -D DISABLE_PYTHON=y -D DISABLE_R=y -D CUDA_ARCH=52 . && make
	@ mv $(KMCUDA_DIR)libKMCUDA.so $(BIN_DIR)

$(BIN_DIR)kmcuda: kmcuda.cpp
	$(CC) $(CFLAGS) $^ $(INCLUDES_DIR) $(LIB_DIR) -l KMCUDA -Wl,-rpath,$(BIN_DIR) -o $@

clean:
	@ rm -f $(BIN_DIR)*