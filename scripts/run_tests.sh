#!/bin/bash

mkdir -p build
cd build
cmake ..
cmake --build . --target unit_kernels

if [ $? -eq 0 ]
then
  # ./tests/bin/unit_kernels -v high kernel_distances_matrix # --rng-seed 12345
  # ./tests/bin/unit_kernels -v high kernel_distances_warp # --rng-seed 1234
  # ./tests/bin/unit_kernels -v high kernel_centroids # --rng-seed 1234
  # ./tests/bin/unit_kernels -v high kernel_argmin # --rng-seed 1234
  ./tests/bin/unit_kernels -v high
  # -s -d yes
fi
