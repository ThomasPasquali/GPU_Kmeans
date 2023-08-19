mkdir -p build
cd build
cmake ..
cmake --build . --target unit_kernels

# ./tests/bin/unit_kernels -v high kernel_distances_matrix --rng-seed 1234
./tests/bin/unit_kernels -v high kernel_distances_warp --rng-seed 1234
# -s -d yes