mkdir -p build
cd build
cmake ..
cmake --build . --target unit_kernels

./tests/bin/unit_kernels -v high 
# kernel_distances
# -s -d yes