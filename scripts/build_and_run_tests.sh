mkdir -p build
cd build
cmake ..
cmake --build . --target unit_kernels

./tests/bin/unit_kernels -v high
# -s -d yes