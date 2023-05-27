mkdir -p build
cd build
cmake ..
cmake --build . --target unit_kernels

./tests/bin/unit_kernels