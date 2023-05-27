find_package(Catch2 3 REQUIRED)
# These tests can use the Catch2-provided main
add_executable(unit_kernels unit_kernels.cpp)
target_link_libraries(unit_kernels PRIVATE Catch2::Catch2WithMain)