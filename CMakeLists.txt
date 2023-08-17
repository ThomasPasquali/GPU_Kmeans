cmake_minimum_required(VERSION 3.22)

project(GPU_Kmeans VERSION 1.0.0 LANGUAGES CXX CUDA)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_library(utils STATIC src/utils.cu src/utils.cuh)
set_target_properties(utils PROPERTIES CUDA_ARCHITECTURES "52") # this is to make CMake happy
set_target_properties(utils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(utils PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")

add_subdirectory(src)
add_subdirectory(tests EXCLUDE_FROM_ALL)