cmake_minimum_required(VERSION 3.22)

project(GPU_Kmeans VERSION 1.0.0)

enable_language(CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")

add_subdirectory(src)
add_subdirectory(tests EXCLUDE_FROM_ALL)