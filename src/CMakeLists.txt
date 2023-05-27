add_executable(gpukmeans main.cu kmeans.cu kmeans.cuh utils.cu utils.cuh)
target_compile_options(gpukmeans PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wall -Wextra>) 