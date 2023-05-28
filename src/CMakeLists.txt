add_executable(
  gpukmeans
  ./kernels/argmin.cu 
  ./kernels/argmin.cuh
  ./kernels/distances.cu 
  ./kernels/distances.cuh
  main.cu 
  kmeans.cu 
  kmeans.cuh 
  )
target_link_libraries(gpukmeans PRIVATE utils)
target_compile_options(gpukmeans PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wall -Wextra>)