add_executable(
  gpukmeans
  ./kernels/kernels.cuh
  ./kernels/argmin.cu
  ./kernels/centroids.cu 
  main.cu 
  kmeans.cu 
  kmeans.cuh 
  )
target_link_libraries(gpukmeans PRIVATE utils)
target_compile_options(gpukmeans PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wall -Wextra>)