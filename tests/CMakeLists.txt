find_package(Catch2 3 REQUIRED)

add_executable(
  unit_kernels 
  unit_kernels.cu
  ../src/utils.cu
  ../src/kernels/kernels.cuh
  ../src/kernels/argmin.cu 
  ../src/kernels/centroids.cu 
  ../src/kernels/distances.cu 
  )

target_link_libraries(unit_kernels PRIVATE -lcublas)
target_link_libraries(unit_kernels PRIVATE Catch2::Catch2WithMain)
# target_link_libraries(unit_kernels PRIVATE utils)