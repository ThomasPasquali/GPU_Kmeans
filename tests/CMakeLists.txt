find_package(Catch2 3 REQUIRED)

add_executable(
  unit_kernels 
  unit_kernels.cu
  ../src/kernels/argmin.cu 
  ../src/kernels/argmin.cuh
  )

target_link_libraries(unit_kernels PRIVATE Catch2::Catch2WithMain)
target_link_libraries(unit_kernels PRIVATE utils)