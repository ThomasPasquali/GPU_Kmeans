#include <catch2/catch_test_macros.hpp>

#include <limits>

#include "../src/kernels/argmin.cuh"
#include "../src/utils.cuh"
#include "../src/include/common.h"

// This function is not decalred in utils.cu because of compilation problems with __host__ __device__
__host__ __device__ unsigned int next_pow_2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

#define TESTS_N 8
const unsigned int N[TESTS_N] = {2, 10, 17, 51, 159, 1000, 3456, 10056};
const unsigned int K[TESTS_N] = {1,  2,  7,  5, 129,  997, 1023, 1024};
DATA_TYPE infty = numeric_limits<DATA_TYPE>::infinity();

TEST_CASE("kernel argmin", "[kernel][argmin]") {
  for (int i = 0; i < TESTS_N; ++i) {
    const unsigned int n = N[i];
    const unsigned int k = K[i];
    const unsigned int SIZE = n * k;
    char test_name[20];
    
    sprintf(test_name, "kernel argmin %u %u", n, k);

    SECTION(test_name) {
        
        DATA_TYPE *h_distances = new DATA_TYPE[SIZE];
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t j = 0; j < k; ++j) {
                h_distances[i * k + j] = static_cast <DATA_TYPE> (std::rand() / 100.0);
                // printf("%-2u %-2u -> %.0f\n", i, j, h_distances[i * k + j]);
            }
        }
        DATA_TYPE *d_distances;
        cudaMalloc((void **)&d_distances, sizeof(DATA_TYPE) * SIZE);
        cudaMemcpy(d_distances, h_distances, sizeof(DATA_TYPE) * SIZE,  cudaMemcpyHostToDevice);

        uint32_t* d_clusters_len;
        cudaMalloc(&d_clusters_len, k * sizeof(uint32_t));
        cudaMemset(d_clusters_len, 0, k * sizeof(uint32_t));

        uint32_t* d_points_clusters;
        cudaMalloc((void **)&d_points_clusters, sizeof(uint32_t) * n);
        
        uint32_t warps_per_block = (k + 32 - 1) / 32; // Ceil
        clusters_argmin_shfl<<<n, max(next_pow_2(k), 32)>>>(n, k, d_distances, d_points_clusters, d_clusters_len, warps_per_block, infty);
        cudaDeviceSynchronize();

        uint32_t h_points_clusters[n];
        cudaMemcpy(h_points_clusters, d_points_clusters, sizeof(uint32_t) * n,  cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        for (uint32_t i = 0; i < n; i++) {
          DATA_TYPE min = infty;
          uint32_t idx = 0;
          for (uint32_t j = 0, ii = i * k; j < k; j++, ii++) {
            // printf("j: %u, ii: %u, v: %.0f\n", j, ii, h_distances[ii]);
            if (h_distances[ii] < min) {
              min = h_distances[ii];
              idx = j;
            }
          }
          REQUIRE( h_points_clusters[i] == idx );
          //printf("%-7u -> %5u (should be %-5u %.3f)\n", i, h_points_clusters[i], idx, min);
        }
    }
  }
}