#include <catch2/catch_test_macros.hpp>

#include "../src/kernels/argmin.cuh"
#include "../src/utils.cuh"
#include "../src/include/common.h"

__host__ __device__ unsigned int next_pow_2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

TEST_CASE("kernel argmin", "[kernel][argmin]" ) { // TODO test with different walues of k and n
    const unsigned int n = 500;
    const unsigned int k = 1024;
    const unsigned int SIZE = n * k;
    float *h_distances = new float[SIZE];
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < k; ++j) {
            h_distances[i * k + j] = std::rand() % 100 + 2;
            // printf("%-2u %-2u -> %.0f\n", i, j, h_distances[i * k + j]);
        }
    }
    float *d_distances;
    cudaMalloc((void **)&d_distances, sizeof(float) * SIZE);
    cudaMemcpy(d_distances, h_distances, sizeof(float) * SIZE,  cudaMemcpyHostToDevice);

    uint32_t* d_clusters_len;
    cudaMalloc(&d_clusters_len, k * sizeof(uint32_t));
    cudaMemset(d_clusters_len, 0, k * sizeof(uint32_t));

    uint32_t* d_points_clusters;
    cudaMalloc((void **)&d_points_clusters, sizeof(uint32_t) * n);
    
    uint32_t warps_per_block = (k + 32 - 1) / 32; // Ceil
    clusters_argmin_shfl<<<n, max(next_pow_2(k), 32)>>>(n, k, d_distances, d_points_clusters, d_clusters_len, warps_per_block, 1e9);
    cudaDeviceSynchronize();

    uint32_t h_points_clusters[n];
    cudaMemcpy(h_points_clusters, d_points_clusters, sizeof(uint32_t) * n,  cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (uint32_t i = 0; i < n; i++) {
      float min = 1e9;
      uint32_t idx = 0;
      for (uint32_t j = 0, ii = i * k; j < k; j++, ii++) {
        // printf("j: %u, ii: %u, v: %.0f\n", j, ii, h_distances[ii]);
        if (h_distances[ii] < min) {
          min = h_distances[ii];
          idx = j;
        }
      }
      REQUIRE( h_points_clusters[i] == idx );
      printf("%-2u -> %-2u (should be %u   %.0f)\n", i, h_points_clusters[i], idx, min);
    }
}