#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../src/kernels/kernels.cuh"
#include "../src/utils.cuh"
#include "../src/include/common.h"

#define TEST_DEBUG 0
#define WARP_SIZE  32

const DATA_TYPE infty   = numeric_limits<DATA_TYPE>::infinity();
const DATA_TYPE EPSILON = numeric_limits<DATA_TYPE>::epsilon();

void initRandomMatrix (DATA_TYPE* M, uint32_t rows, uint32_t cols) {
  for (uint32_t i = 0; i < rows; ++i) {
    for (uint32_t j = 0; j < cols; ++j) {
      //std::rand() / 100000005.32;
      M[IDX2C(i, j, rows)] = ((int)trunc(std::rand() / 100000005.32)) % 6;
    }
  }
}

/**
 * @brief 
 * 
 * @param C a matrix of size (d+1)x(d+1)
 * @param center a vector of size d
 * @param d number of dimensions
 * @param ki index of the center
 * @param ld number of rows of the matrix center
 */
void computeCentroidAssociatedMatrix (DATA_TYPE* C, DATA_TYPE* centers, uint32_t d, uint32_t ki, uint32_t ld) {
  ++d; // d = d + 1
  DATA_TYPE c;
  DATA_TYPE c_11 = 0;
  for (size_t i = 0; i < d - 1; ++i) { // Matrix borders
    c = centers[IDX2C(ki, i, ld)];
    C[i + 1] = -c;
    C[(i + 1) * d] = -c;
    c_11 += c * c;
  }
  C[0] = c_11;
  for (size_t i = 1; i < d; ++i) { // Matrix diagonal + fill with 0s 
    for (size_t j = 1; j < d; ++j) {
      C[i * d + j] = i == j ? 1 : 0;
    }
  }
}

TEST_CASE("kernel_distances_matrix", "[kernel][distances]") {
  const unsigned int TESTS_N = 8;
  const unsigned int N[TESTS_N] = {10, 10, 17, 30, 17,   15,  300, 1056};
  const unsigned int D[TESTS_N] = { 1,  2,  3, 11, 42, 1500,  500,  700};
  const unsigned int K[TESTS_N] = { 2,  6,  3, 11, 20,    5,   10,  506};

  for (int test_i = 6; test_i < 7; ++test_i) {
    printf("TTTTT %d\n", test_i);
    const unsigned int n = N[test_i];
    const unsigned int d = D[test_i];
    const unsigned int d1 = d + 1;
    const unsigned int k = K[test_i];

    char test_name[50];
    sprintf(test_name, "kernel compute_distances_matrix n: %u  d: %u  k: %u", n, d, k);
    SECTION(test_name) {

      const unsigned int m = n > d1 ? n : d1;
      DATA_TYPE *h_points = new DATA_TYPE[n * d1];
      DATA_TYPE *h_centroids = new DATA_TYPE[k * d];
      DATA_TYPE *h_distances = new DATA_TYPE[n];
      DATA_TYPE *h_res = new DATA_TYPE[m * m];

      // Constructing P
      initRandomMatrix(h_points, n, d1);
      for (size_t i = 0; i < n; ++i) {
        h_points[i] = 1;
      }
      if (TEST_DEBUG) {
        printf("\nPOINTS:\n");
        printMatrixColMaj(h_points, n, d1);
      }

      // Constructing C
      initRandomMatrix(h_centroids, k, d);
      if (TEST_DEBUG) {
        printf("\nCENTERS:\n");
        printMatrixColMaj(h_centroids, k, d);
      }

      cublasStatus_t stat;
      cublasHandle_t handle;
      DATA_TYPE* d_C;
      cudaMalloc(&d_C, d1 * d1 * sizeof(DATA_TYPE));
      DATA_TYPE* d_P;
      cudaMalloc(&d_P, n * d1 * sizeof(DATA_TYPE));
      DATA_TYPE* d_RES;
      cudaMalloc(&d_RES, m * m * sizeof(DATA_TYPE));

      DATA_TYPE* h_C = new DATA_TYPE[d1 * d1];
      for (uint32_t ki = 0; ki < k; ++ki) {
        // TODO test compute_point_associated_matrices (distances.cu)
        computeCentroidAssociatedMatrix(h_C, h_centroids, d, ki, k);
        if (TEST_DEBUG) {
          printf("\nCenter %u associated matrix:\n", ki);
          printMatrixColMaj(h_C, d1, d1);
        }

        // TODO test compute_gemm_distances (distances.cu)
        stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
          printf ("CUBLAS initialization failed\n");
          exit(EXIT_FAILURE);
        }
        stat = cublasSetMatrix (d1, d1, sizeof(DATA_TYPE), h_C, d1, d_C, d1);
        if (stat != CUBLAS_STATUS_SUCCESS) {
          printf ("data download failed");
          cublasDestroy(handle);
          exit(EXIT_FAILURE);
        }
        stat = cublasSetMatrix (n, d1, sizeof(DATA_TYPE), h_points, n, d_P, n);
        if (stat != CUBLAS_STATUS_SUCCESS) {
          printf ("data download failed");
          cublasDestroy(handle);
          exit(EXIT_FAILURE);
        }
        // Invoke the GEMM, ensuring k, lda, ldb, and ldc are all multiples of 8, 
        // and m is a multiple of 4:
        DATA_TYPE alpha = 1, beta = 0;
        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                          n, d1, d1, &alpha,
                          d_P, n,
                          d_C, d1,
                          &beta, d_RES, n);
        stat = cublasGetMatrix (n, d1, sizeof(DATA_TYPE), d_RES, n, h_res, n);
        if (stat != CUBLAS_STATUS_SUCCESS) {
          printf ("data upload failed %d", stat);
          cublasDestroy(handle);
          exit(EXIT_FAILURE);
        }
        if (TEST_DEBUG) {
          printf("\nRES:\n");
          printMatrixColMaj(h_res, n, d1);
        }

        stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                          n, n, d1, &alpha,
                          d_RES, n,
                          d_P, n,
                          &beta, d_RES, n);
        stat = cublasGetMatrix (n, n, sizeof(DATA_TYPE), d_RES, n, h_res, n);
        if (stat != CUBLAS_STATUS_SUCCESS) {
          printf ("data upload failed %d", stat);
          cublasDestroy(handle);
          exit(EXIT_FAILURE);
        }
        if (TEST_DEBUG) {
          printf("\nRES:\n");
          printMatrixColMaj(h_res, n, n);
        }

        for (size_t j = 0; j < n; ++j) {
          h_distances[j] = h_res[IDX2C(j, j, n)];
        }
        if (TEST_DEBUG) printf("\n\n");

        for (uint32_t ni = 0; ni < n; ++ni) {
          DATA_TYPE cpu_dist = 0, tmp;
          for (uint32_t di = 0; di < d; ++di) {
            // printf("DD: %d (%d) - %d   %f\n", IDX2C(ni, di + 1, n), n*d1, IDX2C(ki, di, k), cpu_dist);
            tmp = h_points[IDX2C(ni, di + 1, n)] - h_centroids[IDX2C(ki, di, k)];
            cpu_dist += tmp * tmp;
          }
          DATA_TYPE gpu_dist = h_distances[ni];
          // if (TEST_DEBUG) 
          printf("point: %u center: %u gpu(%.6f) cpu(%.6f)\n", ni, ki, gpu_dist, cpu_dist);
          REQUIRE( gpu_dist - cpu_dist < EPSILON );
        }
        if (TEST_DEBUG) printf("\n\n");
        cublasDestroy(handle);

      }

      delete h_C;
      delete h_points;
      delete h_centroids;
      delete h_distances;
      delete h_res;
      cudaFree(d_C);
      cudaFree(d_P);
      cudaFree(d_RES);

    }
  }
}

TEST_CASE("kernel_distances_warp", "[kernel][distances]") {
  const unsigned int TESTS_N = 8;
  const unsigned int N[TESTS_N] = {10, 10, 17, 51, 159, 1000, 3456, 10056};
  const unsigned int D[TESTS_N] = { 1,  2,  3,  5,  11,   12,   24,    32};
  const unsigned int K[TESTS_N] = { 2,  6,  3, 28,   7,  500, 1763,  9056};

  for (int i = 2; i < 3; ++i) { // FIXME
    const unsigned int n = N[i];
    const unsigned int d = D[i];
    const unsigned int k = K[i];

    char test_name[50];
    sprintf(test_name, "kernel compute_distances_shfl n: %u  d: %u  k: %u", n, d, k);
    SECTION(test_name) {

      DATA_TYPE *h_points = new DATA_TYPE[n * d];
      DATA_TYPE *h_centroids = new DATA_TYPE[k * d];
      DATA_TYPE *h_distances = new DATA_TYPE[n * k];
      for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < d; ++j) {
          h_points[i * d + j] = std::rand() / 10002.32;
          if (TEST_DEBUG) printf("%.3f, ", h_points[i * d + j]);
        }
        if (TEST_DEBUG) printf("\n");
      }
      if (TEST_DEBUG) printf("\n");
      for (uint32_t i = 0; i < k; ++i) {
        for (uint32_t j = 0; j < d; ++j) {
          h_centroids[i * d + j] = static_cast <DATA_TYPE> (std::rand() / 10002.45);
          if (TEST_DEBUG) printf("%.3f, ", h_points[i * d + j]);
        }
        if (TEST_DEBUG) printf("\n");
      }
      DATA_TYPE *d_distances;
      cudaMalloc(&d_distances, sizeof(DATA_TYPE) * n * k);
      DATA_TYPE *d_points;
      cudaMalloc(&d_points, sizeof(DATA_TYPE) * n * d);
      cudaMemcpy(d_points, h_points, sizeof(DATA_TYPE) * n * d, cudaMemcpyHostToDevice);
      DATA_TYPE *d_centroids;
      cudaMalloc(&d_centroids, sizeof(DATA_TYPE) * k * d);
      cudaMemcpy(d_centroids, h_centroids, sizeof(DATA_TYPE) * k * d, cudaMemcpyHostToDevice);

      const uint32_t dist_max_points_per_warp = WARP_SIZE / next_pow_2(d);
      dim3 dist_grid_dim(ceil(((float) n) / dist_max_points_per_warp), k);
      dim3 dist_block_dim(dist_max_points_per_warp * next_pow_2(d));

      compute_distances_shfl<<<dist_grid_dim, dist_block_dim>>>(d_distances, d_centroids, d_points, n, dist_max_points_per_warp, d, next_pow_2(d));
      cudaMemcpy(h_distances, d_distances, sizeof(DATA_TYPE) * n * k,  cudaMemcpyDeviceToHost);

      DATA_TYPE* cpu_distances = new DATA_TYPE[n * k];
      for (uint32_t ni = 0; ni < n; ++ni) {
        for (uint32_t ki = 0; ki < k; ++ki) {
          DATA_TYPE dist = 0, tmp;
          for (uint32_t di = 0; di < d; ++di) {
            tmp = h_points[ni * d + di] - h_centroids[ki * d + di];
            dist += tmp * tmp;
          }
          cpu_distances[ni * k + ki] = dist;
        }
      }

      cudaDeviceSynchronize();

      for (uint32_t i = 0; i < n * k; ++i) {
        if (TEST_DEBUG) printf("point: %u center: %u cmp: %.6f -- %.6f\n", i / k, i % k, h_distances[i], cpu_distances[i]);
        REQUIRE( h_distances[i] - cpu_distances[i] < EPSILON );
      }

      delete[] cpu_distances;
      cudaFree(d_distances);
      cudaFree(d_points);
      cudaFree(d_centroids);
    }
  }
}

TEST_CASE("kernel_argmin", "[kernel][argmin]") {
  const unsigned int TESTS_N = 8;
  const unsigned int N[TESTS_N] = {2, 10, 17, 51, 159, 1000, 3456, 10056};
  const unsigned int K[TESTS_N] = {1,  2,  7,  5, 129,  997, 1023, 1024};

  for (int i = 0; i < TESTS_N; ++i) {
    const unsigned int n = N[i];
    const unsigned int k = K[i];
    const unsigned int SIZE = n * k;

    char test_name[50];
    sprintf(test_name, "kernel clusters_argmin_shfl n: %u  k: %u", n, k);
    SECTION(test_name) {
        
      DATA_TYPE *h_distances = new DATA_TYPE[SIZE];
      for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < k; ++j) {
          h_distances[i * k + j] = static_cast <DATA_TYPE> (std::rand() / 105.456);
          if (TEST_DEBUG) { printf("%-2u %-2u -> %.0f\n", i, j, h_distances[i * k + j]); }
        }
      }
      DATA_TYPE *d_distances;
      cudaMalloc(&d_distances, sizeof(DATA_TYPE) * SIZE);
      cudaMemcpy(d_distances, h_distances, sizeof(DATA_TYPE) * SIZE,  cudaMemcpyHostToDevice);

      uint32_t* d_clusters_len;
      cudaMalloc(&d_clusters_len, k * sizeof(uint32_t));
      cudaMemset(d_clusters_len, 0, k * sizeof(uint32_t));

      uint32_t* d_points_clusters;
      cudaMalloc(&d_points_clusters, sizeof(uint32_t) * n);
      
      uint32_t warps_per_block = (k + WARP_SIZE - 1) / WARP_SIZE; // Ceil
      uint32_t sh_mem = warps_per_block * sizeof(Pair);
      clusters_argmin_shfl<<<n, max(next_pow_2(k), WARP_SIZE), sh_mem>>>(n, k, d_distances, d_points_clusters, d_clusters_len, warps_per_block, infty);
      cudaDeviceSynchronize();

      uint32_t h_points_clusters[n];
      cudaMemcpy(h_points_clusters, d_points_clusters, sizeof(uint32_t) * n,  cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      for (uint32_t i = 0; i < n; i++) {
        DATA_TYPE min = infty;
        uint32_t idx = 0;
        for (uint32_t j = 0, ii = i * k; j < k; j++, ii++) {
          if (TEST_DEBUG) { printf("j: %u, ii: %u, v: %.0f\n", j, ii, h_distances[ii]); }
          if (h_distances[ii] < min) {
            min = h_distances[ii];
            idx = j;
          }
        }
        
        REQUIRE( h_points_clusters[i] == idx );
        if (TEST_DEBUG) { printf("%-7u -> %5u (should be %-5u %.3f)\n", i, h_points_clusters[i], idx, min); }
      }
      cudaFree(d_distances);
      cudaFree(d_clusters_len);
      cudaFree(d_points_clusters);
    }
  }
}

TEST_CASE("kernel_centroids", "[kernel][centroids]") {
  #define TESTS_N 8
  const unsigned int D[TESTS_N] = {2,  3,  10,  32,  50,  100, 1000, 1024};
  const unsigned int N[TESTS_N] = {2, 10, 100,  51, 159, 1000, 3456, 10056};
  const unsigned int K[TESTS_N] = {1,  4,   7,  10, 129,  997, 1023, 1024};
  
  for (int d_idx = 0; d_idx < TESTS_N; ++d_idx) {
    for (int n_idx = 0; n_idx < TESTS_N; ++n_idx) {
      for (int k_idx = 0; k_idx < TESTS_N; ++k_idx) {
        const unsigned int d = D[d_idx];
        const unsigned int n = N[n_idx];
        const unsigned int k = K[k_idx];
        char test_name[50];
      
        snprintf(test_name, 49, "kernel centroids d=%u n=%u k=%u", d, n, k);

        SECTION(test_name) {
          DATA_TYPE *h_centroids = new DATA_TYPE[k * d];
          DATA_TYPE *h_points = new DATA_TYPE[n * d];
          uint32_t  *h_points_clusters = new uint32_t[n];
          uint32_t  *h_clusters_len = new uint32_t[k];
          
          memset(h_clusters_len, 0, k * sizeof(uint32_t));
          for (uint32_t i = 0; i < n; ++i) {
            h_points_clusters[i] = (static_cast <uint32_t> (std::rand() % k));
            h_clusters_len[h_points_clusters[i]]++;
            for (uint32_t j = 0; j < d; ++j) {
              h_points[i * d + j] = (static_cast <DATA_TYPE> (std::rand() / 1000.0)) / 1000.00;
            }
          }
          
          memset(h_centroids, 0, k * d * sizeof(DATA_TYPE));
          for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t j = 0; j < d; ++j) {
              h_centroids[h_points_clusters[i] * d + j] += h_points[i * d + j];
            }
          }

          for (uint32_t i = 0; i < k; ++i) {
            for (uint32_t j = 0; j < d; ++j) {
              uint64_t count = h_clusters_len[i] > 1 ? h_clusters_len[i] : 1; 
              DATA_TYPE scale = 1.0 / ((double) count); 
              h_centroids[i * d + j] *= scale; 
            }
          }  
          
          dim3 cent_grid_dim(k);
          dim3 cent_block_dim((((int) n) > WARP_SIZE) ? next_pow_2((n + 1) / 2) : WARP_SIZE, 
                              (((int) d) > WARP_SIZE) ? WARP_SIZE : d); 
          int cent_threads_tot = cent_block_dim.x * cent_block_dim.y;
          int rounds = ((d - 1) / WARP_SIZE) + 1;
          while (cent_threads_tot > 1024) {
            cent_block_dim.x /= 2;
            cent_grid_dim.y *= 2;
            cent_threads_tot = cent_block_dim.x * cent_block_dim.y;
          }

          DATA_TYPE* d_centroids;
          CHECK_CUDA_ERROR(cudaMalloc(&d_centroids, k * d * sizeof(DATA_TYPE)));
          DATA_TYPE* d_points;
          CHECK_CUDA_ERROR(cudaMalloc(&d_points, n * d * sizeof(DATA_TYPE)));
          CHECK_CUDA_ERROR(cudaMemcpy(d_points, h_points, n * d * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
          uint32_t* d_points_clusters;
          CHECK_CUDA_ERROR(cudaMalloc(&d_points_clusters, n * sizeof(uint32_t)));
          CHECK_CUDA_ERROR(cudaMemcpy(d_points_clusters, h_points_clusters, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
          uint32_t* d_clusters_len;
          CHECK_CUDA_ERROR(cudaMalloc(&d_clusters_len, k * sizeof(uint32_t)));
          CHECK_CUDA_ERROR(cudaMemcpy(d_clusters_len, h_clusters_len, k * sizeof(uint32_t), cudaMemcpyHostToDevice));
          
          for (int i = 0; i < rounds; i++) {
            compute_centroids_shfl<<<cent_grid_dim, cent_block_dim>>>(d_centroids, d_points, d_points_clusters, d_clusters_len, n, d, k, i); 
          }
          CHECK_CUDA_ERROR(cudaDeviceSynchronize());

          DATA_TYPE *h_centroids_cpy = new DATA_TYPE[k * d];
          CHECK_CUDA_ERROR(cudaMemcpy(h_centroids_cpy, d_centroids, k * d * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));

          const DATA_TYPE EPSILON = numeric_limits<DATA_TYPE>::round_error();
          bool is_equal = true;
          for (uint32_t i = 0; i < k; ++i) {
            for (uint32_t j = 0; j < d; ++j) {
              is_equal &= fabs(h_centroids[i * d + j] - h_centroids_cpy[i * d + j]) < EPSILON;
            }
          }  
          
          free(h_centroids);
          free(h_centroids_cpy);
          free(h_points);
          free(h_points_clusters);
          free(h_clusters_len);
          cudaFree(d_centroids);
          cudaFree(d_points);
          cudaFree(d_points_clusters);
          cudaFree(d_clusters_len);

          REQUIRE(is_equal);
        }
      }
    }
  }
}
