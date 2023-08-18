#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

#include "kernels.cuh"
#include "../utils.cuh"

#define DEBUG_GEMM 0

/*** Warp oriented ***/

__global__ void compute_distances_one_point_per_warp(DATA_TYPE* distances, DATA_TYPE* centroids, DATA_TYPE* points, uint32_t next_pow_2) {
  const uint64_t point_offset = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t center_offset = blockIdx.y * blockDim.x + threadIdx.x;
  DATA_TYPE dist = points[point_offset] - centroids[center_offset];
  dist *= dist;
  
  for (int i = next_pow_2; i > 0; i /= 2)
    dist += __shfl_down_sync(DISTANCES_SHFL_MASK, dist, i);

  if (threadIdx.x == 0) {
    distances[(blockIdx.x * gridDim.y) + blockIdx.y] = dist;
  }
}

__global__ void compute_distances_shmem(DATA_TYPE* distances, DATA_TYPE* centroids, DATA_TYPE* points, const uint32_t points_per_warp, const uint32_t d) {
  const uint64_t point_i = (blockIdx.x * points_per_warp) + (threadIdx.x / d);
  const uint64_t center_i = blockIdx.y;
  const uint32_t d_i = threadIdx.x % d;
  const uint64_t dists_i = (center_i * blockDim.y * d) + ((point_i % points_per_warp) * d) + d_i;

  extern __shared__ DATA_TYPE dists[];

  if (threadIdx.x < points_per_warp * d) {
    DATA_TYPE dist = fabs(points[point_i * d + d_i] - centroids[center_i * d + d_i]);
    dists[dists_i] = dist * dist;
    __syncthreads();
    if (d_i == 0) {
      for (int i = 1; i < d; i++) {
        dists[dists_i] += dists[dists_i + i];
      }
      distances[(point_i * center_i) + point_i] = dists[dists_i];
    }
  }
}

__global__ void compute_distances_shfl(DATA_TYPE* distances, DATA_TYPE* centroids, DATA_TYPE* points, const uint32_t points_n, const uint32_t points_per_warp, const uint32_t d, const uint32_t d_closest_2_pow) {
  const uint32_t point_i = (blockIdx.x * points_per_warp) + (threadIdx.x / d_closest_2_pow);
  const uint32_t center_i = blockIdx.y;
  const uint32_t d_i = threadIdx.x % d_closest_2_pow;

  if (point_i < points_n && d_i < d) {
    DATA_TYPE dist = fabs(points[point_i * d + d_i] - centroids[center_i * d + d_i]);
    dist *= dist;
    for (int i = d_closest_2_pow / 2; i > 0; i /= 2) {
      dist += __shfl_down_sync(DISTANCES_SHFL_MASK, dist, i);
      // if (point_i == 3) printf("%d  p: %lu c: %lu d: %u v: %.3f\n", i, point_i, center_i, d_i, dist);
    }
    if (d_i == 0) {
      distances[(point_i * gridDim.y) + center_i] = dist;
    }
  }
}

/*** END Warp oriented ***/

/*** Matrix multiplication ***/
/**
 * NOTICE: the reduction limits the maximum block size to 32 (warpSize) 
*/
__global__ void compute_point_associated_matrices (const DATA_TYPE* points, DATA_TYPE* associated_matrices, const uint32_t d, const uint32_t round) {
  const uint32_t block_base = warpSize * round;
  const uint32_t p_i = blockIdx.x;
  const uint32_t d_i = block_base + threadIdx.x;
  const uint32_t d_i1 = d_i + 1;

  // If dim in the thread is greater than d, then return to avoid illegal writes
  if (d_i >= d) { return; } 

  DATA_TYPE c = points[p_i * d + d_i];
  DATA_TYPE c_11 = c * c;

  for (int i = warpSize / 2; i > 0; i /= 2) { // Reduce c_11
    c_11 += __shfl_down_sync(DISTANCES_SHFL_MASK, c_11, i);
  }

  const uint32_t d1 = d + 1;
  const uint32_t matrix_base_i = p_i * d1 * d1;
  if (threadIdx.x == 0) {
    atomicAdd(&associated_matrices[matrix_base_i], c_11); // Write reduced c_11
  }
  associated_matrices[matrix_base_i + d_i1] = -c;               // Write first column
  associated_matrices[matrix_base_i + (d_i1 * d1)] = -c;        // Write first row
  associated_matrices[matrix_base_i + (d_i1 * d1) + d_i1] = 1;  // Write diagonal
}

DATA_TYPE* d_tmp = NULL; // https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm
/**
 * @brief Computes and writes to d_distances TODO
 * 
 * @param handle 
 * @param d1 
 * @param n 
 * @param k 
 * @param d_P the points associated matrices
 * @param d_C the matrix of centers (prefixed with 1s)
 * @param d_distances 
 */
void compute_gemm_distances (cublasHandle_t& handle, uint32_t d1, uint32_t n, uint32_t k, DATA_TYPE* d_P, DATA_TYPE* d_C, DATA_TYPE* d_distances) {
  DATA_TYPE alpha = (DATA_TYPE)1;
  DATA_TYPE beta = (DATA_TYPE)0;
  uint32_t d1d1 = d1 * d1;
  DATA_TYPE* P = d_P;
  uint32_t max_k_d1 = max(k, d1);
  DATA_TYPE h_distances[k * n];
  DATA_TYPE h_tmp[max_k_d1 * max_k_d1];
  if (d_tmp == NULL) {
    cudaMalloc(&d_tmp, max_k_d1 * max_k_d1 * sizeof(DATA_TYPE));
  }

  for (uint32_t p_i = 0; p_i < n; ++p_i, P += d1d1) { // Iterate over points associated matrices
    #if DEBUG_GEMM
      printf("\nc\n");
      DATA_TYPE tmp_debug1[n * d1];
      CHECK_CUBLAS_ERROR(cublasGetMatrix(k, d1, sizeof(DATA_TYPE), d_C, k, tmp_debug1, k));
      printMatrixColMaj(tmp_debug1, k, d1);
      printf("\nP_%d associated matrix\n", p_i);
      DATA_TYPE tmp_debug[d1d1];
      CHECK_CUBLAS_ERROR(cublasGetMatrix(d1, d1, sizeof(DATA_TYPE), P, d1, tmp_debug, d1));
      printMatrixColMaj(tmp_debug, d1, d1);
      printf("\n");
    #endif
    
    CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, // c * P
                                    k, d1, d1, &alpha,
                                    d_C, k,
                                    P, d1,
                                    &beta, d_tmp, k));

    #if DEBUG_GEMM
      printf("\nc * P\n");
      DATA_TYPE tmp_debug2[k * d1];
      CHECK_CUBLAS_ERROR(cublasGetMatrix(k, d1, sizeof(DATA_TYPE), d_tmp, k, tmp_debug2, k));
      printMatrixColMaj(tmp_debug2, k, d1);
      printf("\n");
    #endif

    CHECK_CUBLAS_ERROR(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, // (c * P) * c^T
                                    k, k, d1, &alpha,
                                    d_tmp, k,
                                    d_C, k,
                                    &beta, d_tmp, k));
    
    
    for (size_t i = 0; i < k; i++) {
      CHECK_CUBLAS_ERROR(cublasGetMatrix(k, k, sizeof(DATA_TYPE), d_tmp, k, h_tmp, k));
      h_distances[p_i * k + i] = h_tmp[IDX2C(i, i, k)];
    }

    #if DEBUG_GEMM
      printf("Distances from P_%d\n", p_i);
      printMatrixColMaj(h_tmp, k, k);
      printf("\n----------\n");
    #endif
  }
  // Copy distances to GPU
  CHECK_CUDA_ERROR(cudaMemcpy(d_distances, h_distances, n * k * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
}

/*** END Matrix multiplication ***/