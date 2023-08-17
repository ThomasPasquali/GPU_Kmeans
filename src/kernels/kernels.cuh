#ifndef __KERNEL_ARGMIN__
#define __KERNEL_ARGMIN__

#include <stdint.h>
#include <cublas_v2.h>

#include "../include/common.h"

#define DISTANCES_SHFL_MASK 0xFFFFFFFF
#define ARGMIN_SHFL_MASK    0xFFFFFFFF
#define CENTROIDS_SHFL_MASK 0xFFFFFFFF

struct Pair {
  float v;
  uint32_t i;
};

__global__ void compute_distances_one_point_per_warp(DATA_TYPE* distances, DATA_TYPE* centroids, DATA_TYPE* points, uint32_t next_pow_2);
__global__ void compute_distances_shmem(DATA_TYPE* distances, DATA_TYPE* centroids, DATA_TYPE* points, const uint32_t points_per_warp, const uint32_t d);
__global__ void compute_distances_shfl(DATA_TYPE* distances, DATA_TYPE* centroids, DATA_TYPE* points, const uint32_t points_n, const uint32_t points_per_warp, const uint32_t d, const uint32_t d_closest_2_pow);

/**
 * @brief Generates associated matrices for points. 
 * 
 * @param points row-major matrix
 * @param associated_matrices the function will store here the associated matrices
 * @param d 
 */
__global__ void compute_point_associated_matrices (const DATA_TYPE* points, DATA_TYPE* associated_matrices, const uint32_t d);
void compute_gemm_distances (cublasHandle_t& handle, uint32_t d1, uint32_t n, uint32_t k, DATA_TYPE* d_P, DATA_TYPE* d_C, DATA_TYPE* d_distances);

__global__ void clusters_argmin_shfl(const uint32_t n, const uint32_t k, float* d_distances, uint32_t* points_clusters,  uint32_t* clusters_len, uint32_t warps_per_block, DATA_TYPE infty);


__global__ void compute_centroids_shfl(DATA_TYPE* centroids, DATA_TYPE* points, uint32_t* points_clusters, uint32_t* clusters_len, uint64_t n, uint32_t d, uint32_t k, int round);
__global__ void compute_centroids_shfl_shrd(DATA_TYPE* centroids, DATA_TYPE* points, uint32_t* points_clusters, uint32_t* clusters_len, uint64_t n, uint32_t d, uint32_t k, int round);

#endif