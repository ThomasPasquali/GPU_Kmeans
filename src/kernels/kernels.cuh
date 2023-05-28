#ifndef __KERNEL_ARGMIN__
#define __KERNEL_ARGMIN__

#include <stdint.h>

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

__global__ void clusters_argmin_shfl(const uint32_t n, const uint32_t k, float* d_distances, uint32_t* points_clusters,  uint32_t* clusters_len, uint32_t warps_per_block, DATA_TYPE infty);

__global__ void compute_centroids_shfl(DATA_TYPE* centroids, DATA_TYPE* points, uint32_t* points_clusters, uint32_t* clusters_len, uint64_t n, uint32_t d);
__global__ void compute_centroids_shfl_shrd(DATA_TYPE* centroids, DATA_TYPE* points, uint32_t* points_clusters, uint32_t* clusters_len, uint64_t n, uint32_t d);

#endif