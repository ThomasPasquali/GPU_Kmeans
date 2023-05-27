#ifndef __KERNEL_ARGMIN__
#define __KERNEL_ARGMIN__

#include <stdint.h>

#include "../include/common.h"

struct Pair {
  float v;
  uint32_t i;
};

__global__ void clusters_argmin_shfl(const uint32_t n, const uint32_t k, float* d_distances, uint32_t* points_clusters,  uint32_t* clusters_len, uint32_t warps_per_block, DATA_TYPE infty);

#endif