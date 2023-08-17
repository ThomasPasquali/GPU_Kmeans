#include "kernels.cuh"

__global__ void compute_centroids_shfl(DATA_TYPE* centroids, DATA_TYPE* points, uint32_t* points_clusters, uint32_t* clusters_len, uint64_t n, uint32_t d, uint32_t k, int round) {  
  uint32_t block_base = (warpSize * round);
  if (threadIdx.y >= d - block_base) { return; }
  
  uint32_t cluster_idx   = 2 * blockIdx.y * blockDim.x + threadIdx.x;
  uint32_t point_idx     = block_base + cluster_idx * d + threadIdx.y;
  uint32_t cluster_off   = blockDim.x;
  uint32_t point_off     = cluster_off * d;
  uint32_t centroids_idx = block_base + blockIdx.x * d + threadIdx.y;
  
  float val = 0;

  if (point_idx < n * d && blockIdx.x == points_clusters[cluster_idx]) { 
    val = points[point_idx]; 
  }
  
  if (point_idx + point_off < n * d && blockIdx.x == points_clusters[cluster_idx + cluster_off]) { 
    val += points[point_idx + point_off]; 
  } 
  
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(CENTROIDS_SHFL_MASK, val, offset);
  }
  
  if (threadIdx.x % warpSize == 0 && centroids_idx < k * d) {
    uint32_t count = clusters_len[blockIdx.x] > 1 ? clusters_len[blockIdx.x] : 1; 
    DATA_TYPE scale = 1.0 / ((double) count); 
    val *= scale;
    atomicAdd(&centroids[centroids_idx], val);
  }
}

__global__ void compute_centroids_shfl_shrd(DATA_TYPE* centroids, DATA_TYPE* points, uint32_t* points_clusters, uint32_t* clusters_len, uint64_t n, uint32_t d, uint32_t k, int round) {  
  uint32_t block_base = (warpSize * round);
  if (threadIdx.y >= d - block_base) { return; }

  uint32_t cluster_idx   = 2 * blockIdx.y * blockDim.x + threadIdx.x;
  uint32_t point_idx     = block_base + cluster_idx * d + threadIdx.y;
  uint32_t cluster_off   = blockDim.x;
  uint32_t point_off     = cluster_off * d;
  uint32_t centroids_idx = block_base + blockIdx.x * d + threadIdx.y;
  
  float val = 0;
  extern __shared__ DATA_TYPE shrd_mem[];

  if (point_idx < n * d && blockIdx.x == points_clusters[cluster_idx]) { 
    val = points[point_idx]; 
  }
  
  if (point_idx + point_off < n * d && blockIdx.x == points_clusters[cluster_idx + cluster_off]) { 
    val += points[point_idx + point_off]; 
  }
  
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(CENTROIDS_SHFL_MASK, val, offset);
  }
  
  if (threadIdx.x % warpSize == 0 && centroids_idx < k * d) {
    uint32_t warp_idx   = threadIdx.x / warpSize;
    uint32_t shrd_dim_y = blockDim.x  / warpSize;
    uint32_t shrd_idx   = threadIdx.y * shrd_dim_y + warp_idx;
    
    shrd_mem[shrd_idx] = val;
    __syncthreads();
    
    for (int offset = shrd_dim_y / 2; offset > 0; offset /= 2) {
      if (warp_idx < offset) {
        shrd_mem[shrd_idx] += shrd_mem[shrd_idx + offset];
      }
      __syncthreads();
    }

    if (shrd_idx % shrd_dim_y == 0) {
      uint32_t count = clusters_len[blockIdx.x] > 1 ? clusters_len[blockIdx.x] : 1; 
      DATA_TYPE scale = 1.0 / ((double) count); 
      val = shrd_mem[shrd_idx] * scale;   
      atomicAdd(&centroids[centroids_idx], val);
    }
  }
}