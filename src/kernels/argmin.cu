#include "kernels.cuh"

__device__ Pair shfl_xor_sync (Pair p, unsigned delta){
  return Pair{
    __shfl_xor_sync(ARGMIN_SHFL_MASK, p.v, delta),
    __shfl_xor_sync(ARGMIN_SHFL_MASK, p.i, delta),
  };
}

__device__ Pair argmin (Pair a, Pair b) {
  return a.v <= b.v ? a : b;
}

__device__ Pair warp_argmin (float a) {
  Pair t{a, (uint32_t)threadIdx.x & 31};

  t = argmin(t, shfl_xor_sync(t, 1));
  t = argmin(t, shfl_xor_sync(t, 2));
  t = argmin(t, shfl_xor_sync(t, 4));
  t = argmin(t, shfl_xor_sync(t, 8));
  t = argmin(t, shfl_xor_sync(t, 16));
  return t;
}

__global__ void clusters_argmin_shfl(const uint32_t n, const uint32_t k, float* d_distances, uint32_t* points_clusters,  uint32_t* clusters_len, uint32_t warps_per_block, DATA_TYPE infty) {
  extern __shared__ Pair shrd[];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wid = tid / warpSize;
  const uint32_t idx = blockIdx.x * k + tid;
  float val = tid < k ? d_distances[idx] : infty;

  Pair p = warp_argmin(val);

  if (lane == 0) {
    p.i += 32 * wid; // Remap p.i
    shrd[wid] = p;
  }
  
  __syncthreads();


  if (tid == 0) { // Intra-block reduction
    Pair* tmp = shrd;
    float minV = tmp->v;
    uint32_t minI = tmp->i;
    for (uint32_t i = 1; i < warps_per_block; i++) {
      Pair* tmp = shrd + i;
      if (tmp->v < minV) {
        minV = tmp->v;
        minI = tmp->i;
      }
    }
    points_clusters[blockIdx.x] = minI;
    atomicAdd(&clusters_len[minI], 1); // TODO optimization (if possible...)
  }
}