#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include <random>
#include <chrono>

#define checkCudaErr(val) check( (val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

__device__ float fatomicMin(float *addr, float value) {

  float old = *addr, assumed;

  if(old <= value) return old;

  do {
    assumed = old;
    old = atomicCAS((unsigned int*)addr, __float_as_int(assumed), __float_as_int(value));
  } while(old != assumed);

  return old;

}

__host__ __device__ unsigned int next_pow_2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

struct Pair {
  float v;
  uint32_t i;
};


#define SHFL_MASK 0xFFFFFFFF
__device__ Pair shfl_xor_sync (Pair p, unsigned delta){
  return Pair{
    __shfl_xor_sync(SHFL_MASK, p.v, delta),
    __shfl_xor_sync(SHFL_MASK, p.i, delta),
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
  // printf("b: %u t: %u t.i: %u t.v: %.0f     t.i: %u t.v: %.0f\n", blockIdx.x, threadIdx.x, t.i, t.v, tsh.i, tsh.v);
  return t;
}

__global__ void clusters_argmin_shfl(const uint32_t n, const uint32_t k, float* d_distances, uint32_t* points_clusters, uint32_t warps_per_block) {
  __shared__ Pair shrd[1024];
  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid % warpSize;
  const uint32_t wid = tid / warpSize;
  const uint32_t idx = blockIdx.x * k + tid;
  float val = tid < k ? d_distances[idx] : 100000;

  /* if (tid == 0) {
    shrdVal = 100000;
  }
  __syncthreads(); */

  Pair p = warp_argmin(val);
  //if (blockIdx.x == 2) 
  // printf("b: %u t: %3u argmin: %u min: %.3f idx: %4u v: %8.0f\n", blockIdx.x, lane, p.i, p.v, idx, val);

  if (lane == 0) {
    /* if (p.v < shrdVal) {
      printf("b: %u t: %3u argmin: %u %u min: %.0f %.0f\n", blockIdx.x, tid, p.i, shrdIdx, p.v, shrdVal);
      shrdVal = p.v;
      shrdIdx = p.i + (32 * wid);
    } */
    // printf("b: %u wid: %u, i: %u v: %.0f\n", blockIdx.x, wid, p.i, p.v);
    p.i += 32 * wid;
    // printf("b: %u wid: %u, i: %u v: %.0f\n", blockIdx.x, wid, p.i, p.v);
    shrd[wid] = p;
  }
  
  __syncthreads();


  if (tid == 0) {
    Pair* tmp = shrd;
    float minV = tmp->v;
    uint32_t minI = tmp->i;
    for (uint32_t i = 1; i < warps_per_block; i++) {
      Pair* tmp = shrd + i;
      // printf("b: %u  %u, i: %u %u v: %.0f %.0f\n", blockIdx.x, i, minI, tmp->i, minV, tmp->v);
      if (tmp->v < minV) {
        minV = tmp->v;
        minI = tmp->i;
      }
    }
    
    points_clusters[blockIdx.x] = minI;
    // printf("b: %u, i: %u v: %.0f\n", blockIdx.x, minI, minV);
  }
}

int main(){
    const unsigned int n = 8;
    const unsigned int k = 1024;
    const unsigned int SIZE = n * k;
    float *h_distances = new float[SIZE];
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < k; ++j) {
          h_distances[i * k + j] = std::rand() % 100 + 2;
          // printf("%-2u %-2u -> %.0f\n", i, j, h_distances[i * k + j]);
        }
    std::cout << std::endl;
    std::chrono::steady_clock::time_point begin, end;
    /* float min_dist;
    unsigned int argmin_dist;
    begin = std::chrono::steady_clock::now();
    CPUGetMaxArgMax(SIZE, h_distances, min_dist, argmin_dist);
    end = std::chrono::steady_clock::now();
    float cpu_time = (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0;
    printf("CPU: Min (Argmin): %.0f (%u)\t time %.6fs\n", min_dist, argmin_dist, cpu_time); */
    
    float *d_distances;
    uint32_t* d_points_clusters;
    //cudaMallc occupies most of the time consuming!
    cudaMalloc((void **)&d_distances, sizeof(float) * SIZE);
    cudaMalloc((void **)&d_points_clusters, sizeof(uint32_t) * n);
    // cudaMalloc((void **)&d_index, sizeof(unsigned int) * grid_size);
    cudaMemcpy(d_distances, h_distances, sizeof(float) * SIZE,  cudaMemcpyHostToDevice);
    begin = std::chrono::steady_clock::now();    
    // GPUGetMaxArgMax(n, d_distances, d_index, min_dist, argmin_dist, block_size);
    uint32_t warps_per_block = (k + 32 - 1) / 32; // Ceil
    clusters_argmin_shfl<<<n, max(next_pow_2(k), 32)>>>(n, k, d_distances, d_points_clusters, warps_per_block);
    cudaDeviceSynchronize();
    end = std::chrono::steady_clock::now();

    float gpu_time = (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0;
    // printf("GPU: Min (Argmin): %.0f (%u)\t time %.6fs\n", min_dist, argmin_dist, cpu_time);

    uint32_t h_points_clusters[n];
    cudaMemcpy(h_points_clusters, d_points_clusters, sizeof(uint32_t) * n,  cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (uint32_t i = 0; i < n; i++) {
      float min = 100000;
      uint32_t idx = 0;
      for (uint32_t j = 0, ii = i * k; j < k; j++, ii++) {
        // printf("j: %u, ii: %u, v: %.0f\n", j, ii, h_distances[ii]);
        if (h_distances[ii] < min) {
          min = h_distances[ii];
          idx = j;
        }
      }
      printf("%-2u -> %-2u (should be %u   %.0f)\n", i, h_points_clusters[i], idx, min);
    }
    
}