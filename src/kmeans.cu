#include <stdio.h>
#include <vector>
#include <random>
#include <iomanip>
#include <cub/cub.cuh>
#include <cmath>

#include "../include/common.h"
#include "../include/colors.h"
#include "kmeans.cuh"
#include "../lib/cuda/utils.cuh"

using namespace std;

random_device rd;
seed_seq seed{0}; // FIXME use rd()
mt19937 rng(seed);

__host__ __device__ unsigned int next_pow_2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

#define SHFL_MASK 0xffffffff
/* Device kernels */
__global__ void compute_distances_one_point_per_warp(DATA_TYPE* distances, DATA_TYPE* centers, DATA_TYPE* points) {
  const uint64_t point_offset = blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t center_offset = blockIdx.y * blockDim.x + threadIdx.x;
  DATA_TYPE dist = points[point_offset] - centers[center_offset];
  dist *= dist;
  
  for (int i = next_pow_2(blockDim.x); i > 0; i /= 2)
    dist += __shfl_down_sync(SHFL_MASK, dist, i);

  if (threadIdx.x == 0) {
    distances[(blockIdx.x * gridDim.y) + blockIdx.y] = dist;
  }
}

__global__ void compute_distances_shmem(DATA_TYPE* distances, DATA_TYPE* centers, DATA_TYPE* points, const uint32_t points_per_warp, const uint32_t d) {
  const uint64_t point_i = (blockIdx.x * points_per_warp) + (threadIdx.x / d);
  const uint64_t center_i = blockIdx.y;
  const uint32_t d_i = threadIdx.x % d;
  const uint64_t dists_i = (center_i * blockDim.y * d) + ((point_i % points_per_warp) * d) + d_i;

  extern __shared__ DATA_TYPE dists[];

  if (threadIdx.x < points_per_warp * d) {
    DATA_TYPE dist = fabs(points[point_i * d + d_i] - centers[center_i * d + d_i]);
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

__global__ void compute_distances_shfl(DATA_TYPE* distances, DATA_TYPE* centers, DATA_TYPE* points, const uint32_t points_n, const uint32_t points_per_warp, const uint32_t d, const uint32_t d_closest_2_pow) {
  const uint64_t point_i = (blockIdx.x * points_per_warp) + (threadIdx.x / d_closest_2_pow);
  const uint64_t center_i = blockIdx.y;
  const uint32_t d_i = threadIdx.x % d_closest_2_pow;

  if (point_i < points_n && d_i < d) {
    DATA_TYPE dist = fabs(points[point_i * d + d_i] - centers[center_i * d + d_i]);
    dist *= dist;
    for (int i = d_closest_2_pow / 2; i > 0; i /= 2) {
      dist += __shfl_down_sync(SHFL_MASK, dist, i);
      // if (point_i == 3) printf("%d  p: %lu c: %lu d: %u v: %.3f\n", i, point_i, center_i, d_i, dist);
    }
    if (d_i == 0) {
      distances[(point_i * gridDim.y) + center_i] = dist;
    }
  }
}

__global__ void compute_centers(DATA_TYPE* centers, DATA_TYPE* points, uint32_t* points_clusters, uint64_t* clusters_len) {
  uint32_t point   = blockIdx.x;
  uint32_t cluster = points_clusters[point];
  uint32_t d       = blockDim.x;
  uint32_t d_i     = threadIdx.x;
  // extern __shared__ DATA_TYPE centers_shared[];
  
  // if (point >= clusters_len[cluster]) { return; }

  DATA_TYPE val = points[point * d + d_i];
  
  // if (cluster == 0 && d_i == 1) printf("cl: %u p: %u d: %u p_d: %.3f\n", cluster, point, d_i, val);
  
  // for (int i = next_pow_2(blockDim.x); i > 0; i /= 2) sum += __shfl_down_sync(SHFL_MASK, sum, i);
  // atomicAdd(centers_shared + cluster * d + d_i, val);
  // __syncthreads();
  
  //if (point == 0) {
  atomicAdd(centers + cluster * d + d_i, val);  
  // if (cluster == 0 && d_i == 1) printf("blk: %u %u %u, part_sum: %.3f\n", blockIdx.x, blockIdx.y, threadIdx.x, centers[cluster * d + d_i]);
  //}
}

/* Kmeans class */
void Kmeans::initCenters (Point<DATA_TYPE>** points) {
  uniform_int_distribution<int> random_int(0, n - 1);
  CHECK_CUDA_ERROR(cudaHostAlloc(&h_centers, CENTERS_BYTES, cudaHostAllocDefault));
  CHECK_CUDA_ERROR(cudaHostAlloc(&h_last_centers, CENTERS_BYTES, cudaHostAllocDefault));
  unsigned int i = 0;
  vector<Point<DATA_TYPE>*> usedPoints;
  Point<DATA_TYPE>* centers[k];
  while (i < k) {
    Point<DATA_TYPE>* p = points[random_int(rng)];
    bool found = false;
    for (auto p1 : usedPoints) {
      if ((*p1) == (*p)) { // FIXME Is it better use some min distance??
        found = true;
        break;
      }
    }
    if (!found) {
      for (unsigned int j = 0; j < d; ++j) {
        h_centers[i * d + j] = p->get(j);
      }
      centers[i] = new Point<DATA_TYPE>(p);
      usedPoints.push_back(p);
      ++i;
    }
  }
  if (DEBUG_INIT_CENTERS) { cout << endl << "Centers" << endl; for (i = 0; i < k; ++i) cout << *(centers[i]) << endl; }

  CHECK_CUDA_ERROR(cudaHostAlloc(&h_centers, CENTERS_BYTES, cudaHostAllocDefault));
  for (size_t i = 0; i < k; ++i) {
    for (size_t j = 0; j < d; ++j) {
      h_centers[i * d + j] = centers[i]->get(j);
    }
  }
  CHECK_CUDA_ERROR(cudaMalloc(&d_centers, CENTERS_BYTES));
}

Kmeans::Kmeans (size_t _n, unsigned int _d, unsigned int _k, Point<DATA_TYPE>** _points, cudaDeviceProp* _deviceProps)
    : n(_n), d(_d), k(_k),
    POINTS_BYTES(_n * _d * sizeof(DATA_TYPE)),
    CENTERS_BYTES(_k * _d * sizeof(DATA_TYPE)),
    points(_points),
    deviceProps(_deviceProps) {

  CHECK_CUDA_ERROR(cudaHostAlloc(&h_points, POINTS_BYTES, cudaHostAllocDefault));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < d; ++j) {
      h_points[i * d + j] = _points[i]->get(j);
    }
  }
  CHECK_CUDA_ERROR(cudaMalloc(&d_points, POINTS_BYTES));
  CHECK_CUDA_ERROR(cudaMemcpy(d_points, h_points, POINTS_BYTES, cudaMemcpyHostToDevice));

  initCenters(_points);
}

Kmeans::~Kmeans () {
  CHECK_CUDA_ERROR(cudaFreeHost(h_points));
  CHECK_CUDA_ERROR(cudaFreeHost(h_centers));
  CHECK_CUDA_ERROR(cudaFreeHost(h_last_centers));
  CHECK_CUDA_ERROR(cudaFreeHost(h_points_clusters));
  CHECK_CUDA_ERROR(cudaFree(d_centers));
  CHECK_CUDA_ERROR(cudaFree(d_points));
}

uint64_t Kmeans::run (uint64_t maxiter) {
  uint64_t converged = maxiter;

  /* INIT */
  DATA_TYPE* d_distances;
  CHECK_CUDA_ERROR(cudaMalloc(&d_distances, n * k * sizeof(DATA_TYPE)));
  uint32_t* d_points_clusters;
  CHECK_CUDA_ERROR(cudaMalloc(&d_points_clusters, n * sizeof(uint32_t)));
  CHECK_CUDA_ERROR(cudaMallocHost(&h_points_clusters, n * sizeof(uint32_t)));
  uint64_t* h_clusters_len;
  CHECK_CUDA_ERROR(cudaMallocHost(&h_clusters_len, k * sizeof(uint64_t)));
  uint64_t* d_clusters_len;
  CHECK_CUDA_ERROR(cudaMalloc(&d_clusters_len, k * sizeof(uint64_t)));

  uint64_t iter = 0;
  uint64_t max_cluster_len = 0;
  dim3 argmin_block_dim(k, d);

  #if COMPUTE_DISTANCES_KERNEL == 1
    const uint32_t dist_max_points_per_warp = deviceProps->warpSize / d;
    dim3 dist_grid_dim(ceil(((float) n) / dist_max_points_per_warp), k);
    dim3 dist_block_dim(dist_max_points_per_warp * d);
    uint32_t dist_kernel_sh_mem = k * dist_max_points_per_warp * d * sizeof(DATA_TYPE);
  #elif COMPUTE_DISTANCES_KERNEL == 2
    const uint32_t dist_max_points_per_warp = deviceProps->warpSize / next_pow_2(d);
    dim3 dist_grid_dim(ceil(((float) n) / dist_max_points_per_warp), k);
    dim3 dist_block_dim(dist_max_points_per_warp * next_pow_2(d));
    uint32_t dist_kernel_sh_mem = 0;
  #else
    dim3 dist_grid_dim(n, k);
    dim3 dist_block_dim(d);
    uint32_t dist_kernel_sh_mem = 0;
  #endif
  

  /* MAIN LOOP */
  while (iter++ < maxiter) {

    /* COMPUTE DISTANCES */
    CHECK_CUDA_ERROR(cudaMemcpy(d_centers, h_centers, CENTERS_BYTES, cudaMemcpyHostToDevice));
    if (DEBUG_KERNELS_INVOKATION) printf(YELLOW "[KERNEL]" RESET " compute_distances: Grid (%d, %d, %d), Block (%d, %d, %d), Sh.mem. %uB\n", dist_grid_dim.x, dist_grid_dim.y, dist_grid_dim.z, dist_block_dim.x, dist_block_dim.y, dist_block_dim.z, dist_kernel_sh_mem);
    #if PERFORMANCES_KERNEL_DISTANCES
      cudaEvent_t e_perf_dist_start, e_perf_dist_stop;
      cudaEventCreate(&e_perf_dist_start);
      cudaEventCreate(&e_perf_dist_stop);
      cudaEventRecord(e_perf_dist_start);
    #endif
    #if COMPUTE_DISTANCES_KERNEL == 1
      compute_distances_shmem<<<dist_grid_dim, dist_block_dim, dist_kernel_sh_mem>>>(d_distances, d_centers, d_points, dist_max_points_per_warp, d);
    #elif COMPUTE_DISTANCES_KERNEL == 2     
      compute_distances_shfl<<<dist_grid_dim, dist_block_dim>>>(d_distances, d_centers, d_points, n, dist_max_points_per_warp, d, next_pow_2(d));
    #else
      compute_distances_one_point_per_warp<<<dist_grid_dim, dist_block_dim>>>(d_distances, d_centers, d_points);
    #endif
    CHECK_LAST_CUDA_ERROR();
    #if PERFORMANCES_KERNEL_DISTANCES
      cudaEventRecord(e_perf_dist_stop);
      cudaEventSynchronize(e_perf_dist_stop);
      float e_perf_dist_ms = 0;
      cudaEventElapsedTime(&e_perf_dist_ms, e_perf_dist_start, e_perf_dist_stop);
      printf(CYAN "[PERFORMANCE]" RESET " compute_distances time: %.6f\n", e_perf_dist_ms / 1000);
      cudaEventDestroy(e_perf_dist_start);
      cudaEventDestroy(e_perf_dist_stop);
    #endif

    #if DEBUG_KERNEL_DISTANCES
      printf(GREEN "[DEBUG_KERNEL_DISTANCES]\n");
      DATA_TYPE tmp[n * k];
      CHECK_CUDA_ERROR(cudaMemcpy(tmp, d_distances, n * k * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
      cudaDeviceSynchronize();
      for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = 0; j < k; ++j)
          printf("%u %u -> %.3f\n", i, j, tmp[i * k + j]);
      cout << RESET << endl;
    #endif


    /* ASSIGN POINTS TO NEW CLUSTERS */
    #if DEBUG_KERNEL_ARGMIN
      printf("DEBUG_KERNEL_ARGMIN\n");
    #endif
    memset(h_clusters_len, 0, k * sizeof(uint64_t));
    for (size_t i = 0; i < n; i++) {
      cub::KeyValuePair<int32_t, DATA_TYPE> *d_argmin = NULL;
      CHECK_CUDA_ERROR(cudaMalloc(&d_argmin, sizeof(int32_t) + sizeof(DATA_TYPE)));
      // Allocate temporary storage
      void *d_temp_storage = NULL; size_t temp_storage_bytes = 0;
      cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_distances, d_argmin, k);
      CHECK_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
      
      // Run argmin-reduction
      cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_distances + i * k, d_argmin, k);

      int32_t argmin_idx;
      DATA_TYPE argmin_val;
      CHECK_CUDA_ERROR(cudaMemcpy(&argmin_idx, &(d_argmin->key), sizeof(int32_t), cudaMemcpyDeviceToHost));
      CHECK_CUDA_ERROR(cudaMemcpy(&argmin_val, &(d_argmin->value), sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
      CHECK_CUDA_ERROR(cudaFree(d_temp_storage));
      CHECK_CUDA_ERROR(cudaFree(d_argmin));
      CHECK_CUDA_ERROR(cudaDeviceSynchronize());

      #if DEBUG_KERNEL_ARGMIN
        printf("Argmin point %lu: %d %.3f\n", i, argmin_idx, argmin_val);
      #endif

      ++h_clusters_len[argmin_idx];
      max_cluster_len = max_cluster_len > h_clusters_len[argmin_idx] ? max_cluster_len : h_clusters_len[argmin_idx];
      h_points_clusters[i] = argmin_idx;
    }
    #if DEBUG_KERNEL_ARGMIN
      printf("\n");
    #endif
    CHECK_CUDA_ERROR(cudaMemcpy(d_points_clusters, h_points_clusters, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_clusters_len, h_clusters_len, k * sizeof(uint64_t), cudaMemcpyHostToDevice));
    

    /* COMPUTE NEW CENTERS */
    cudaMemset(d_centers, 0, k * d * sizeof(DATA_TYPE));
    dim3 centers_grid_dim(n);
    if (DEBUG_KERNELS_INVOKATION) printf(YELLOW "[KERNEL]" RESET "compute_centers: Grid (%d, %d, %d), Block (%d, %d, %d)\n", centers_grid_dim.x, centers_grid_dim.y, centers_grid_dim.z, d, 1, 1);
    compute_centers<<<centers_grid_dim, d/*, k * d * sizeof(DATA_TYPE) */>>>(d_centers, d_points, d_points_clusters, d_clusters_len);
    CHECK_LAST_CUDA_ERROR();

    CHECK_CUDA_ERROR(cudaMemcpy(h_centers, d_centers, d * k * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < k; ++i)
        for (uint32_t j = 0; j < d; ++j)
          h_centers[i * d + j] /= h_clusters_len[i];
    

    #if DEBUG_KERNEL_CENTERS
      printf("DEBUG_KERNEL_CENTERS\n");
      cout << endl << "CENTERS" << endl;
      for (uint32_t i = 0; i < k; ++i) {
        for (uint32_t j = 0; j < d; ++j)
          printf("%.3f, ", h_centers[i * d + j]);
        cout << endl;
      }
      cout << endl;
    #endif

    /* CHECK IF CONVERGED */
    if (iter > 1 && cmpCenters()) { // Exit
      converged = iter;
      break;
    } else { // Copy centers
      memcpy(h_last_centers, h_centers, CENTERS_BYTES);
    }
  }
  /* MAIN LOOP END */

  /* COPY BACK RESULTS*/
  for (size_t i = 0; i < n; i++) {
    points[i]->setCluster(h_points_clusters[i]);
  }
  

  /* FREE MEMORY */
  CHECK_CUDA_ERROR(cudaMalloc(&d_clusters_len, k * sizeof(uint64_t)));
  CHECK_CUDA_ERROR(cudaFree(d_distances));
  CHECK_CUDA_ERROR(cudaFree(d_points_clusters));
  CHECK_CUDA_ERROR(cudaFree(d_clusters_len));
  CHECK_CUDA_ERROR(cudaFreeHost(h_clusters_len));

  return converged;
}

bool Kmeans::cmpCenters () {
  const DATA_TYPE EPSILON = numeric_limits<DATA_TYPE>::epsilon();
  for (uint32_t i = 0; i < k; ++i)
        for (uint32_t j = 0; j < d; ++j)
          if (fabs(h_centers[i * d + j] - h_last_centers[i * d + j]) >= EPSILON)
            return false;
  return true;
  // return memcmp(h_centers, h_last_centers, CENTERS_BYTES);
}

void Kmeans::to_csv(ostream& o, char separator) {
  o << "cluster" << separator;
  for (size_t i = 0; i < d; ++i) {
    o << "d" << i;
    if (i != (d - 1)) o << separator;
  }
  o << endl;
  for (size_t i = 0; i < n; ++i) {
    o << h_points_clusters[i] << separator;
    for (size_t j = 0; j < d; ++j) {
      o << setprecision(8) << h_points[i * d + j];
      if (j != (d - 1)) o << separator;
    }
    o << endl;
  }
}