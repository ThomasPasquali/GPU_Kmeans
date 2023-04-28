#include <stdio.h>
#include <vector>
#include <random>
#include <iomanip>
#include <cub/cub.cuh>
// #include <cub/device/device_reduce.cuh>

#include "../../include/common.h"
#include "kmeans.h"
#include "utils.cuh"

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

// #define SHFL_MASK 0xffffffff
#define SHFL_MASK 0xffffffff
/* Device kernels */
__global__ void compute_distances(DATA_TYPE* distances, DATA_TYPE* centers, DATA_TYPE* points) {
  uint64_t point_offset = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t center_offset = blockIdx.y * blockDim.x + threadIdx.x;
  DATA_TYPE dist = points[point_offset] - centers[center_offset];
  dist *= dist;

  // if (blockIdx.x == 0 && blockIdx.y == 0) printf("p: %2lu, d: %-5d p_j: %.3f c_j: %.3f dist2: %.3f\n", point_offset - threadIdx.x, threadIdx.x, points[point_offset], centers[center_offset], dist);

  // [].reduce((v,n)=>v+n,0)
  for (int i = next_pow_2(blockDim.x); i > 0; i /= 2)
    dist += __shfl_down_sync(SHFL_MASK, dist, i);

  // if (blockIdx.x == 0 && blockIdx.y == 0)// && threadIdx.x == 0) printf("blk: %d %d %d, dist: %.3f, warpSize: %d\n", blockIdx.x, blockIdx.y, threadIdx.x, dist, blockDim.x);

  distances[(blockIdx.x * gridDim.y) + blockIdx.y] = dist;
}

/*
extern __shared__ DATA_TYPE* aggr_distances;
uint64_t offset = (blockIdx.x * gridDim.y * blockDim.x) + (blockIdx.y * blockDim.x) + threadIdx.x;
DATA_TYPE dist = distances[offset];
if (threadIdx.x >= blockDim.x - 1) {
  dist += __shfl_up_sync(SHFL_MASK, dist, 1);
} else if (threadIdx.x > 0) {
  dist += __shfl_up_sync(SHFL_MASK, dist, 1);
} else {
  __shfl_up_sync(SHFL_MASK, dist, 1);
}
for (int i=16; i>0; i=i/2)
  value += __shfl_down_sync(-1, value, i);
// aggr_distances[(blockIdx.x * gridDim.y * blockDim.x) + (blockIdx.y * blockDim.x)]
printf("off: %4lu, d: %-5d val: %.3f shfl: %.3f\n", offset, threadIdx.x, distances[offset], dist);
*/
/*
*/

/* __global__ void compute_centers(DATA_TYPE* distances, uint32_t* points_clusters) {
  extern __shared__ DATA_TYPE* aggr_distances;
  uint32_t point_i = blockIdx.x;
  uint32_t center_i = threadIdx.x;
  uint32_t d_i = threadIdx.y;
  uint32_t warpid = (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) % 32;
  
  aggr_distances[(blockIdx.x * gridDim.y * blockDim.x) + (blockIdx.y * blockDim.x)]
  printf("p: %4lu, c: %4lu, d: %4lu warp: %4lu\n", point_i, center_i, d_i, warpid);
} */

__global__ void compute_centers(DATA_TYPE* centers, DATA_TYPE* points, uint32_t* points_clusters, uint64_t* clusters_len) {
  uint64_t cluster_offset = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t point_offset   = blockIdx.y * blockDim.x + threadIdx.x;

  DATA_TYPE sum = points[point_offset];
  
  if (blockIdx.y >= clusters_len[blockIdx.x]) { return; }
  
  if (blockIdx.x == 1 && blockIdx.y == 0)  printf("cl: %d p: %d d: %d p: %f sum: %f\n", blockIdx.x, blockIdx.y, threadIdx.x, points[point_offset], sum);
  
  for (int i = next_pow_2(blockDim.x); i > 0; i /= 2)
    sum += __shfl_down_sync(SHFL_MASK, sum, i);

  if (blockIdx.x == 1 && blockIdx.y == 0) printf("blk: %d %d %d, sum: %.3f\n", blockIdx.x, blockIdx.y, threadIdx.x, sum);

  if (threadIdx.x == 0) {
    centers[(blockIdx.x * blockDim.x) + blockIdx.y] = sum;
  }
}

/* Kmeans class */
void Kmeans::initCenters (Point<DATA_TYPE>** points) {
  uniform_int_distribution<int> random_int(0, n - 1);
  CHECK_CUDA_ERROR(cudaHostAlloc(&h_centers, CENTERS_BYTES, cudaHostAllocDefault));
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
  CHECK_CUDA_ERROR(cudaMemcpy(d_centers, h_centers, CENTERS_BYTES, cudaMemcpyHostToDevice));
}

Kmeans::Kmeans (size_t _n, unsigned int _d, unsigned int _k, Point<DATA_TYPE>** points)
    : n(_n), d(_d), k(_k),
    POINTS_BYTES(_n * _d * sizeof(DATA_TYPE)),
    CENTERS_BYTES(_k * _d * sizeof(DATA_TYPE)) {

  CHECK_CUDA_ERROR(cudaHostAlloc(&h_points, POINTS_BYTES, cudaHostAllocDefault));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < d; ++j) {
      h_points[i * d + j] = points[i]->get(j);
    }
  }
  CHECK_CUDA_ERROR(cudaMalloc(&d_points, POINTS_BYTES));
  CHECK_CUDA_ERROR(cudaMemcpy(d_points, h_points, POINTS_BYTES, cudaMemcpyHostToDevice));

  initCenters(points);
}

Kmeans::~Kmeans () {
  CHECK_CUDA_ERROR(cudaFreeHost(h_points));
  CHECK_CUDA_ERROR(cudaFreeHost(h_centers));
  CHECK_CUDA_ERROR(cudaFree(d_centers));
  CHECK_CUDA_ERROR(cudaFree(d_points));
}

bool Kmeans::run (uint64_t maxiter) {
  cout << "Running..." << endl;
  DATA_TYPE* d_distances;
  CHECK_CUDA_ERROR(cudaMalloc(&d_distances, n * k * sizeof(DATA_TYPE)));
  uint32_t* d_points_clusters;
  CHECK_CUDA_ERROR(cudaMalloc(&d_points_clusters, n * sizeof(uint32_t)));
  uint32_t* h_points_clusters;
  CHECK_CUDA_ERROR(cudaMallocHost(&h_points_clusters, n * sizeof(uint32_t)));
  uint64_t* h_clusters_len;
  CHECK_CUDA_ERROR(cudaMallocHost(&h_clusters_len, k * sizeof(uint64_t)));
  uint64_t* d_clusters_len;
  CHECK_CUDA_ERROR(cudaMalloc(&d_clusters_len, k * sizeof(uint64_t)));

  uint64_t iter = 0;
  uint64_t max_cluster_len = 0;
  dim3 dist_grid_dim(n, k);
  dim3 argmin_block_dim(k, d);
  while (iter++ < maxiter) { // && !cmpCenters()) { 
    if (DEBUG_KERNELS_INVOKATION) printf("compute_distances: Grid (%d, %d, %d), Block (%d, %d, %d)\n", dist_grid_dim.x, dist_grid_dim.y, dist_grid_dim.z, d, 1, 1);
    compute_distances<<<dist_grid_dim, d>>>(d_distances, d_centers, d_points);
    CHECK_LAST_CUDA_ERROR();

    DATA_TYPE tmp[n * k];
    CHECK_CUDA_ERROR(cudaMemcpy(tmp, d_distances, n * k * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < n; ++i)
      for (uint32_t j = 0; j < k; ++j)
        printf("%u %u -> %.3f\n", i, j, tmp[i * k + j]);
    cout << endl;

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

      printf("Argmin point %lu: %d %.3f\n", i, argmin_idx, argmin_val);
      ++h_clusters_len[argmin_idx];
      max_cluster_len = max_cluster_len > h_clusters_len[argmin_idx] ? max_cluster_len : h_clusters_len[argmin_idx];
      h_points_clusters[i] = argmin_idx;
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_points_clusters, h_points_clusters, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_clusters_len, h_clusters_len, k * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
    dim3 centers_grid_dim(k, max_cluster_len);
    if (DEBUG_KERNELS_INVOKATION) printf("compute_distances: Grid (%d, %d, %d), Block (%d, %d, %d)\n", centers_grid_dim.x, centers_grid_dim.y, centers_grid_dim.z, d, 1, 1);
    compute_centers<<<centers_grid_dim, d>>>(d_centers, d_points, d_points_clusters, d_clusters_len);
    CHECK_LAST_CUDA_ERROR();

    DATA_TYPE tmp2[d * k];
    CHECK_CUDA_ERROR(cudaMemcpy(tmp2, d_centers, d * k * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
    cout << endl << "CENTERS" << endl;
    for (uint32_t i = 0; i < k; ++i) {
      for (uint32_t j = 0; j < d; ++j)
        printf("%.3f, ", tmp2[i * d + j]);
      cout << endl;
    }
    cout << endl;
  }

  CHECK_CUDA_ERROR(cudaFree(d_distances));
  return false;
}

bool Kmeans::cmpCenters () {
  return true;
}

void Kmeans::to_csv(ostream& o, char separator) {
  o << "cluster" << separator;
  for (size_t i = 0; i < d; ++i) {
    o << "d" << i;
    if (i != (d - 1)) o << separator;
  }
  o << endl;
  for (size_t i = 0; i < n; ++i) {
    o << 0 << separator; // FIXME cluser
    for (size_t j = 0; j < d; ++j) {
      o << setprecision(8) << h_points[i * d + j];
      if (j != (d - 1)) o << separator;
    }
    o << endl;
  }
}