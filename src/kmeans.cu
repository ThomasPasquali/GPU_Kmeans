#include <stdio.h>
#include <vector>
#include <random>
#include <iomanip>
#include <cub/cub.cuh>
#include <cmath>
#include <limits>

#include "include/common.h"
#include "include/colors.h"

#include "utils.cuh"
#include "kmeans.cuh"

#include "kernels/kernels.cuh"

using namespace std;

random_device rd;
seed_seq seed{0}; // FIXME use rd()
mt19937 rng(seed);

/* Kmeans class */
void Kmeans::init_centroids (Point<DATA_TYPE>** points) {
  uniform_int_distribution<int> random_int(0, n - 1);
  CHECK_CUDA_ERROR(cudaHostAlloc(&h_centroids, CENTROIDS_BYTES, cudaHostAllocDefault));
  CHECK_CUDA_ERROR(cudaHostAlloc(&h_last_centroids, CENTROIDS_BYTES, cudaHostAllocDefault));
  unsigned int i = 0;
  vector<Point<DATA_TYPE>*> usedPoints;
  Point<DATA_TYPE>* centroids[k];
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
        h_centroids[i * d + j] = p->get(j);
      }
      centroids[i] = new Point<DATA_TYPE>(p);
      usedPoints.push_back(p);
      ++i;
    }
  }
  #if DEBUG_INIT_CENTROIDS
    cout << endl << "Centroids" << endl; 
    for (i = 0; i < k; ++i) 
      cout << *(centroids[i]) << endl;
  #endif

  CHECK_CUDA_ERROR(cudaHostAlloc(&h_centroids, CENTROIDS_BYTES, cudaHostAllocDefault));
  for (size_t i = 0; i < k; ++i) {
    for (size_t j = 0; j < d; ++j) {
      h_centroids[i * d + j] = centroids[i]->get(j);
    }
  }
  CHECK_CUDA_ERROR(cudaMalloc(&d_centroids, CENTROIDS_BYTES));
}

Kmeans::Kmeans (size_t _n, unsigned int _d, unsigned int _k, Point<DATA_TYPE>** _points, cudaDeviceProp* _deviceProps)
    : n(_n), d(_d), k(_k),
    POINTS_BYTES(_n * _d * sizeof(DATA_TYPE)),
    CENTROIDS_BYTES(_k * _d * sizeof(DATA_TYPE)),
    points(_points),
    deviceProps(_deviceProps) {

  CHECK_CUDA_ERROR(cudaHostAlloc(&h_points, POINTS_BYTES, cudaHostAllocDefault));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < d; ++j) {
      h_points[i * d + j] = _points[i]->get(j);
    }
  }
  CHECK_CUDA_ERROR(cudaMalloc(&d_points, POINTS_BYTES));

  const float MAX_CPY_VALUES_PER_STREAM = 1024.0; // TODO find best value
  const uint16_t streams_n = ceil((n * d) / MAX_CPY_VALUES_PER_STREAM);
  cudaStream_t streams[streams_n];
  uint64_t offset = 0, count;
  
  for (int i = 0; i < streams_n; ++i) {
    cudaStreamCreate(&(streams[i]));
    count = MAX_CPY_VALUES_PER_STREAM;
    if (i + 1 >= streams_n) {
      count -= MAX_CPY_VALUES_PER_STREAM * streams_n - (n * d);
    }
    // printf("Stream %d  off: %lu count: %lu\n", i, offset, count);
    cudaMemcpyAsync(d_points + offset, h_points + offset, sizeof(DATA_TYPE) * count, cudaMemcpyDeviceToHost, streams[i]);
    offset += count;
  }
  // CHECK_CUDA_ERROR(cudaMemcpy(d_points, h_points, POINTS_BYTES, cudaMemcpyHostToDevice));

  init_centroids(_points);
  cudaDeviceSynchronize();
  for (int i = 0; i < streams_n; ++i) {
    cudaStreamDestroy(streams[i]);
  }
}

Kmeans::~Kmeans () {
  CHECK_CUDA_ERROR(cudaFreeHost(h_points));
  CHECK_CUDA_ERROR(cudaFreeHost(h_centroids));
  CHECK_CUDA_ERROR(cudaFreeHost(h_last_centroids));
  CHECK_CUDA_ERROR(cudaFreeHost(h_points_clusters));
  CHECK_CUDA_ERROR(cudaFree(d_centroids));
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
  uint32_t* d_clusters_len;
  CHECK_CUDA_ERROR(cudaMalloc(&d_clusters_len, k * sizeof(uint32_t)));

  uint64_t iter = 0;

  #if COMPUTE_DISTANCES_KERNEL == 1
    const uint32_t dist_max_points_per_warp = deviceProps->warpSize / d;
    dim3 dist_grid_dim(ceil(((float) n) / dist_max_points_per_warp), k);
    dim3 dist_block_dim(dist_max_points_per_warp * d);
    uint32_t dist_kernel_sh_mem = k * dist_max_points_per_warp * d * sizeof(DATA_TYPE);
  #elif COMPUTE_DISTANCES_KERNEL == 2
    const uint32_t dist_max_points_per_warp = deviceProps->warpSize / next_pow_2(d); // FIXME k > 32
    dim3 dist_grid_dim(ceil(((float) n) / dist_max_points_per_warp), k);
    dim3 dist_block_dim(dist_max_points_per_warp * next_pow_2(d));
    uint32_t dist_kernel_sh_mem = 0;
  #else
    dim3 dist_grid_dim(n, k);
    dim3 dist_block_dim(d);
    uint32_t dist_kernel_sh_mem = 0;
  #endif

  #if ARGMIN_KERNEL == 1
    dim3 argmin_grid_dim(n);
    dim3 argmin_block_dim(max(next_pow_2(k), deviceProps->warpSize));
    uint32_t argmin_warps_per_block = (k + deviceProps->warpSize - 1) / deviceProps->warpSize; // Ceil
    uint32_t argmin_kernel_sh_mem = argmin_warps_per_block * sizeof(Pair);
  #endif
  
  dim3 cent_grid_dim(k);
  dim3 cent_block_dim((((int) n) > deviceProps->warpSize) ? next_pow_2((n + 1) / 2) : deviceProps->warpSize, d); 
  int cent_threads_tot = cent_block_dim.x * cent_block_dim.y;
  while (cent_threads_tot > deviceProps->maxThreadsPerBlock) {
    cent_block_dim.x /= 2;
    cent_grid_dim.y *= 2;
    cent_threads_tot = cent_block_dim.x * cent_block_dim.y;
  }  
  size_t cent_sh_mem = 0;
  #if COMPUTE_CENTROIDS_KERNEL == 1
    cent_sh_mem = (cent_block_dim.x / deviceProps->warpSize) * k * d * sizeof(DATA_TYPE);
  #endif

  /* MAIN LOOP */
  while (iter++ < maxiter) {
    /* COMPUTE DISTANCES */
    CHECK_CUDA_ERROR(cudaMemcpy(d_centroids, h_centroids, CENTROIDS_BYTES, cudaMemcpyHostToDevice));
    if (DEBUG_KERNELS_INVOKATION) printf(YELLOW "[KERNEL]" RESET " %-25s: Grid (%4u, %4u, %4u), Block (%4u, %4u, %4u), Sh.mem. %uB\n", "compute_distances", dist_grid_dim.x, dist_grid_dim.y, dist_grid_dim.z, dist_block_dim.x, dist_block_dim.y, dist_block_dim.z, dist_kernel_sh_mem);
    #if PERFORMANCES_KERNEL_DISTANCES
      cudaEvent_t e_perf_dist_start, e_perf_dist_stop;
      cudaEventCreate(&e_perf_dist_start);
      cudaEventCreate(&e_perf_dist_stop);
      cudaEventRecord(e_perf_dist_start);
    #endif
    #if COMPUTE_DISTANCES_KERNEL == 1
      compute_distances_shmem<<<dist_grid_dim, dist_block_dim, dist_kernel_sh_mem>>>(d_distances, d_centroids, d_points, dist_max_points_per_warp, d);
    #elif COMPUTE_DISTANCES_KERNEL == 2     
      compute_distances_shfl<<<dist_grid_dim, dist_block_dim>>>(d_distances, d_centroids, d_points, n, dist_max_points_per_warp, d, next_pow_2(d));
    #else
      compute_distances_one_point_per_warp<<<dist_grid_dim, dist_block_dim>>>(d_distances, d_centroids, d_points, next_pow_2(dist_block_dim.x));
    #endif
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());  
    #if PERFORMANCES_KERNEL_DISTANCES
      cudaEventRecord(e_perf_dist_stop);
      cudaEventSynchronize(e_perf_dist_stop);
      float e_perf_dist_ms = 0;
      cudaEventElapsedTime(&e_perf_dist_ms, e_perf_dist_start, e_perf_dist_stop);
      printf(CYAN "[PERFORMANCE]" RESET " compute_distances time: %.8f\n", e_perf_dist_ms / 1000);
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
          printf("%-2u %-2u -> %.3f\n", i, j, tmp[i * k + j]);
      cout << RESET << endl;
    #endif


    /* ASSIGN POINTS TO NEW CLUSTERS */
    #if DEBUG_KERNEL_ARGMIN && ARGMIN_KERNEL == 0
      printf(GREEN "[DEBUG_KERNEL_ARGMIN]\n" RESET);
    #endif
    #if PERFORMANCES_KERNEL_ARGMIN
      cudaEvent_t e_perf_argmin_start, e_perf_argmin_stop;
      cudaEventCreate(&e_perf_argmin_start);
      cudaEventCreate(&e_perf_argmin_stop);
      cudaEventRecord(e_perf_argmin_start);
    #endif
    #if ARGMIN_KERNEL == 0
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
          printf(GREEN "Argmin point %lu: %d %.3f" RESET "\n", i, argmin_idx, argmin_val);
        #endif

        ++h_clusters_len[argmin_idx];
        h_points_clusters[i] = argmin_idx;
      }
      CHECK_CUDA_ERROR(cudaMemcpy(d_points_clusters, h_points_clusters, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    #else
      CHECK_CUDA_ERROR(cudaMemset(d_clusters_len, 0, k * sizeof(uint32_t)));
      DATA_TYPE infty = numeric_limits<DATA_TYPE>::infinity();
      if (DEBUG_KERNELS_INVOKATION) printf(YELLOW "[KERNEL]" RESET " %-25s: Grid (%4u, %4u, %4u), Block (%4u, %4u, %4u), Sh.mem. %uB\n", "clusters_argmin_shfl", argmin_grid_dim.x, argmin_grid_dim.y, argmin_grid_dim.z, argmin_block_dim.x, argmin_block_dim.y, argmin_block_dim.z, argmin_kernel_sh_mem);
      clusters_argmin_shfl<<<argmin_grid_dim, argmin_block_dim, argmin_kernel_sh_mem>>>(n, k, d_distances, d_points_clusters, d_clusters_len, argmin_warps_per_block, infty);
      cudaDeviceSynchronize();
    #endif
    #if PERFORMANCES_KERNEL_ARGMIN
      cudaEventRecord(e_perf_argmin_stop);
      cudaEventSynchronize(e_perf_argmin_stop);
      float e_perf_argmin_ms = 0;
      cudaEventElapsedTime(&e_perf_argmin_ms, e_perf_argmin_start, e_perf_argmin_stop);
      printf(CYAN "[PERFORMANCE]" RESET " clusters_argmin_shfl time: %.8f\n", e_perf_argmin_ms / 1000);
      cudaEventDestroy(e_perf_argmin_stop);
      cudaEventDestroy(e_perf_argmin_start);
    #endif
    #if DEBUG_KERNEL_ARGMIN
      #if ARGMIN_KERNEL == 0
        printf("\n");
      #elif ARGMIN_KERNEL == 1
        printf(GREEN "[DEBUG_KERNEL_ARGMIN]\n" RESET);
        uint32_t tmp1[n];
        CHECK_CUDA_ERROR(cudaMemcpy(tmp1, d_points_clusters, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        printf(GREEN "p  -> c\n");
        for (uint32_t i = 0; i < n; ++i)
            printf("%-2u -> %-2u\n", i, tmp1[i]);
        cout << RESET << endl;
      #endif
    #endif

    /* COMPUTE NEW CENTROIDS */
    
    CHECK_CUDA_ERROR(cudaMemset(h_centroids, 0, k * d * sizeof(DATA_TYPE)));
    CHECK_CUDA_ERROR(cudaMemset(d_centroids, 0, k * d * sizeof(DATA_TYPE)));

    #if PERFORMANCES_KERNEL_CENTROIDS
      cudaEvent_t e_perf_cent_start, e_perf_cent_stop;
      cudaEventCreate(&e_perf_cent_start);
      cudaEventCreate(&e_perf_cent_stop);
      cudaEventRecord(e_perf_cent_start);
    #endif

    if (DEBUG_KERNELS_INVOKATION) printf(YELLOW "[KERNEL]" RESET " %-25s: Grid (%4u, %4u, %4u), Block (%4u, %4u, %4u), Sh.mem. %luB\n", "compute_centroids", cent_grid_dim.x, cent_grid_dim.y, cent_grid_dim.z, cent_block_dim.x, cent_block_dim.y, cent_block_dim.z, cent_sh_mem);
    
    #if COMPUTE_CENTROIDS_KERNEL == 1
      compute_centroids_shfl_shrd<<<cent_grid_dim, cent_block_dim, cent_sh_mem>>>(d_centroids, d_points, d_points_clusters, d_clusters_len, n, d);
    #else 
      compute_centroids_shfl<<<cent_grid_dim, cent_block_dim>>>(d_centroids, d_points, d_points_clusters, d_clusters_len, n, d);
    #endif
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());    

    #if PERFORMANCES_KERNEL_CENTROIDS
      cudaEventRecord(e_perf_cent_stop);
      cudaEventSynchronize(e_perf_cent_stop);
      float e_perf_cent_ms = 0;
      cudaEventElapsedTime(&e_perf_cent_ms, e_perf_cent_start, e_perf_cent_stop);
      printf(CYAN "[PERFORMANCE]" RESET " compute_centroids time: %.8f\n", e_perf_cent_ms / 1000);
      cudaEventDestroy(e_perf_cent_start);
      cudaEventDestroy(e_perf_cent_stop);
    #endif

    #if DEBUG_KERNEL_CENTROIDS
      uint32_t* h_clusters_len;
      CHECK_CUDA_ERROR(cudaMallocHost(&h_clusters_len, k * sizeof(uint32_t)));
      CHECK_CUDA_ERROR(cudaMemcpy(h_points_clusters, d_points_clusters, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
      CHECK_CUDA_ERROR(cudaMemcpy(h_clusters_len,    d_clusters_len,    k * sizeof(uint32_t), cudaMemcpyDeviceToHost));
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
      cout << GREEN "[DEBUG_KERNEL_CENTROIDS]" << endl;
      cout << endl << "CENTROIDS (CPU)" << endl;
      for (uint32_t i = 0; i < k; ++i) {
        for (uint32_t j = 0; j < d; ++j)
          printf("%.3f, ", h_centroids[i * d + j]);
        cout << endl;
      }
      CHECK_CUDA_ERROR(cudaMemset(h_centroids, 0, d * k * sizeof(DATA_TYPE)));     
      CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_centroids, d * k * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
      cout << endl << "CENTROIDS (GPU)" << endl;
      for (uint32_t i = 0; i < k; ++i) {
        for (uint32_t j = 0; j < d; ++j)
          printf("%.3f, ", h_centroids[i * d + j]);
        cout << endl;
      }
      cout << RESET << endl;
      CHECK_CUDA_ERROR(cudaFreeHost(h_clusters_len));
    #endif

    CHECK_CUDA_ERROR(cudaMemcpy(h_centroids, d_centroids, d * k * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));

    /* CHECK IF CONVERGED */
    if (iter > 1 && cmp_centroids()) { converged = iter; break; } // Exit
    else { memcpy(h_last_centroids, h_centroids, CENTROIDS_BYTES); } // Copy current centroids
  }
  /* MAIN LOOP END */

  /* COPY BACK RESULTS*/
  CHECK_CUDA_ERROR(cudaMemcpy(h_points_clusters, d_points_clusters, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n; i++) {
    points[i]->setCluster(h_points_clusters[i]);
  }
  
  /* FREE MEMORY */
  CHECK_CUDA_ERROR(cudaFree(d_distances));
  CHECK_CUDA_ERROR(cudaFree(d_points_clusters));
  CHECK_CUDA_ERROR(cudaFree(d_clusters_len));

  return converged;
}

bool Kmeans::cmp_centroids () {
  const DATA_TYPE EPSILON = numeric_limits<DATA_TYPE>::epsilon();
  DATA_TYPE dist_sum = 0, norm = 0;
  
  for (size_t i = 0; i < k; ++i) {
    for (size_t j = 0; j < d; ++j) {
      DATA_TYPE dist = fabs(h_centroids[i * d + j] - h_last_centroids[i * d + j]);
      dist_sum += dist * dist;
      norm += h_last_centroids[i * d + j] * h_last_centroids[i * d + j];
    }
    if (sqrt(dist_sum) > EPSILON) { return false; }
  }

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
    o << h_points_clusters[i] << separator;
    for (size_t j = 0; j < d; ++j) {
      o << setprecision(8) << h_points[i * d + j];
      if (j != (d - 1)) o << separator;
    }
    o << endl;
  }
}