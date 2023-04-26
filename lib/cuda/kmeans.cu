#include <stdio.h>
#include <vector>
#include <random>
#include <iomanip>

#include "../../include/common.h"
#include "kmeans.h"
#include "utils.cuh"

using namespace std;

random_device rd;
mt19937 rng(rd());

/* Device vars and kernels */
__device__ DATA_TYPE* d_points;
__device__ DATA_TYPE* d_centers;

__device__ void compute_distances(DATA_TYPE* centers, DATA_TYPE* point, uint32_t d, uint32_t k) {
  // uint64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  // for (uint32_t i = 0; i < d; ++i) // What if d is big?
}

__device__ void compute_nearest_centers(DATA_TYPE* centers, DATA_TYPE* point, uint32_t d, uint32_t k) {
  // uint64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  // for (uint32_t i = 0; i < d; ++i) // What if d is big?
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
  if (DEBUG_PRINTS) { cout << endl << "Centers" << endl; for (i = 0; i < k; ++i) cout << *(centers[i]) << endl; }

  CHECK_CUDA_ERROR(cudaHostAlloc(&h_centers, CENTERS_BYTES, cudaHostAllocDefault));
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < k; ++j) {
      h_centers[i * k + j] = centers[i]->get(j);
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

Kmeans::~Kmeans () { // Free memory
  CHECK_CUDA_ERROR(cudaFreeHost(h_points));
  CHECK_CUDA_ERROR(cudaFreeHost(h_centers));
  CHECK_CUDA_ERROR(cudaFree(d_centers));
  CHECK_CUDA_ERROR(cudaFree(d_points));
}

bool run (uint64_t maxiter) {
  uint64_t iter = 0;
  while (iter++ < maxiter) { // && !cmpCenters()) {
    
  }
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