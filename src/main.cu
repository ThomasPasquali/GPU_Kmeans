#include <stdio.h>
#include <vector>
#include <algorithm>
#include <random>

#include "../include/cxxopts.hpp"
#include "../include/input_parser.hpp"
#include "../include/errors.hpp"
#include "../lib/cuda/utils.cuh"
#include "../lib/cuda/kmeans.cuh"

#define DATA_TYPE float

#define ARG_DIMENSIONS  0
#define ARG_SAMPLES     1
#define ARG_CLUSTERS    2
#define ARG_MAXITER     3
const char* ARG_STR[4] = {"dimensions", "n-samples", "clusters", "maxiter"};

using namespace std;

__device__ DATA_TYPE* d_points;

cxxopts::ParseResult args;
int getArg_u (int arg) {
  try {
    return args[ARG_STR[arg]].as<int>();
  } catch(...) {
    printErrDesc(EXIT_ARGS);
    cerr << ARG_STR[arg] << endl;
    exit(EXIT_ARGS);
  }
}
/* FIXME size_t getArg_ul (int arg) { 
  try {
    return args[ARG_STR[arg]].as<size_t>();
  } catch(...) {
    printErrDesc(EXIT_ARGS);
    cerr << ARG_STR[arg] << endl;
    exit(EXIT_ARGS);
  }
} */

random_device rd;
mt19937 rng(rd());

int main(int argc, char **argv) {
  // Read input args
  cxxopts::Options options("gpukmeans", "gpukmeans is an implementation of the K-means algorithm that uses a GPU");
  
  options.add_options()
    ("h,help", "Print usage")
    ("d,dimensions",  "Number of dimensions of a point",  cxxopts::value<int>())
    ("n,n-samples",   "Number of points",                 cxxopts::value<int>())
    ("k,clusters",    "Number of clusters",               cxxopts::value<int>())
    ("m,maxiter",     "Maximum number of iterations",     cxxopts::value<int>());

  args = options.parse(argc, argv);

  if (args.count("help")) {
    cout << options.help() << endl;
    exit(0);
  }

  unsigned int  d = getArg_u(ARG_DIMENSIONS);
  size_t        n = getArg_u(ARG_SAMPLES);
  unsigned int  k = getArg_u(ARG_CLUSTERS);
  size_t        maxiter = getArg_u(ARG_MAXITER);
  
  InputParser<DATA_TYPE> input(cin, d, n);
  cout << input << endl;

  // Check devices
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
    exit(EXIT_FAILURE);
  }
  if (deviceCount == 0) {
    printErrDesc(EXIT_CUDA_DEV);
    exit(EXIT_CUDA_DEV);
  } else {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  int dev = 0;
  cudaSetDevice(dev); // Use device 0 by default
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  // describeDevice(dev, deviceProp);

  // K-means variables setup
  uint64_t iter = 0, i;
  DATA_TYPE* h_points;
  CHECK_CUDA_ERROR(cudaHostAlloc(&h_points, n * d * sizeof(DATA_TYPE), cudaHostAllocDefault));
  for (i = 0; i < n; ++i) {
    for (unsigned int j = 0; j < d; ++j) {
      h_points[i * d + j] = input.get_dataset()[i]->get(j);
    }
  }

  uniform_int_distribution<int> random_int(0, n - 1);
  DATA_TYPE* h_centers;
  CHECK_CUDA_ERROR(cudaHostAlloc(&h_centers, k * d * sizeof(DATA_TYPE), cudaHostAllocDefault));
  i = 0;
  // Add k random centers sampled form points
  vector<Point<DATA_TYPE>*> usedPoints;
  Point<DATA_TYPE>* centers[k];
  while (i < k) {
    Point<DATA_TYPE>* p = input.get_dataset()[random_int(rng)];
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
  cout << endl << "Centers" << endl; for (i = 0; i < k; ++i) cout << *(centers[i]) << endl;

  CHECK_CUDA_ERROR(cudaMalloc(&d_points, n * d * sizeof(DATA_TYPE)));
  

  // K-means algoritm
  while (iter++ < maxiter) { // && !cmpCenters()) {
    
  }


  // Free memory
  cudaFreeHost(h_points);
  cudaFreeHost(h_centers);
  return 0;
}