#include <stdio.h>
#include <vector>
#include <algorithm>

#include "../include/common.h"
#include "../include/cxxopts.hpp"
#include "../include/input_parser.hpp"
#include "../include/errors.hpp"
#include "../lib/cuda/utils.cuh"
#include "../lib/cuda/kmeans.h"

#define ARG_DIMENSIONS  0
#define ARG_SAMPLES     1
#define ARG_CLUSTERS    2
#define ARG_MAXITER     3
const char* ARG_STR[4] = {"dimensions", "n-samples", "clusters", "maxiter"};

using namespace std;

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
  if (DEBUG_INPUT_DATA) cout << input << endl;

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
  } else if (DEBUG_DEVICE) {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  int dev = 0;
  cudaSetDevice(dev); // Use device 0 by default
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  if (DEBUG_DEVICE) describeDevice(dev, deviceProp);
  
  Kmeans kmeans(n, d, k, input.get_dataset());
  kmeans.run(maxiter);
  kmeans.to_csv(cout);

  return 0;
}