#include <stdio.h>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>

#include "../include/common.h"
#include "../include/cxxopts.hpp"
#include "../include/input_parser.hpp"
#include "../include/errors.hpp"
#include "../lib/cuda/utils.cuh"
#include "./kmeans.h"

#define ARG_DIMENSIONS  0
#define ARG_SAMPLES     1
#define ARG_CLUSTERS    2
#define ARG_MAXITER     3
#define ARG_OUTFILE     4
const char* ARG_STR[5] = {"dimensions", "n-samples", "clusters", "maxiter", "out-file"};

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
string getArg_s (int arg) {
  try {
    return args[ARG_STR[arg]].as<string>();
  } catch(...) {
    printErrDesc(EXIT_ARGS);
    cerr << ARG_STR[arg] << endl;
    exit(EXIT_ARGS);
  }
}

int main(int argc, char **argv) {
  // Read input args
  cxxopts::Options options("gpukmeans", "gpukmeans is an implementation of the K-means algorithm that uses a GPU");
  
  options.add_options()
    ("h,help", "Print usage")
    ("d,dimensions",  "Number of dimensions of a point",  cxxopts::value<int>())
    ("n,n-samples",   "Number of points",                 cxxopts::value<int>())
    ("k,clusters",    "Number of clusters",               cxxopts::value<int>())
    ("m,maxiter",     "Maximum number of iterations",     cxxopts::value<int>())
    ("o,out-file",    "Output filename",                  cxxopts::value<string>());

  args = options.parse(argc, argv);

  if (args.count("help")) {
    cout << options.help() << endl;
    exit(0);
  }

  unsigned int  d         = getArg_u(ARG_DIMENSIONS);
  size_t        n         = getArg_u(ARG_SAMPLES);
  unsigned int  k         = getArg_u(ARG_CLUSTERS);
  size_t        maxiter   = getArg_u(ARG_MAXITER);
  string        out_file  = getArg_s(ARG_OUTFILE);
  
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
  uint64_t converged = kmeans.run(maxiter);

  #if DEBUG_OUTPUT_INFO
    if (converged < maxiter)
      printf("K-means converged at iteration %lu\n", converged);
    else
      printf("K-means did NOT converge\n");
  #endif

  ofstream fout(out_file);
  kmeans.to_csv(fout);
  fout.close();

  return 0;
}