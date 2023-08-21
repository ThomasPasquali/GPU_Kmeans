#include <stdio.h>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <limits>
#include <chrono>

#include "include/common.h"
#include "include/colors.h"
#include "include/cxxopts.hpp"
#include "include/input_parser.hpp"
#include "include/errors.hpp"
#include "utils.cuh"
#include "kmeans.cuh"

#define DEVICE          0
#define ARG_DIM         "n-dimensions"
#define ARG_SAMPLES     "n-samples"
#define ARG_CLUSTERS    "n-clusters"
#define ARG_MAXITER     "maxiter"
#define ARG_OUTFILE     "out-file"
#define ARG_INFILE      "in-file"
#define ARG_TOL         "tolerance"

const char* ARG_STR[] = {"dimensions", "n-samples", "clusters", "maxiter", "out-file", "in-file", "tolerance"};
const float DEF_EPSILON = numeric_limits<float>::epsilon();

using namespace std;

cxxopts::ParseResult args;
int getArg_u (const char *arg, const int *def_val) {
  try {
    return args[arg].as<int>();
  } catch(...) {
    if (def_val) { return *def_val; }
    printErrDesc(EXIT_ARGS);
    cerr << arg << endl;
    exit(EXIT_ARGS);
  }
}

float getArg_f (const char *arg, const float *def_val) {
  try {
    return args[arg].as<float>();
  } catch(...) {
    if (def_val) { return *def_val; }
    printErrDesc(EXIT_ARGS);
    cerr << arg << endl;
    exit(EXIT_ARGS);
  }
}

string getArg_s (const char *arg, const string *def_val) {
  try {
    return args[arg].as<string>();
  } catch(...) {
    if (def_val) { return *def_val; }
    printErrDesc(EXIT_ARGS);
    cerr << arg << endl;
    exit(EXIT_ARGS);
  }
}

int main(int argc, char **argv) {
  // Read input args
  cxxopts::Options options("gpukmeans", "gpukmeans is an implementation of the K-means algorithm that uses a GPU");

  options.add_options()
    ("h,help", "Print usage")
    ("d," ARG_DIM,      "Number of dimensions of a point",  cxxopts::value<int>())
    ("n," ARG_SAMPLES,  "Number of points",                 cxxopts::value<int>())
    ("k," ARG_CLUSTERS, "Number of clusters",               cxxopts::value<int>())
    ("m," ARG_MAXITER,  "Maximum number of iterations",     cxxopts::value<int>())
    ("o," ARG_OUTFILE,  "Output filename",                  cxxopts::value<string>())
    ("i," ARG_INFILE,   "Input filename",                   cxxopts::value<string>())
    ("t," ARG_TOL,      "Tolerance of the difference in the cluster centers "\
                        "of two consecutive iterations to declare convergence", cxxopts::value<float>()->default_value(to_string(DEF_EPSILON)));

  args = options.parse(argc, argv);

  if (args.count("help")) {
    cout << options.help() << endl;
    exit(0);
  }

  const uint32_t d        = getArg_u(ARG_DIM,      NULL);
  const size_t   n        = getArg_u(ARG_SAMPLES,  NULL);
  const uint32_t k        = getArg_u(ARG_CLUSTERS, NULL);
  const size_t   maxiter  = getArg_u(ARG_MAXITER,  NULL);
  const string   out_file = getArg_s(ARG_OUTFILE,  NULL);
  const float    tol      = getArg_f(ARG_TOL,      &DEF_EPSILON);

  InputParser<DATA_TYPE>* input;

  if(args[ARG_INFILE].count() > 0) {
    const string in_file = getArg_s(ARG_INFILE, NULL);
    filebuf fb;
    if (fb.open(in_file, ios::in)) {
      istream file(&fb);
      input = new InputParser<DATA_TYPE>(file, d, n);
      fb.close();
    } else {
      printErrDesc(EXIT_INVALID_INFILE);
      exit(EXIT_INVALID_INFILE);
    }
  } else {
    input = new InputParser<DATA_TYPE>(cin, d, n);
  }

  if (DEBUG_INPUT_DATA) cout << "Points" << endl << *input << endl;

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

  cudaDeviceProp deviceProp;
  getDeviceProps(DEVICE, &deviceProp);
  if (DEBUG_DEVICE) describeDevice(DEVICE, deviceProp);

  Kmeans kmeans(n, d, k, tol, input->get_dataset(), &deviceProp);

  const auto start = chrono::high_resolution_clock::now();
  uint64_t converged = kmeans.run(maxiter);
  const auto end = chrono::high_resolution_clock::now();
  const auto duration = chrono::duration_cast<chrono::duration<double>>(end - start);

  #if DEBUG_OUTPUT_INFO
    if (converged < maxiter)
      printf(BOLDBLUE "K-means converged at iteration %lu\n", converged);
    else
      printf(BOLDBLUE "K-means did NOT converge\n");
    printf("Time: %lf\n" RESET, duration.count());
  #endif

  ofstream fout(out_file);
  kmeans.to_csv(fout);
  fout.close();

  return 0;
}