#include <chrono>

#include "include/common.h"
#include "include/colors.h"
#include "include/input_parser.hpp"
#include "include/utils.hpp"

#include "kmeans.cuh"
#include "cuda_utils.cuh"

#define DEVICE 0

using namespace std;

int main(int argc, char **argv) {
  uint32_t d, k, runs;
  size_t   n, maxiter;
  string   out_file;
  float    tol;
  int     *seed = NULL;
  InputParser<float> *input = NULL;

  parse_input_args(argc, argv, &d, &n, &k, &maxiter, out_file, &tol, &runs, &seed, &input);

  #if DEBUG_INPUT_DATA
    cout << "Points" << endl << *input << endl;
  #endif

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

  printf(BOLDBLUE);
  double tot_time = 0;
  for (uint32_t i = 0; i < runs; i++) {
    Kmeans kmeans(n, d, k, tol, seed, input->get_dataset(), &deviceProp);
    const auto start = chrono::high_resolution_clock::now();
    uint64_t converged = kmeans.run(maxiter);
    const auto end = chrono::high_resolution_clock::now();

    const auto duration = chrono::duration_cast<chrono::duration<double>>(end - start);
    tot_time += duration.count();

    #if DEBUG_OUTPUT_INFO
      if (converged < maxiter)
        printf("K-means converged at iteration %lu - ", converged);
      else
        printf("K-means did NOT converge - ");
      printf("Time: %lf\n", duration.count());
    #endif
  }

  printf("GPU_Kmeans: %lfs (%u runs)\n", tot_time / runs, runs);
  printf(RESET);

  ofstream fout(out_file);
  input->dataset_to_csv(fout);
  fout.close();
  delete seed;

  return 0;
}