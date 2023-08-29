#include <assert.h>
#include <kmcuda.h>
#include <chrono>

#include <colors.h>
#include <input_parser.hpp>
#include <utils.hpp>

int main(int argc, const char **argv) {
  uint32_t d, k, runs;
  size_t   n, maxiter;
  string   out_file;
  float    tol;
  int     *seed = NULL;
  InputParser<float> *input = NULL;

  parse_input_args(argc, argv, &d, &n, &k, &maxiter, out_file, &tol, &runs, &seed, &input);

  float *samples = new float[n * d];
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < d; ++j) {
      samples[i * d + j] = input->get_dataset()[i]->get(j);
    }
  }

  float *centroids = new float[k * d];
  uint32_t *assignments = new uint32_t[n];
  float average_distance;

  printf(BOLDGREEN);
  double tot_time = 0;
  for (uint32_t i = 0; i < runs; i++) {
    const auto start = chrono::high_resolution_clock::now();

    KMCUDAResult result = kmeans_cuda(
      kmcudaInitMethodRandom, NULL,    // centroids initialization
      tol,                             // less than 1% of the samples are reassigned in the end
      0,                               // deactivate Yinyang refinement
      kmcudaDistanceMetricL2,          // Euclidean distance
      n, d, k,
      seed ? *seed : 0xDEADBEEF,       // random generator seed
      1,                               // use all available CUDA devices
      -1,                              // samples are supplied from host
      0,                               // not in float16x2 mode
      0,                               // no verbosity
      samples, centroids, assignments, &average_distance);

    const auto end = chrono::high_resolution_clock::now();
    const auto duration = chrono::duration_cast<chrono::duration<double>>(end - start);
    tot_time += duration.count();

    assert(result == kmcudaSuccess);
    printf("Run %u - Time: %lfs\n", i, duration.count());
  }

  printf("KMCUDA: %lfs (%u runs)\n", tot_time / runs, runs);
  printf(RESET);

  for (uint32_t i = 0; i < n; i++) {
    input->get_dataset()[i]->setCluster(assignments[i]);
  }

  ofstream fout(out_file);
  input->dataset_to_csv(fout);
  fout.close();

  delete   seed;
  delete[] samples;
  delete[] centroids;
  delete[] assignments;
  return 0;
}