#include <chrono>

#include <colors.h>
#include <input_parser.hpp>
#include <utils.hpp>

#include "../include/kmeans.h"

#define DEBUG_INPUT_DATA  0
#define DEBUG_OUTPUT_INFO 1

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

  printf(BOLDMAGENTA);
  double tot_time = 0;
  for (uint32_t i = 0; i < runs; i++) {
    Kmeans kmeans(n, d, k, tol, seed, input->get_dataset());
    const auto start = chrono::high_resolution_clock::now();
    uint64_t converged = kmeans.run(maxiter);
    const auto end = chrono::high_resolution_clock::now();

    const auto duration = chrono::duration_cast<chrono::duration<double>>(end - start);
    tot_time += duration.count();

    #if DEBUG_OUTPUT_INFO
      if (converged < maxiter)
        printf("K-means converged at iteration %lu - ", converged);
      else
        printf("K-means did NOT converge - \n");
      printf("Time: %lf\n", duration.count());
    #endif
  }

  printf("CPU_Kmeans: %lfs (%u runs)\n", tot_time / runs, runs);
  printf(RESET);

  ofstream fout(out_file);
  input->dataset_to_csv(fout);
  fout.close();
  delete seed;

  return 0;
}