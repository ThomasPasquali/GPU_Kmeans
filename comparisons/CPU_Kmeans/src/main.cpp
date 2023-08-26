#include <stdio.h>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <limits>
#include <chrono>

#include "../include/cxxopts.hpp"
#include "../include/input_parser.hpp"
#include "../include/kmeans.h"

#define ARG_DIM         "n-dimensions"
#define ARG_SAMPLES     "n-samples"
#define ARG_CLUSTERS    "n-clusters"
#define ARG_MAXITER     "maxiter"
#define ARG_OUTFILE     "out-file"
#define ARG_INFILE      "in-file"
#define ARG_TOL         "tolerance"
#define ARG_RUNS        "runs"
#define ARG_SEED        "seed"

#define DEBUG_INPUT_DATA  0
#define DEBUG_OUTPUT_INFO 1

const float DEF_EPSILON = numeric_limits<float>::epsilon();
const int   DEF_RUNS    = 1;

using namespace std;

cxxopts::ParseResult args;
int getArg_u (const char *arg, const int *def_val) {
  try {
    return args[arg].as<int>();
  } catch(...) {
    if (def_val) { return *def_val; }
    fprintf(stderr, "Invalid or missing argument: ");
    cerr << arg << endl;
    exit(1);
  }
}

float getArg_f (const char *arg, const float *def_val) {
  try {
    return args[arg].as<float>();
  } catch(...) {
    if (def_val) { return *def_val; }
    fprintf(stderr, "Invalid or missing argument: ");
    cerr << arg << endl;
    exit(1);
  }
}

string getArg_s (const char *arg, const string *def_val) {
  try {
    return args[arg].as<string>();
  } catch(...) {
    if (def_val) { return *def_val; }
    fprintf(stderr, "Invalid or missing argument: ");
    cerr << arg << endl;
    exit(1);
  }
}

int main(int argc, char **argv) {
  // Read input args
  cxxopts::Options options("kmeans", "kmeans is an implementation of the K-means algorithm that uses a CPU");

  options.add_options()
    ("h,help", "Print usage")
    ("d," ARG_DIM,      "Number of dimensions of a point",  cxxopts::value<int>())
    ("n," ARG_SAMPLES,  "Number of points",                 cxxopts::value<int>())
    ("k," ARG_CLUSTERS, "Number of clusters",               cxxopts::value<int>())
    ("m," ARG_MAXITER,  "Maximum number of iterations",     cxxopts::value<int>())
    ("o," ARG_OUTFILE,  "Output filename",                  cxxopts::value<string>())
    ("i," ARG_INFILE,   "Input filename",                   cxxopts::value<string>())
    ("r," ARG_RUNS,     "Number of k-means runs",           cxxopts::value<int>()->default_value(to_string(DEF_RUNS)))
    ("s," ARG_SEED,     "Seed for centroids generator",     cxxopts::value<int>())
    ("t," ARG_TOL,      "Tolerance to declare convergence", cxxopts::value<float>()->default_value(to_string(DEF_EPSILON)));

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
  const uint32_t runs     = getArg_u(ARG_RUNS,     &DEF_RUNS);

  int *seed = NULL;
  if (args[ARG_SEED].count() > 0) {
    int in_seed = getArg_u(ARG_SEED, NULL);
    seed = new int(in_seed);
  }

  InputParser<DATA_TYPE>* input;
  if(args[ARG_INFILE].count() > 0) {
    const string in_file = getArg_s(ARG_INFILE, NULL);
    filebuf fb;
    if (fb.open(in_file, ios::in)) {
      istream file(&fb);
      input = new InputParser<DATA_TYPE>(file, d, n);
      fb.close();
    } else {
      fprintf(stderr, "Invalid input file\n");
      exit(1);
    }
  } else {
    input = new InputParser<DATA_TYPE>(cin, d, n);
  }

  if (DEBUG_INPUT_DATA) cout << "Points" << endl << *input << endl;

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
        printf("K-means converged at iteration %lu\n", converged);
      else
        printf("K-means did NOT converge\n");
      printf("Time: %lf\n", duration.count());
    #endif
  }

  printf("CPU_Kmeans: %lfs (%u runs)\n", tot_time / runs, runs);

  ofstream fout(out_file);
  input->dataset_to_csv(fout);
  fout.close();
  delete seed;

  return 0;
}