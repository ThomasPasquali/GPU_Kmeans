#include <stdio.h>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>

#include "../include/cxxopts.hpp"
#include "../include/input_parser.hpp"
#include "../include/kmeans.h"

#define ARG_DIMENSIONS  0
#define ARG_SAMPLES     1
#define ARG_CLUSTERS    2
#define ARG_MAXITER     3
#define ARG_OUTFILE     4
#define ARG_INFILE      5

#define DEBUG_INPUT_DATA  1
#define DEBUG_OUTPUT_INFO 1

const char* ARG_STR[6] = {"dimensions", "n-samples", "clusters", "maxiter", "out-file", "in-file"};

using namespace std;

cxxopts::ParseResult args;
int getArg_u (int arg) {
  try {
    return args[ARG_STR[arg]].as<int>();
  } catch(...) {
    fprintf(stderr, "Invalid or missing argument: ");
    cerr << ARG_STR[arg] << endl;
    exit(1);
  }
}
string getArg_s (int arg) {
  try {
    return args[ARG_STR[arg]].as<string>();
  } catch(...) {
    fprintf(stderr, "Invalid or missing argument: ");
    cerr << ARG_STR[arg] << endl;
    exit(1);
  }
}

int main(int argc, char **argv) {
  // Read input args
  cxxopts::Options options("kmeans", "kmeans is an implementation of the K-means algorithm that uses a CPU");
  
  options.add_options()
    ("h,help", "Print usage")
    ("d,dimensions",  "Number of dimensions of a point",  cxxopts::value<int>())
    ("n,n-samples",   "Number of points",                 cxxopts::value<int>())
    ("k,clusters",    "Number of clusters",               cxxopts::value<int>())
    ("m,maxiter",     "Maximum number of iterations",     cxxopts::value<int>())
    ("o,out-file",    "Output filename",                  cxxopts::value<string>())
    ("i,in-file",     "Input filename",                   cxxopts::value<string>());

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

  InputParser<DATA_TYPE>* input;

  if(args[ARG_STR[ARG_INFILE]].count() > 0) {
    string in_file = getArg_s(ARG_INFILE);
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
  
  Kmeans kmeans(n, d, k, input->get_dataset());
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