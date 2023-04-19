#include <stdio.h>

#define DATA_TYPE float

using namespace std;

#include "../include/cxxopts.hpp"
#include "../include/input_parser.hpp"

int main(int argc, char **argv) {
  cxxopts::Options options("gpukmeans", "gpukmeans is an implementation of the K-means algorithm that uses a GPU");
  
  options.add_options()
    ("h,help", "Print usage")
    ("d,dimensions",  "Number of dimensions of a point",  cxxopts::value<int>())
    ("n,n-samples", "Number of points",                   cxxopts::value<int>());

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    cout << options.help() << endl;
    exit(0);
  }

  int d = 0, n = 0;
  try {
    d  = result["dimensions"].as<int>();
    n = result["n-samples"].as<int>();
  } catch(...) {
    cerr << "Missing args!" << endl;
    exit(1);
  }
  
  InputParser<DATA_TYPE> input(cin, d, n);
  cout << input << endl;
  
  return 0;
}