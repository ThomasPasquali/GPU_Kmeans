#include <stdio.h>

#define DATA_TYPE float

using namespace std;

#include "../include/cxxopts.hpp"
#include "../include/input_parser.hpp"

int main(int argc, char **argv) {
  cxxopts::Options options("MyProgram", "One line description of MyProgram");
  
  options.add_options()
    ("d,dim",  "Number of dimensions of a point", cxxopts::value<int>())
    ("h,help", "Print usage")
    ("l,size", "Number of points",                cxxopts::value<int>());

  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    cout << options.help() << endl;
    exit(0);
  }

  int dim  = 0, 
      size = 0;
  try {
    dim  = result["dim"].as<int>();
    size = result["size"].as<int>();
  } catch(...) {
    cerr << "Missing args!" << endl;
    exit(1);
  }

  cout << "DIM = " << dim << " SIZE = " << size << endl;
  
  InputParser<DATA_TYPE> input(cin, dim, size);
  cout << input << endl;
  
  return 0;
}