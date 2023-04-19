#ifndef __INPUT_PARSER__
#define __INPUT_PARSER__

#include <string.h>
#include "point.hpp"

#define SEPARATOR ","
#define MAX_LINE  4096

using namespace std;

template <typename T> class InputParser {
  private:
    Point<T> *dataset;
    int size;

  public:
    InputParser (istream &in, int dim, int size) {
      this->dataset = new Point<T>[size];
      this->size = size;
      
      T *point = new T[dim];
      for (int i = -1; i < size; i++) {
        char str[MAX_LINE] = { 0 }; in >> str; 
        
        if (i == -1) { continue; }
        if (!str[0]) { break; }
        
        int j = 0;
        char *tok = strtok(str, SEPARATOR);
        while (tok && j < dim) {
          point[j++] = atof(tok);
          tok = strtok(NULL, SEPARATOR);
        }

        Point<T> p(point, dim);
        dataset[i] = p;
      }
      
      delete[] point;
    }

    ~InputParser() {
      delete[] dataset;
    }

    Point<T> *get_dataset();
    int       get_dataset_size();

    friend ostream& operator<< (ostream &os, InputParser const& p) {
      os << "DATASET={";
      for (int i = 0; i < p.size; i++) {
        os << p.dataset[i]; 
        i == (p.size - 1) ? os << "" : os << ", ";
      }
      os << "}";
      return os;
    }
};

#endif