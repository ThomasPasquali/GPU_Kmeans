#ifndef __INPUT_PARSER__
#define __INPUT_PARSER__

#include <string.h>
#include "point.hpp"

#define SEPARATOR ","
#define MAX_LINE  4096

using namespace std;

template <typename T> 
class InputParser {
  private:
    Point<T> **dataset;
    int d;
    size_t n;

  public:
    InputParser (istream &in, int _d, size_t _n) {
      this->dataset = new Point<T>*[_n];
      this->n = _n;
      this->d = _d;
      
      T *point = new T[_d];
      for (size_t i = 0; i <= _n; i++) {
        char str[MAX_LINE] = { 0 };
        in >> str;
        
        if (i == (size_t)0) { continue; }
        if (!str[0]) { break; }
        
        int j = 0;
        char *tok = strtok(str, SEPARATOR);
        while (tok && j < _d) {
          point[j++] = atof(tok);
          tok = strtok(NULL, SEPARATOR);
        }

        dataset[i - 1] = new Point<T>(point, _d);
      }
      
      delete[] point;
    }

    ~InputParser() {
      for (size_t i = 0; i < n; ++i) delete dataset[i];
      delete[] dataset;
    }

    Point<T> **get_dataset () { return dataset; }
    size_t get_dataset_size() { return n; };

    friend ostream& operator<< (ostream &os, InputParser const& p) {
      const int W = 9;
      size_t count = 0;
      os << "i  cluster";
      for (int i = 0; i < p.d; ++i) {
        char s[W];
        sprintf(s, "d%d", i);
        os << setw(W) << s;
      }
      os << endl;
      for (size_t i = 0; i < p.n; ++i) {
        os << setw(-W) << (count++) << setw(W) << *p.dataset[i] << endl; 
        // i == (p.n - 1) ? os << "" : os << ", ";
      }
      os << endl << "[" << p.n << " rows x " << p.d << " columns]" << endl;
      return os;
    }
};

#endif