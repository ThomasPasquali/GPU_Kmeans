#ifndef __POINT_H__
#define __POINT_H__

#include <iostream>

using namespace std;

template <typename T> class Point {
  private:
    T *dims;
    int n; 
  
  public:
    Point() {
      this->dims = NULL;
      this->n = 0;
    }

    Point(T *points, int n) {
      this->dims = new T[n];
      this->n = n;
      
      for (int i = 0; i < n; i++) {
        dims[i] = points[i];
      }
    }

    ~Point() {
      delete[] dims;
    }

    Point& operator= (Point const &p) {
      if (dims) { delete[] dims; }
      dims = new T[p.n];
      
      for (int i = 0; i < p.n; i++) {
        dims[i] = p.dims[i];
      }
      n = p.n;

      return *this;
    }

    friend ostream& operator<< (ostream &os, Point const &p) {
      os << "(";
      for (int i = 0; i < p.n; i++) {
        os << p.dims[i];
        i == (p.n - 1) ? os << "" : os << ", ";
      }
      os << ")";
      return os;
    }
};

#endif