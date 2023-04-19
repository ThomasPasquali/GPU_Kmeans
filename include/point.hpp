#ifndef __POINT_H__
#define __POINT_H__

#include <iostream>

using namespace std;

template <typename T> 
class Point {

  private:
    T *x;
    int d; 
  
  public:
    Point() {
      this->x = NULL;
      this->d = 0;
    }

    Point(T *_x, int _d) {
      this->x = new T[_d];
      this->d = _d;
      
      for (int i = 0; i < _d; i++) {
        x[i] = _x[i];
      }
    }

    ~Point() {
      delete[] x;
    }

    Point& operator= (Point const &p) {
      if (x) { delete[] x; }
      x = new T[p.d];
      
      for (int i = 0; i < p.d; i++) {
        x[i] = p.x[i];
      }
      d = p.d;

      return *this;
    }

    friend ostream& operator<< (ostream &os, Point const &p) {
      os << "(";
      for (int i = 0; i < p.d; i++) {
        os << p.x[i];
        i == (p.d - 1) ? os << "" : os << ", ";
      }
      os << ")";
      return os;
    }
};

#endif