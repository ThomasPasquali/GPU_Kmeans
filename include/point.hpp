#ifndef __POINT_H__
#define __POINT_H__

#include <iostream>
#include <iomanip>

using namespace std;

template <typename T> 
class Point {

  private:
    T *x;
    int d;
    int cluster;
  
  public:
    Point() {
      this->x = NULL;
      this->d = 0;
      this->cluster = 0;
    }

    Point(T *_x, int _d): cluster(0) {
      this->x = new T[_d];
      this->d = _d;
      
      for (int i = 0; i < _d; i++) {
        x[i] = _x[i];
      }
    }

    Point(T *_x, int _d, int _cluster): cluster(_cluster) {
      this(_x, _d);
    }

    int getCluster () { return cluster; }

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
      os << setw(9) << p.cluster;
      for (int i = 0; i < p.d; i++) {
        os << setw(9) << setprecision(5) << p.x[i];
      }
      return os;
    }
};

#endif