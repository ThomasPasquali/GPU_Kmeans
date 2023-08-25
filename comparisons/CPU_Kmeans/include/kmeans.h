#ifndef __KMEANS__
#define __KMEANS__

#include <random>
#include "point.hpp"

#define DATA_TYPE float

class Kmeans {
  private:
    const size_t n;
    const uint32_t d, k;
    const float tol;
    const uint64_t POINTS_BYTES;
    const uint64_t CENTROIDS_BYTES;
    Point<DATA_TYPE>** points;
    mt19937* generator;

    DATA_TYPE* h_points;
    DATA_TYPE* h_centroids;
    DATA_TYPE* h_last_centroids;
    uint32_t*  points_clusters;

    void init_centroids();
    void compute_distances(DATA_TYPE* distances);
    void clusters_argmin(const DATA_TYPE *distances, uint32_t *points_clusters, uint32_t *clusters_len);
    void compute_centroids(const uint32_t *clusters_len);
    bool cmp_centroids();

  public:
    Kmeans(const size_t n, const uint32_t d, const uint32_t k, const float tol, const int *seed, Point<DATA_TYPE>** points);
    ~Kmeans();
    uint64_t run(uint64_t maxiter);
};

#endif