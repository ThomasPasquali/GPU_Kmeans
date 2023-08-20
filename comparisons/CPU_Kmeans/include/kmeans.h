#ifndef __KMEANS__
#define __KMEANS__

#include "point.hpp"

#define DATA_TYPE float

class Kmeans {
  private:
    const size_t n;
    const unsigned int d, k;
    const float tol;
    const uint64_t POINTS_BYTES;
    const uint64_t CENTROIDS_BYTES;
    Point<DATA_TYPE>** points;

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
    Kmeans(size_t n, unsigned int d, unsigned int k, const float tol, Point<DATA_TYPE>** points);
    ~Kmeans();
    uint64_t run(uint64_t maxiter);
    void to_csv(ostream& o, char separator = ',');
};

#endif