#ifndef __KMEANS__
#define __KMEANS__

#include "../include/common.h"
#include "../include/point.hpp"

/**
 * @brief 
 * 0: compute_distances_one_point_per_warp
 * 1: compute_distances_shmem 
 */
#define COMPUTE_DISTANCES_KERNEL 0

class Kmeans {
  private:
    const size_t n;
    const unsigned int d, k;
    const uint64_t POINTS_BYTES, CENTERS_BYTES;
    Point<DATA_TYPE>** points;

    DATA_TYPE* h_points;
    DATA_TYPE* h_centers;
    DATA_TYPE* h_last_centers;
    uint32_t*  h_points_clusters;
    DATA_TYPE* d_points;
    DATA_TYPE* d_centers;

    cudaDeviceProp* deviceProps;

    /**
     * @brief Select k random centers sampled form points
     */
    void initCenters(Point<DATA_TYPE>** points);
    bool cmpCenters();

  public:
    Kmeans(size_t n, unsigned int d, unsigned int k, Point<DATA_TYPE>** points, cudaDeviceProp* deviceProps);
    ~Kmeans();
    
    /**
     * @brief 
     * Notice: once finished will set clusters on each of Point<DATA_TYPE> of points passed in contructor
     * @param maxiter 
     * @return iter at which k-means converged
     * @return maxiter if did not converge
     */
    uint64_t run(uint64_t maxiter);
    void to_csv(ostream& o, char separator = ',');
};

#endif