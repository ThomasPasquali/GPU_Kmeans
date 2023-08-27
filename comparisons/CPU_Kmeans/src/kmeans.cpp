#include <stdio.h>
#include <string.h>
#include <vector>
#include <iomanip>
#include <cmath>
#include <limits>

#include "../include/kmeans.h"

using namespace std;

#define DEBUG 0

const DATA_TYPE INFNTY = numeric_limits<DATA_TYPE>::infinity();

Kmeans::Kmeans (const size_t _n, const uint32_t _d, const uint32_t _k, const float _tol, const int *seed, Point<DATA_TYPE>** _points) :
  n(_n), d(_d), k(_k), tol(_tol),
  POINTS_BYTES(_n * _d * sizeof(DATA_TYPE)),
  CENTROIDS_BYTES(_k * _d * sizeof(DATA_TYPE)),
  points(_points) {

  if (seed) {
    seed_seq s{*seed};
    generator = new mt19937(s);
  }
  else {
    random_device rd;
    generator = new mt19937(rd());
  }

  h_points = new DATA_TYPE[POINTS_BYTES];
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < d; ++j) {
      h_points[i * d + j] = _points[i]->get(j);
    }
  }

  init_centroids();
}

Kmeans::~Kmeans () {
  delete   generator;
  delete[] h_points;
  delete[] points_clusters;
  delete[] h_centroids;
  delete[] h_last_centroids;
}

void Kmeans::init_centroids () {
  uniform_int_distribution<int> random_int(0, n - 1);

  h_centroids = new DATA_TYPE[CENTROIDS_BYTES];
  h_last_centroids = new DATA_TYPE[CENTROIDS_BYTES];

  unsigned int i = 0;
  vector<Point<DATA_TYPE>*> usedPoints;
  Point<DATA_TYPE>* centroids[k];
  while (i < k) {
    Point<DATA_TYPE>* p = points[random_int(*generator)];
    bool found = false;
    for (auto p1 : usedPoints) {
      if ((*p1) == (*p)) {
        found = true;
        break;
      }
    }
    if (!found) {
      centroids[i] = new Point<DATA_TYPE>(p);
      usedPoints.push_back(p);
      ++i;
    }
  }

  // cout << endl << "Centroids" << endl;
  // for (i = 0; i < k; ++i)
  //   cout << *(centroids[i]) << endl;

  for (size_t i = 0; i < k; ++i) {
    for (size_t j = 0; j < d; ++j) {
      h_centroids[i * d + j] = centroids[i]->get(j);
    }
  }
  memcpy(h_last_centroids, h_centroids, CENTROIDS_BYTES);
}

void Kmeans::compute_distances(DATA_TYPE* distances) {
  for (uint32_t ni = 0; ni < n; ++ni) {
    for (uint32_t ki = 0; ki < k; ++ki) {
      DATA_TYPE dist = 0, tmp;
      for (uint32_t di = 0; di < d; ++di) {
        tmp = h_points[ni * d + di] - h_centroids[ki * d + di];
        dist += tmp * tmp;
      }
      distances[ni * k + ki] = dist;
    }
  }

  if (DEBUG) {
    printf("\nDISTANCES DEBUG\n");
    for (uint32_t i = 0; i < n; ++i)
      for (uint32_t j = 0; j < k; ++j)
        printf("N=%-2u K=%-2u -> DIST=%.3f \n", i, j, distances[i * k + j]);
  }
}

void Kmeans::clusters_argmin(const DATA_TYPE *distances, uint32_t *points_clusters, uint32_t *clusters_len) {
  memset(clusters_len, 0, k * sizeof(uint32_t));
  for (uint32_t i = 0; i < n; i++) {
    DATA_TYPE min = INFNTY;
    uint32_t idx = 0;
    for (uint32_t j = 0, ii = i * k; j < k; j++, ii++) {
      if (distances[ii] < min) {
        min = distances[ii];
        idx = j;
      }
    }
    points_clusters[i] = idx;
    clusters_len[idx]++;
  }

  if (DEBUG) {
    printf("\nARGMIN DEBUG\n");
    for (uint32_t i = 0; i < n; ++i)
      printf("N=%-2u K=%-2u \n", i, points_clusters[i]);
    printf("\n");
    for (uint32_t i = 0; i < k; ++i)
      printf("K=%-2u N=%-2u \n", i, clusters_len[i]);
  }
}

void Kmeans::compute_centroids(const uint32_t *clusters_len) {
  memset(h_centroids, 0, k * d * sizeof(DATA_TYPE));
  for (uint32_t i = 0; i < n; ++i) {
    for (uint32_t j = 0; j < d; ++j) {
      h_centroids[points_clusters[i] * d + j] += h_points[i * d + j];
    }
  }

  for (uint32_t i = 0; i < k; ++i) {
    for (uint32_t j = 0; j < d; ++j) {
      uint32_t count = clusters_len[i] > 1 ? clusters_len[i] : 1;
      DATA_TYPE scale = 1.0 / ((double) count);
      h_centroids[i * d + j] *= scale;
    }
  }

  if (DEBUG) {
    printf("\nCENTROIDS DEBUG\n");
    for (uint32_t i = 0; i < k; ++i) {
      for (uint32_t j = 0; j < d; ++j)
        printf("%.3f, ", h_centroids[i * d + j]);
      printf("\n");
    }
  }
}

uint64_t Kmeans::run (uint64_t maxiter) {
  uint64_t converged = maxiter, iter = 0;

  DATA_TYPE* distances = new DATA_TYPE[n * k];
  points_clusters = new uint32_t[n];
  uint32_t* clusters_len = new uint32_t[k];

  /* MAIN LOOP */
  while (iter++ < maxiter) {
    compute_distances(distances);
    clusters_argmin(distances, points_clusters, clusters_len);
    compute_centroids(clusters_len);

    if (iter > 1 && cmp_centroids()) {
      converged = iter;
      break;
    }

    memcpy(h_last_centroids, h_centroids, CENTROIDS_BYTES);
  }
  /* MAIN LOOP END */

  /* COPY BACK RESULTS*/
  for (size_t i = 0; i < n; i++) {
    points[i]->setCluster(points_clusters[i]);
  }

  delete[] distances;
  delete[] clusters_len;

  return converged;
}

bool Kmeans::cmp_centroids () {
  for (size_t i = 0; i < k; ++i) {
    DATA_TYPE dist_sum = 0;
    for (size_t j = 0; j < d; ++j) {
      DATA_TYPE dist = h_centroids[i * d + j] - h_last_centroids[i * d + j];
      dist_sum += dist * dist;
    }
    if (sqrt(dist_sum) > tol) { return false; }
  }
  return true;
}