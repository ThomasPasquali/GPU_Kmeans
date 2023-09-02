# Setup

## Python
Needed for python utils like `data_generator` and `scatter_plot`.

```bash
pip install -r py_utils/requirements.txt
```

## Catch2
Unit testing for C++.

```bash
git clone https://github.com/catchorg/Catch2.git
```

# Python utils

## Data generator
This script can be used to generate random datasets. If attribute `-k` is specified the dataset contains clusterized points
```bash
python3 py_utils/data_generator.py -n 1000 -d 3 -k 4 -min 0 -max 10 -o datasets/3Dpoints.csv
```

## Plots
This script can be used to display results (reading csv output files).

```bash
python3 py_utils/scatter_plot.py -f 3Dpoints.csv -d 3
```

# GPU Kmeans
## Compile
These are the commands that can be used to compile GPU Kmeans.

*Notice: cmake is required.*
```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```
This will build the target `gpukmeans` to `build/src/bin/gpukmeans`.

This code has is provided in `scripts/build.sh` so that (from the root of the project) you can run `./scripts/build.sh` to build `gpukmeans`.

### Tests
The target `unit_kernels` is excluded from `ALL_TARGETS` therefore it has to be compiled explicitly.
```bash
cmake --build . --target unit_kernels
```
Use the script `scripts/run_tests.sh` to build and run unit tests.

## Usage
The executable `gpukmeans` reads inputs from `stdin` or from csv file. The directory `datasets` contains some examples of data input.
```bash
./build/src/bin/gpukmeans --help
gpukmeans is an implementation of the K-means algorithm that uses a GPU
Usage:
  gpukmeans [OPTION...]

  -h, --help              Print usage
  -d, --n-dimensions arg  Number of dimensions of a point
  -n, --n-samples arg     Number of points
  -k, --n-clusters arg    Number of clusters
  -m, --maxiter arg       Maximum number of iterations
  -o, --out-file arg      Output filename
  -i, --in-file arg       Input filename
  -r, --runs arg          Number of k-means runs (default: 1)
  -s, --seed arg          Seed for centroids generator
  -t, --tolerance arg     Tolerance to declare convergence
```

Example:
```bash
./build/src/bin/gpukmeans -d 64 -n 1797 -k 10 -m 300 -o res.csv -i ./datasets/N1797_D64_digits-sklearn.csv -t 0.0001
```

Runs `gpukmeans` on the dataset `N1797_D64_digits-sklearn.csv` considering: 64 dimensions, 1797 points, 10 clusters, 300 maximum iterations, tolerance 1e-4, and output the results to `res.csv`.