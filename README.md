# Setup

## Python
Needed for python utils like `data_generator` and `scatter_plot`.

```bash
pip install plotly argparse scikit-learn
```

## Catch2
Unit testing for C++.

```bash
git clone https://github.com/catchorg/Catch2.git
```

# Python utils

## Data generator
This script can be used to generate random datasets.
```bash
python3 py_utils/data_generator.py -n 1000 -d 3 -min 0 -max 10 -o datasets/3Dpoints.csv
```

## Plots
This script can be used to display results (reading csv output files).

```bash
python3 py_utils/scatter_plot.py -f 3Dpoints.csv -d 3
python3 py_utils/data_generator.py -n 100 -d 3 -min 0 -max 10 | python3 py_utils/scatter_plot.py -d 3
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
The target `unit_kernels` is excluded from `ALL_TARGETS` therefore it has to be compiled explicitely.
```bash
cmake --build . --target unit_kernels
```
Use the script `scripts/run_tests.sh` to build and run unit tests.

## Usage
The executable `gpukmeans` reads inputs from `stdin`.
```bash
./build/src/bin/gpukmeans --help
gpukmeans is an implementation of the K-means algorithm that uses a GPU
Usage:
  gpukmeans [OPTION...]

  -h, --help            Print usage
  -d, --dimensions arg  Number of dimensions of a point
  -n, --n-samples arg   Number of points
  -k, --clusters arg    Number of clusters
  -m, --maxiter arg     Maximum number of iterations
  -o, --out-file arg    Output filename
```

Example:
```bash
cat datasets/A_N3000_D2_K20.csv | ./build/src/bin/gpukmeans -d 2 -n 3000 -k 20 -m 1000 -o out/GPU_Kmeans/A_N3000_D2_K20.csv
```

Runs `gpukmeans` on the dataset `A_N3000_D2_K20.csv` considering: 2 dimensions, 3000 points, 20 clusters, 1000 maximum iterations and output results to `out/GPU_Kmeans/A_N3000_D2_K20.csv`.

# Other implementations
The following are other programs that implement the K-means algorithm:

### GPU-python

```bash
python3 comparisons/GPU_python.py -d 2 -k 20 -f datasets/A_N3000_D2_K20.csv -o out/GPU_python/A_N3000_D2_K20.csv
```

# Run comparison
We provided a script `run_comparison` to compare the results of different implementations of the algorithm.

Parameters (in order):
- n (number of points)
- d (dimensionality)
- k (number of clusters)
- m (max iterations)
- f (dataset filename)

```bash
./scripts/run_comparison.sh 3000 2 20 100 datasets/A_N3000_D2_K20.csv
```

Compiles (if needed) and runs `GPU_Kmeans`, `GPU_python` on a given dataset end then shows a comparison of the results.