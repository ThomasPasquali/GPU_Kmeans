# GPU_Kmeans

## Python setup

```bash
pip install plotly
pip install argparse
pip install scikit-learn
```

## Data generator

```bash
python3 py_utils/data_generator.py -n 1000 -d 3 -min 0 -max 10 -o datasets/3Dpoints.csv
```

## Plots

```bash
python3 py_utils/scatter_plot.py -f 3Dpoints.csv -d 3
python3 py_utils/data_generator.py -n 100 -d 3 -min 0 -max 10 | python3 py_utils/scatter_plot.py -d 3
```

## K-means

```bashmake clean all && cat datasets/3Dpoints.csv | ./bin/src/main -d 3 -n 1000 -k 20 -m 1000 -o out/GPU_Kmeans/res.csv
cat datasets/A_N3000_D2_K20.csv | ./bin/src/main -d 2 -n 3000 -k 20 -m 1000 -o out/GPU_Kmeans/A_N3000_D2_K20.csv
```

Build and run our program on `3Dpoints.csv` considering: 3 dimensions, 10 points, 4 clusters, 1000 maximum iterations and output results on `res.csv` file

## Comparisons

### GPU-python

```bash
python3 comparisons/GPU_python.py -d 2 -k 20 -f datasets/A_N3000_D2_K20.csv -o out/GPU_python/A_N3000_D2_K20.csv
```

## All together

Parameters (in order):
- n (number of points)
- d (dimensionality)
- k (number of clusters)
- m (max iterations)
- f (dataset filename)

```bash
./runComparison.sh 3000 2 20 100 datasets/A_N3000_D2_K20.csv
```

Compiles (if needed) and runs `GPU_Kmeans`, `GPU_python` on a given dataset end then shows a comparison of the results

<!-- norma(x-y)<= norma(x) dice che x==y) -->