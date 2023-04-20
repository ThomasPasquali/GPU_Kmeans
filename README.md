# GPU_Kmeans

## Python setup

```bash
pip install plotly
pip install argparse
```

## Data generator

```bash
python3 data_generator.py --help
python3 data_generator.py -n 1000 -d 3 -min 0 -max 10 -o 3Dpoints.csv
```

## Plots

```bash
python3 scatter_plot.py -f 3Dpoints.csv -d 3
python3 data_generator.py -n 100 -d 3 -min 0 -max 10 | python3 scatter_plot.py -d 3
```

## Input parser 

```bash
cat 3Dpoints.csv | ./bin/src/main -d 3 -n 1000
```