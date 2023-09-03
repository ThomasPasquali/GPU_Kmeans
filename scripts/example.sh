#!/bin/bash

./scripts/build.sh && \
./build/src/bin/gpukmeans -d 2 -n 100000 -k 4 -m 300 -o res_s0.csv -i ./datasets/N100K_D2.csv -s 0 -t 0.00001 && \
./build/src/bin/gpukmeans -d 2 -n 100000 -k 4 -m 300 -o res_s5.csv -i ./datasets/N100K_D2.csv -s 5 -t 0.00001 && \
python3 ./py_utils/scatter_plot.py -d 2 -f res_s0.csv res_s5.csv
