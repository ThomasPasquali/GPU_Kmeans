python3 data_generator.py -n 1000 -d 3 -min 0 -max 10 -o 3Dpoints.csv
# TODO run k-means
python3 scatter_plot.py -f 3Dpoints.csv -d 3