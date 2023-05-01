from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import argparse
import sys

argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--dimensions", help="The number of dimensions of the points", type=int, required=True)
argParser.add_argument("-k", "--clusters", help="The number of clusters", type=int, required=True)
argParser.add_argument("-m", "--maxiter", help="The maximun nuber of iterations", type=int, required=True)
argParser.add_argument("-f", "--input-file", help="The CSV file to read from", type=str, required=False, default=None)
argParser.add_argument("-o", "--out-filename", help="The file to write to", type=str, required=False, default=None)
args = argParser.parse_args()

df = pd.read_csv(sys.stdin if args.input_file == None else args.input_file)

kmeans = KMeans(n_clusters=args.clusters, random_state=0, n_init="auto", max_iter=args.maxiter).fit(df)
df['clusters'] = kmeans.labels_
df = df[['clusters'] + [f"d{i}" for i in range(args.dimensions)]]

if args.out_filename == None:
  from io import StringIO
  output = StringIO()
  df.to_csv(output, index=False)
  print(output.getvalue())
else:
  df.to_csv(args.out_filename, index=False)
