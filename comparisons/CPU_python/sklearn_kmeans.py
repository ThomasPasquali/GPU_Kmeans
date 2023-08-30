from sklearn.cluster import KMeans
import pandas as pd
import argparse
import sys
import time

argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--n-samples",    help="The number of points",            type=int,   required=True)
argParser.add_argument("-d", "--dimensions",   help="The number of dimensions",        type=int,   required=True)
argParser.add_argument("-k", "--clusters",     help="The number of clusters",          type=int,   required=True)
argParser.add_argument("-m", "--maxiter",      help="The maximun nuber of iterations", type=int,   required=True)
argParser.add_argument("-i", "--input-file",   help="The CSV file to read from",       type=str,   required=False, default=None)
argParser.add_argument("-o", "--out-filename", help="The file to write to",            type=str,   required=False, default=None)
argParser.add_argument('-r', '--runs',         help="Number of k-means runs",          type=int,   required=False, default=1)
argParser.add_argument('-s', '--seed',         help="Seed for centroids",              type=int,   required=False, default=None)
argParser.add_argument('-t', '--tol',          help="Tolerance for convergence",       type=float, required=False, default=1e-4)
args = argParser.parse_args()

df = pd.read_csv(sys.stdin if args.input_file == None else args.input_file, nrows=args.n_samples)
if df.shape[1] > args.dimensions:
  df.drop(columns=df.columns[args.dimensions - df.shape[1]:], axis=1, inplace=True)

kmeans = None
total_elapsed = 0
for i in range(args.runs):
  start = time.time()
  kmeans = KMeans(n_clusters=args.clusters, init='random', n_init=1, max_iter=args.maxiter, tol=args.tol, random_state=args.seed).fit(df)
  end = time.time()
  total_elapsed += end - start

  str = f"Time: {end - start}"
  if kmeans.n_iter_ < args.maxiter:
    print(f"K-means converged at iteration {kmeans.n_iter_} - {str}")
  else:
    print(f"K-means did NOT converge - {str}")

print(f'sklearn kmeans: {total_elapsed / args.runs}s ({args.runs} runs)')

# df['clusters'] = kmeans.labels_
# df = df[['clusters'] + [f"d{i}" for i in range(args.dimensions)]]

# if args.out_filename == None:
#   from io import StringIO
#   output = StringIO()
#   df.to_csv(output, index=False)
#   print(output.getvalue())
# else:
#   df.to_csv(args.out_filename, index=False)