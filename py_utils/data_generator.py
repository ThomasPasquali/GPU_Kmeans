import pandas as pd
import argparse
import torch
import random

RADIUS_CLUSTER = 0.1
ROUND_DIGITS = 5

argParser = argparse.ArgumentParser()
argParser.add_argument("-n",   "--n-samples",   help="The num of points generated",                type=int,   required=False, default=1000)
argParser.add_argument("-d",   "--dimensions",  help="The num of dimensions of the points",        type=int,   required=False, default=3)
argParser.add_argument("-k",   "--n-clusters",  help="The num of cluster (if output clusterized)", type=int,   required=False)
argParser.add_argument("-min", "--min-value",   help="Lower bound for the generated values",       type=float, required=False)
argParser.add_argument("-max", "--max-value",   help="Upper bound for the generated values",       type=float, required=False)
argParser.add_argument("-o",   "--out-dir",     help="The file to write to",                       type=str,   required=False)
args = argParser.parse_args()

def normalize(x):
  min = args.min_value if args.min_value != None else 0
  max = args.max_value if args.max_value != None else 1
  return x * (max - min) + min

df = None
# Clusterized points
if args.n_clusters != None:
  # Generate centroids
  centroids = torch.rand(args.n_clusters, args.dimensions).numpy()
  centroids = pd.DataFrame(centroids).applymap(normalize).to_numpy()

  # Choose cluster lengths
  clusters_len = []
  left = args.n_samples
  for i in range(args.n_clusters - 1):
    clusters_len.append(int(random.uniform(0, left / 2)))
    left -= clusters_len[i]
  clusters_len.append(left)

  # Generate points close to centroids
  data = []
  for i in range(args.n_clusters):
    for j in range(clusters_len[i]):
      data.append([(centroids[i][k] - random.uniform(-RADIUS_CLUSTER, RADIUS_CLUSTER)) for k in range(args.dimensions)])
# Uniform random points range [min, max)
elif args.min_value != None and args.max_value != None:
  data = torch.rand(args.n_samples, args.dimensions).numpy()
  data = pd.DataFrame(data).applymap(normalize).to_numpy()
# Std normal random points
else:
  data = torch.randn(args.n_samples, args.dimensions).numpy()

df = pd.DataFrame(data, columns=[f"d{i}" for i in range(args.dimensions)]).applymap( lambda x : round(x, ROUND_DIGITS))

dirname = args.out_dir if args.out_dir != None else "."
df.to_csv(f"{dirname}/N{args.n_samples}_D{args.dimensions}.csv", index=False)