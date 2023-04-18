import pandas as pd
import argparse
import random

argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--n-samples", help="The number of points generated", type=int, required=False, default=1000)
argParser.add_argument("-d", "--dimensions", help="The number of dimensions of the points", type=int, required=False, default=3)
argParser.add_argument("-min", "--min-value", help="Lower bound for the generated values", type=float, required=False, default=-100)
argParser.add_argument("-max", "--max-value", help="Upper bound for the generated values", type=float, required=False, default=100)
argParser.add_argument("-s", "--seed", help="The seed to user for rng", type=int, required=False, default=None)
argParser.add_argument("-o", "--out-filename", help="The file to write to", type=str, required=False, default=None)
args = argParser.parse_args()

random.seed(args.seed)

if args.min_value > args.max_value:
  tmp = args.min_value
  args.min_value = args.max_value
  args.max_value = tmp

# print("args=%s" % args)

data = [] # np.random.rand(args.n_samples, args.dimensions)
for i in range(args.n_samples):
  data.append([0] + [random.uniform(args.min_value, args.max_value) for j in range(args.dimensions)])

columns = ['cluster'] + [f"d{i}" for i in range(args.dimensions)]
df = pd.DataFrame(data, columns=columns)

if args.out_filename == None:
  from io import StringIO
  output = StringIO()
  df.to_csv(output, index=False)
  print(output.getvalue())
else:
  df.to_csv(args.out_filename, index=False)