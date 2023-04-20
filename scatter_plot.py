import plotly.express as px
import pandas as pd
import argparse
import sys

argParser = argparse.ArgumentParser()
argParser.add_argument("-d", "--dimensions", help="The number of dimensions of the points", type=int, choices=[2,3], required=True)
argParser.add_argument("-f", "--input-file", help="The CSV file to read from", type=str, required=False, default=None)
args = argParser.parse_args()

df = pd.read_csv(sys.stdin if args.input_file == None else args.input_file)

if args.dimensions == 2:
  fig = px.scatter(df, x=df.columns[1], y=df.columns[2], color='cluster')
elif args.dimensions == 3:
  fig = px.scatter_3d(df, x=df.columns[1], y=df.columns[2], z=df.columns[3], color='cluster')

print(df)
fig.show()