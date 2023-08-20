#!/bin/bash
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --job-name=kmeans
#SBATCH --output=kmeans.out
#SBATCH --error=kmeans.err

module load cuda

srun ./build/src/bin/gpukmeans -d 3 -n 100 -k 4 -m 1001 -o res.csv -i ./datasets/3Dpoints.csv
