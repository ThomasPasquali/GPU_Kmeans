#!/bin/bash
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=kmeans-test
#SBATCH --output=test.out
#SBATCH --error=test.err

D=2
N=100000
K=4
MAX_ITER=500
TOL=0.00001
SEED=0
RUNS=5

BASEDIR='/home/stefano.daroit/comparisons/'
DATA_PATH="${BASEDIR}datasets/datasets-unzipped/"
DATASET="N100K/N100K_D${D}.csv"
PY="${BASEDIR}sklearn_Kmeans/env/bin/python3"

module load cuda

echo -e "\n============GPU============\n"
srun "${BASEDIR}GPU_Kmeans/build/src/bin/gpukmeans"     -d $D -n $N -k $K -m $MAX_ITER -o res_gpu.csv     -i "${DATA_PATH}${DATASET}" -t $TOL -s $SEED -r $RUNS

echo -e "\n==========GPU_MTX==========\n"
srun "${BASEDIR}GPU_Kmeans_mtx/build/src/bin/gpukmeans" -d $D -n $N -k $K -m $MAX_ITER -o res_gpu_mtx.csv -i "${DATA_PATH}${DATASET}" -t $TOL -s $SEED -r $RUNS

echo -e "\n============CPU============\n"
srun "${BASEDIR}CPU_Kmeans/bin/kmeans"                  -d $D -n $N -k $K -m $MAX_ITER -o res_cpu.csv     -i "${DATA_PATH}${DATASET}" -t $TOL -s $SEED -r $RUNS

echo -e "\n==========PYTHON===========\n"
$PY  "${BASEDIR}sklearn_Kmeans/sklearn_Kmeans.py"       -d $D -n $N -k $K -m $MAX_ITER -o res_sk.csv      -i "${DATA_PATH}${DATASET}" -t $TOL -s $SEED -r $RUNS