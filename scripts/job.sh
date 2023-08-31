#!/bin/bash
#SBATCH --partition=edu5
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --job-name=kmeans-test
#SBATCH --output=test-%j.out
#SBATCH --error=test-%j.err

FEATURES=( 2 3 10 35 256 )
SAMPLES=( 100 1000 100000 )
K=4
MAX_ITER=500
TOL=0.00001
SEED=0
RUNS=5
OUT="res.csv"

module load cuda
source "./env.sh"

DATA_PATH="${BASEDIR}datasets/datasets-unzipped/"
DATASETS=( "N100" "N1000" "N100K" )

echo "############## ${TESTNAME} ##############"

S_IDX=0
for SAMPLE in ${SAMPLES[@]}; do
  DATA_DIR="${DATA_PATH}${DATASETS[${S_IDX}]}/${DATASETS[${S_IDX}]}" && ((S_IDX += 1))
  for FEATURE in ${FEATURES[@]}; do
    INPUT="${DATA_DIR}_D${FEATURE}.csv"
    #K=$(( SAMPLE / 10 ))
    echo "Test N=${SAMPLE} D=${FEATURE} K=${K} INPUT=$(basename "${INPUT}") MAXITER=${MAX_ITER} TOL=${TOL} SEED=${SEED}"
    eval "${CMD} -d ${FEATURE} -n ${SAMPLE} -k ${K} -m ${MAX_ITER} -o ${OUT} -i ${INPUT} -t ${TOL} -s ${SEED} -r ${RUNS}"
    echo ""
  done
done