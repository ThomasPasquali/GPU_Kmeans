#/bin/bash

JOB_SCRIPT='./scripts/job.sh'
ENV_SCRIPT='./scripts/env.sh'

remote_copy () {
  local DST=$1
  local FILES=${@:2}

  ssh $HOST "mkdir -p ${DST}"
  for FILE in ${FILES[@]}; do
    scp -r $FILE $HOST:$DST/
  done
}

write_env_file () {
  echo "## ENVIRONMENT VARS ##" > "${ENV_SCRIPT}"
  for VAR in $@; do
    echo "${VAR//^/ }" >> "${ENV_SCRIPT}"
  done
}

if [[ $# < 1 ]]; then
  echo "Usage: 2remote.sh remote_user remote_folder"
  exit 1;
fi

HOST="${1}@marzola.disi.unitn.it"
WORKDIR="~/comparisons/"

######## GPU KMEANS ########
# SRC_DIR='./'
# REMOTE_DIR="${WORKDIR}gpukmeans"
# EXEC="build/src/bin/gpukmeans"
# BUILD_SCRIPT='./scripts/build.sh'
# TOCOPY=( 'src' 'tests' 'CMakeLists.txt' )

# write_env_file "TESTNAME='GPU-KMEANS'" "BASEDIR=\"${WORKDIR}\"" "CMD=\"${REMOTE_DIR}/${EXEC}\""
# remote_copy $REMOTE_DIR ${TOCOPY[@]/#/$SRC_DIR} "${JOB_SCRIPT}" "${ENV_SCRIPT}" "${BUILD_SCRIPT}"

# ssh $HOST "cd ${REMOTE_DIR} && ./$(basename "${BUILD_SCRIPT}")"
# ssh $HOST "cd ${REMOTE_DIR} && sbatch $(basename "${JOB_SCRIPT}")"

######## CPU KMEANS ########
# SRC_DIR='./comparisons/CPU_Kmeans/'
# REMOTE_DIR="${WORKDIR}gpukmeans/comp/cpukmeans"
# EXEC="bin/kmeans"
# TOCOPY=( 'src' 'include' 'makefile' )

# write_env_file "TESTNAME='CPU-KMEANS'" "BASEDIR=\"${WORKDIR}\"" "CMD=\"${REMOTE_DIR}/${EXEC}\""
# remote_copy $REMOTE_DIR ${TOCOPY[@]/#/$SRC_DIR} "${JOB_SCRIPT}" "${ENV_SCRIPT}"

# ssh $HOST "cd ${REMOTE_DIR} && make"
# ssh $HOST "cd ${REMOTE_DIR} && sbatch $(basename "${JOB_SCRIPT}")"

######## SKLEARN PYTHON ########
# SRC_DIR="./comparisons/CPU_python/"
# REMOTE_DIR="${WORKDIR}sklearn_Kmeans"
# EXEC="sklearn_kmeans.py"
# TOCOPY=( 'requirements.txt' "${EXEC}" )

# write_env_file "TESTNAME='CPU-PYTHON'" "BASEDIR=\"${WORKDIR}\"" "CMD=\"${REMOTE_DIR}/env/bin/python3^${REMOTE_DIR}/${EXEC}\""
# remote_copy $REMOTE_DIR ${TOCOPY[@]/#/$SRC_DIR} "${JOB_SCRIPT}" "${ENV_SCRIPT}"

# ssh $HOST "cd ${REMOTE_DIR} && \
#            [ ! -d "env" ] && \
#            python3 -m venv env && \
#            source ./env/bin/activate && \
#            pip install -r requirements.txt"

# ssh $HOST "cd ${REMOTE_DIR} && sbatch $(basename "${JOB_SCRIPT}")"

######## KCUDA ########
# SRC_DIR='./comparisons/GPU_CUDA/'
# REMOTE_DIR="${WORKDIR}gpukmeans/comp/kmcuda"
# REPO_LINK='https://github.com/src-d/kmcuda.git'
# EXEC="bin/kmcuda"
# TOCOPY=( 'kmcuda.cpp' 'makefile' )

# write_env_file "TESTNAME='GPU-CUDA'" "BASEDIR=\"${WORKDIR}\"" "CMD=\"${REMOTE_DIR}/${EXEC}\""
# remote_copy $REMOTE_DIR ${TOCOPY[@]/#/$SRC_DIR} "${JOB_SCRIPT}" "${ENV_SCRIPT}"

# ssh $HOST "cd ${REMOTE_DIR} && git clone ${REPO_LINK} && make"
# ssh $HOST "cd ${REMOTE_DIR} && make"
# ssh $HOST "cd ${REMOTE_DIR} && sbatch $(basename "${JOB_SCRIPT}")"



rm $ENV_SCRIPT