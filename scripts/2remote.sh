#/bin/bash

BUILD_SCRIPT='build.sh'
JOB_SCRIPT='job.sh'
JOB_OUT='kmeans'
OUTPUT='res.csv'
TOCOPY=( 'datasets' 'src' 'tests' 'CMakeLists.txt' "scripts/${BUILD_SCRIPT}" "scripts/${JOB_SCRIPT}" )

if [[ $# < 2 ]]; then 
  echo "Usage: 2remote.sh remote_user remote_folder"
  exit 1;
fi

HOST="${1}@marzola.disi.unitn.it"
WORKDIR="~/${2}"

ssh $HOST "rm -f -r ${WORKDIR} && mkdir ${WORKDIR}"

for FILE in ${TOCOPY[@]}; do
  scp -r ./$FILE $HOST:$WORKDIR/
done

ssh $HOST "cd ${WORKDIR} && ./$BUILD_SCRIPT"
ssh $HOST "cd ${WORKDIR} && sbatch ${JOB_SCRIPT}"
