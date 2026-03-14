#!/bin/bash -l
#$ -N OPT
#$ -P Gold
#$ -A UCL_chemM_Butler
#$ -t 1-24
#$ -pe mpi 24
#$ -l h_rt=48:00:00
#$ -l mem=4G
#$ -cwd
#$ -o logs/$JOB_NAME.$JOB_ID.$TASK_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.$TASK_ID.err

module purge
module load vasp/6.3.0-24Jan2022/intel-2019-update5

BASE_DIR=$(pwd)
CASE_DIR=$(printf "case%02d" $SGE_TASK_ID)

echo "Running case: $CASE_DIR"
echo "Host: $(hostname)"
echo "Start: $(date)"

cd "$BASE_DIR/$CASE_DIR" || exit 1
gerun vasp_std > vasp.out

echo "Finish: $(date)"
