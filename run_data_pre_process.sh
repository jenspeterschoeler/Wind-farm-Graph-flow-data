#!/bin/bash
#SBATCH --job-name=SPO-Pywake-Gen
#SBATCH --output=./logs/SPO_%A.log
#SBATCH --error=./logs/SPO_%A.err

# Which queue to use
#SBATCH --partition=windfatq

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=infinite


START_TIME=`date +%s`
NODE_ID=$(scontrol show hostnames $SLURM_JOB_NODELIST)

export LC_ALL=en_US.UTF-8


echo ------------------------------------------------------
echo Date: $(date)
echo Sophia job is running on node: ${NODE_ID}
echo Sophia job identifier: $SLURM_JOBID
echo ------------------------------------------------------ 


source ~/.bashrc
micromamba activate ml_GPU
python3 pre_process.py


END_TIME=`date +%s`
echo ------------------------------------------------------
echo Finished job
echo "Elapsed time: $(($END_TIME-$START_TIME)) seconds"
echo ------------------------------------------------------
