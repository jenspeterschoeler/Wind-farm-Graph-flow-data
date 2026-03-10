#!/bin/bash
#SBATCH --job-name=awf_convert
#SBATCH --output=./logs/awf_convert_%A.log
#SBATCH --error=./logs/awf_convert_%A.err

# Which queue to use
#SBATCH --partition=windq,workq,windfatq,fatq

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=48:00:00


START_TIME=`date +%s`
NODE_ID=$(scontrol show hostnames $SLURM_JOB_NODELIST)

export LC_ALL=en_US.UTF-8


echo ------------------------------------------------------
echo Date: $(date)
echo Sophia job is running on node: ${NODE_ID}
echo Sophia job identifier: $SLURM_JOBID
echo ------------------------------------------------------


echo "=== Environment Setup ==="
export PIXI_PROJECT_ROOT=/work/users/jpsch/gno
echo "PIXI_PROJECT_ROOT=$PIXI_PROJECT_ROOT"

# Activate cluster environment from main GNO repo (not the submodule's pixi.toml)
eval "$(pixi shell-hook --manifest-path ${PIXI_PROJECT_ROOT}/pixi.toml -e cluster)"
echo "Pixi environment activated"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Run AWF conversion with default arguments
# This will:
# 1. Convert all layouts from awf_database.nc to graph format
# 2. Save to data/awf_graphs/
# 3. Preprocess with 60/20/20 train/val/test split
# 4. Compute statistics and apply min-max scaling
python convert_awf_to_graphs.py \
    --database data/awf_database.nc \
    --output data/awf_graphs

# Alternative: Specify custom parameters
# python convert_awf_to_graphs.py \
#     --database data/awf_database.nc \
#     --output data/awf_graphs \
#     --max-layouts 100 \
#     --x-upstream-D 10.0 \
#     --y-margin-D 5.0 \
#     --train-size 0.6 \
#     --val-size 0.2 \
#     --test-size 0.2


END_TIME=`date +%s`
echo ------------------------------------------------------
echo Finished job
echo "Elapsed time: $(($END_TIME-$START_TIME)) seconds"
echo ------------------------------------------------------
