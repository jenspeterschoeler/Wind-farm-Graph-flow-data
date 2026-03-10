#!/bin/bash
#SBATCH --job-name=TurbOPark-gen
#SBATCH --output=./logs/turbopark_%A.log
#SBATCH --error=./logs/turbopark_%A.err
#SBATCH --partition=windq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=infinite

#============================================================================
# TurbOPark Dataset Generation - Submission Script
#============================================================================
# Usage: sbatch submit_turbopark.sh
#
# Configure the dataset size by changing DATASET_CONFIG below:
#   - turbopark10_test : 10 layouts × 4 inflows = 40 graphs (quick test)
#   - turbopark250     : 250 layouts × 4 inflows = 1,000 graphs
#   - turbopark2500    : 2500 layouts × 4 inflows = 10,000 graphs (phase 1)
#============================================================================

# ===== USER CONFIGURATION =====
DATASET_CONFIG="turbopark2500"
# ==============================


START_TIME=`date +%s`
NODE_ID=$(scontrol show hostnames $SLURM_JOB_NODELIST)

export LC_ALL=en_US.UTF-8


echo "============================================================"
echo "TurbOPark Dataset Generation"
echo "============================================================"
echo "Date: $(date)"
echo "Node: ${NODE_ID}"
echo "Job ID: $SLURM_JOBID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Configuration: ${DATASET_CONFIG}"
echo "============================================================"


echo ""
echo "=== Environment Setup ==="
export PIXI_PROJECT_ROOT=/work/users/jpsch/gno
echo "PIXI_PROJECT_ROOT=$PIXI_PROJECT_ROOT"

# Activate cluster environment from main GNO repo
eval "$(pixi shell-hook --manifest-path ${PIXI_PROJECT_ROOT}/pixi.toml -e cluster)"
echo "Pixi environment activated"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Ensure logs directory exists
mkdir -p ./logs

echo ""
echo "=== Starting Dataset Generation ==="
python main.py --config ${DATASET_CONFIG}


END_TIME=`date +%s`
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "============================================================"
echo "Job Complete"
echo "Configuration: ${DATASET_CONFIG}"
echo "Elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED}s total)"
echo "============================================================"
