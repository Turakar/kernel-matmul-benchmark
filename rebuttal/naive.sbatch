#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --job-name=rebuttal-naive
#SBATCH --cpus-per-task=28
#SBATCH --output=rebuttal/slurm-output/%J.txt
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --qos=short
#SBATCH --partition=KathleenG

echo "Job started at $(date)"

module purge
module load GCCcore/12.2.0
module load Python/3.11.2-GCCcore-12.2.0
module load libffi/3.4.4-GCCcore-12.2.0
module load CUDA/12.3.0
module load Ninja/1.11.1-GCCcore-12.2.0
module load SWIG/4.1.1-GCCcore-12.2.0

poetry run python rebuttal/naive.py \
    --dataset traffic_hourly \
    --hpo-index 0

echo "Job finished at $(date)"
