#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --job-name=benchmark
#SBATCH --cpus-per-task=28
#SBATCH --output=benchmark-%J.txt
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=KathleenG
#SBATCH --gres=gpu:1
#SBATCH --qos=short

echo "Job started at $(date)"

module purge
module load GCCcore/12.2.0
module load Python/3.11.2-GCCcore-12.2.0
module load libffi/3.4.4-GCCcore-12.2.0
module load CUDA/12.3.0
module load Ninja/1.11.1-GCCcore-12.2.0
module load SWIG/4.1.1-GCCcore-12.2.0

poetry run python kernel_matmul_experiments/benchmark/cli.py \
    run \
    --path data/benchmark/kathleen-rebuttal

echo "Job finished at $(date)"
