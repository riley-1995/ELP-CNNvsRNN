#!/bin/bash
#SBATCH --job-name=test_underfitting
#SBATCH --partition=gpu-debug
#SBATCH --account=cso100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/test_underfitting.o%j

module load singularitypro/3.11

time -p singularity exec --bind "$PWD":/mnt,/expanse,/scratch --nv ../sandbox python -u /mnt/test_underfitting.py
