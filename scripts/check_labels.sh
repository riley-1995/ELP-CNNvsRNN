#!/bin/bash
#SBATCH --job-name=check_labels
#SBATCH --partition=gpu-debug
#SBATCH --account=cso100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:01:00
#SBATCH --output=slurm_logs/check_labels.o%j

module load singularitypro/3.11

time -p singularity exec --bind /expanse,/scratch --nv ../sandbox python -u ./check_labels.py
