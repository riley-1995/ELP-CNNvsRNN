#!/bin/bash
#SBATCH --job-name=inspect_tfrecord
#SBATCH --account=cso100
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:01:00
#SBATCH --output=slurm_logs/inspect_record.o%j

module load singularitypro/3.11

time -p singularity exec --bind "$PWD":/mnt --nv ../sandbox python -u /mnt/inspect_record.py
