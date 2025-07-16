#!/usr/bin/env bash
#SBATCH --job-name=train
####Change account below
#SBATCH --account=cso100
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=%x.o%j.%N

declare -xr SINGULARITY_MODULE='singularitypro/3.11'

module purge
module load "${SINGULARITY_MODULE}"
module list

export NVIDIA_DISABLE_REQUIRE=true

time -p singularity exec --bind /expanse,/scratch --nv ../sandbox python -u ./train.py
