#!/usr/bin/env bash
#SBATCH --job-name=cross_validation_experiment
####Change account below
#SBATCH --account=cso100
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --gpus=2
#SBATCH --time=00:30:00
#SBATCH --output=%x.o%j.%N

declare -xr SINGULARITY_MODULE='singularitypro/3.11'

module purge
module load "${SINGULARITY_MODULE}"
module list

export NVIDIA_DISABLE_REQUIRE=true

time -p singularity exec --bind /expanse,/scratch --nv /home/lbutler2/train-container-sandbox python -u cross_validation_experiment.py
