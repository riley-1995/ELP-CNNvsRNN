#!/usr/bin/env bash
#SBATCH --job-name=cross_validation_experiment
####Change account below
#SBATCH --account=lbutler2
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=90G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=%x.o%j.%N

declare -xr SINGULARITY_MODULE='singularitypro/3.11'

module purge
module load "${SINGULARITY_MODULE}"
module list
printenv

export NVIDIA_DISABLE_REQUIRE=true

time -p singularity exec --bind /expanse,/scratch --nv /cm/shared/apps/containers/singularity/tensorflow/tensorflow-latest.sif ~/.conda/envs/elp/bin/python -u cross_validation_experiment.py
