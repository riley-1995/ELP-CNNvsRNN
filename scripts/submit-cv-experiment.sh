#!/usr/bin/env bash
# filepath: scripts/submit-cv-experiment.sh

# Usage:
#   bash scripts/submit-cv-experiment.sh <cnn|rnn> <shared|debug>
# Example:
#   bash scripts/submit-cv-experiment.sh cnn shared

if [ $# -ne 2 ]; then
    echo "Usage: $0 <cnn|rnn> <shared|debug>"
    exit 1
fi

MODEL_TYPE=$1
PARTITION=$2

if [[ "$PARTITION" == "shared" ]]; then
    JOB_SCRIPT="scripts/run-cross_validation_experiment-gpu-shared.sh"
    JOB_NAME="${MODEL_TYPE}-cross_validation_experiment_shared"
elif [[ "$PARTITION" == "debug" ]]; then
    JOB_SCRIPT="scripts/run-cross_validation_experiment-gpu-debug.sh"
    JOB_NAME="${MODEL_TYPE}-cross_validation_experiment_debug"
else
    echo "Partition must be 'shared' or 'debug'"
    exit 1
fi

sbatch --job-name="$JOB_NAME" "$JOB_SCRIPT" "$MODEL_TYPE"