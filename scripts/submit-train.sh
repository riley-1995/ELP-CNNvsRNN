#!/usr/bin/env bash
# filepath: scripts/submit-train.sh

# Usage:
#   bash scripts/submit-train.sh <cnn|rnn> <shared|debug>
# Example:
#   bash scripts/submit-train.sh cnn shared

if [ $# -ne 2 ]; then
    echo "Usage: $0 <cnn|rnn> <shared|debug>"
    exit 1
fi

MODEL_TYPE=$1
PARTITION=$2

if [[ "$PARTITION" == "shared" ]]; then
    JOB_SCRIPT="scripts/run-train-gpu-shared.sh"
    JOB_NAME="${MODEL_TYPE}-train_shared"
elif [[ "$PARTITION" == "debug" ]]; then
    JOB_SCRIPT="scripts/run-train-gpu-debug.sh"
    JOB_NAME="${MODEL_TYPE}-train_debug"
else
    echo "Partition must be 'shared' or 'debug'"
    exit 1
fi

sbatch --job-name="$JOB_NAME" "$JOB_SCRIPT" "$MODEL_TYPE"