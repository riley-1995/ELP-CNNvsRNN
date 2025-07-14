"""
Usage Instructions:
-------------------
This script should be run from the root directory of the ELP-CNNvsRNN repository.

Ensure your virtual environment is activated before running and that the path
for the Cornell Data root DIR is correct.

To process the TRAINING dataset, run:
    python3 data_creation/pos_audio_clips.py --mode train

To process the TESTING dataset, run:
    python3 data_creation/pos_audio_clips.py --mode test
"""

import os
import pandas as pd
import numpy as np
import argparse
from utils import *
from data_path_config import DataPathConfig

paths = DataPathConfig()

# --- Train + Validation Data Paths ---
POS_TRAIN_VAL1_METADATA_DIR = paths.POS_TRAIN_VAL1_METADATA_DIR
POS_TRAIN_VAL1_SOUNDS_DIR = paths.POS_TRAIN_VAL1_SOUNDS_DIR
POS_TRAIN_VAL2_METADATA_DIR = paths.POS_TRAIN_VAL2_METADATA_DIR
POS_TRAIN_VAL2_SOUNDS_DIR = paths.POS_TRAIN_VAL2_SOUNDS_DIR
POS_TRAIN_VAL_OUTPUT_DIR = paths.POS_TRAIN_VAL_CLIPS_DIR

# --- Holdout Testing Data Paths ---
POS_HOLDOUT_TEST_METADATA_DIR = paths.POS_HOLDOUT_TEST_METADATA_DIR
POS_HOLDOUT_TEST_SOUNDS_DIR = paths.POS_HOLDOUT_TEST_SOUNDS_DIR
POS_HOLDOUT_TEST_OUTPUT_DIR = paths.POS_HOLDOUT_TEST_CLIPS_DIR


# Set up the argument parser
parser = argparse.ArgumentParser(description="Create positive audio clips for training or testing.")
parser.add_argument("--mode", type=str, required=True, choices=['train', 'test'], 
                    help="The dataset to process ('train' or 'test').")
args = parser.parse_args()

if args.mode == 'train':
    # List of (metadata_DIR, sounds_DIR) pairs to combine
    meta_sounds_pairs = [
        (POS_TRAIN_VAL1_METADATA_DIR, POS_TRAIN_VAL1_SOUNDS_DIR),
        (POS_TRAIN_VAL2_METADATA_DIR, POS_TRAIN_VAL2_SOUNDS_DIR)
    ]
    positive_samples_folder = POS_TRAIN_VAL_OUTPUT_DIR
    print("--- Running in TRAINING mode (combining two sources) ---")
else:  # mode is 'test'
    meta_sounds_pairs = [
        (POS_HOLDOUT_TEST_METADATA_DIR, POS_HOLDOUT_TEST_SOUNDS_DIR)
    ]
    positive_samples_folder = POS_HOLDOUT_TEST_OUTPUT_DIR
    print("--- Running in TESTING mode ---")

# Make the target directory if it does not already exist
os.makedirs(positive_samples_folder, exist_ok=True)

total_positive_samples = 0
sample_length = 5  # seconds
target_sr = 4000  # hz
expected_frames = sample_length * target_sr

for meta_data_folder, sounds_folder in meta_sounds_pairs:
    # Open the meta data
    meta_files = [file for file in os.listdir(meta_data_folder) if file.split(".")[-1] == "txt"]

    print(f"Processing sounds from: {sounds_folder}")
    print(os.listdir(sounds_folder))

    for meta_file in meta_files:
        print(f"Opening {meta_file}")
        data = pd.read_csv(os.path.join(meta_data_folder, meta_file), delimiter='\t')
        print(data.columns)

        # Filter out bad meta data
        try:
            negative_signal = ['DUMMY_NoEles', 'DUMMY_noEles']
            data = data[~data['Tag 1'].isin(negative_signal)]
        except:
            pass

        try:
            data = data[~data['notes'].str.contains('faint|marginal|gorilla', case=False, na=False)]
        except:
            pass

        for file in set(data['Begin File']):
            # If in test mode, fix the filename prefix
            if args.mode == 'test' and file.startswith('dzan_'):
                file_on_disk = file.replace('dzan_', 'dz_', 1)
            else:
                file_on_disk = file

            target_file = os.path.join(sounds_folder, file_on_disk)
            print(target_file)

            if os.path.exists(target_file):
                print(f"Opening file: {target_file}")
                clips = data[data['Begin File'] == file]

                # Create positive samples of elephant rumbles
                params = None
                for i, clip in clips.iterrows():

                    output_filename = f'{file.split(".")[0]}_pos_{round(clip["File Offset (s)"])}_{total_positive_samples + 1}.wav'
                    output_filepath = os.path.join(positive_samples_folder, output_filename)

                    # Check if the file already exists. If so, skip this iteration.
                    if os.path.exists(output_filepath):
                        print(f"Skipping already processed clip: {output_filename}")
                        total_positive_samples += 1  # Still need to increment the counter to keep names unique
                        continue

                    print(f"Processing new clip: {output_filename}")
                    params, audio_clip = read_wav_frames(target_file, clip['File Offset (s)'], sample_length)
                    sampling_rate = params.framerate

                    # Process with low pass filter and downsampling
                    audio_clip = np.asarray(audio_clip, dtype=np.int16)
                    audio_clip = apply_low_pass_filter(audio_clip, sampling_rate, cutoff_hz=200)
                    audio_clip = down_sample(audio_clip, sampling_rate, target_sr, expected_frames)

                    if len(audio_clip) == (sample_length * target_sr):
                        total_positive_samples += 1
                        save_audio_to_wav(output_filepath, audio_clip, target_sr, 1, 2)
                    else:
                        print("Removing audio clip: insufficient length.")

            else:
                print(f"Target file {target_file} does not exist...")
print(f"--- Finished. Total positive clips created: {total_positive_samples} ---")