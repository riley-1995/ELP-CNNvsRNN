"""
Usage Instructions:
-------------------
This script should be run from the root directory of the ELP-CNNvsRNN repository.

Ensure your virtual environment is activated before running and that the path
for the Cornell Data root folder is correct.

To process the TRAINING dataset, run:
    python3 data_creation/pos_audio_clips.py --mode train

To process the TESTING dataset, run:
    python3 data_creation/pos_audio_clips.py --mode test
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from dotenv import load_dotenv
from utils import *

# Get the root path from the environment.
# os.getenv() returns None if the variable isn't found.
load_dotenv()
CORNELL_DATA_ROOT = os.getenv("CORNELL_DATA_ROOT")

# Add a check to make sure the path was found
if not CORNELL_DATA_ROOT:
    print("❌ Error: CORNELL_DATA_ROOT not found.")
    print("Please create a .env file in the project root and add the line:")
    print('CORNELL_DATA_ROOT="/path/to/your/data"')
    exit() # Stop the script if the path isn't configured

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Training Data Paths ---
TRAIN_METADATA_FOLDER = os.path.join(CORNELL_DATA_ROOT, "Rumble/Training/pnnn")
TRAIN_SOUNDS_FOLDER = os.path.join(CORNELL_DATA_ROOT, "Rumble/Training/Sounds")
TRAIN_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "training_clips_pos")

# --- Testing Data Paths ---
TEST_METADATA_FOLDER = os.path.join(CORNELL_DATA_ROOT, "Rumble/Testing/PNNN")
TEST_SOUNDS_FOLDER = os.path.join(CORNELL_DATA_ROOT, "Rumble/Testing/PNNN/Sounds")
TEST_OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "testing_clips_pos")

# Set up the argument parser
parser = argparse.ArgumentParser(description="Create positive audio clips for training or testing.")
parser.add_argument("--mode", type=str, required=True, choices=['train', 'test'], 
                    help="The dataset to process ('train' or 'test').")
args = parser.parse_args()

# Select the correct paths based on the mode
if args.mode == 'train':
    meta_data_folder = TRAIN_METADATA_FOLDER
    sounds_folder = TRAIN_SOUNDS_FOLDER
    positive_samples_folder = TRAIN_OUTPUT_FOLDER
    print("--- Running in TRAINING mode ---")
else: # mode is 'test'
    meta_data_folder = TEST_METADATA_FOLDER
    sounds_folder = TEST_SOUNDS_FOLDER
    positive_samples_folder = TEST_OUTPUT_FOLDER
    print("--- Running in TESTING mode ---")

# Open the meta data
meta_files = [file for file in os.listdir(meta_data_folder) if file.split(".")[-1] == "txt"]

print(os.listdir(sounds_folder))

total_positive_samples = 0

sample_length = 5 # seconds
target_sr = 4000 # hz

expected_frames = sample_length * target_sr 

# Make the target directory if it does not already exist
os.makedirs(positive_samples_folder, exist_ok=True)

# Create the Elephant Clips
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
        target_file = os.path.join(sounds_folder, file)
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
                    total_positive_samples += 1 # Still need to increment the counter to keep names unique
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