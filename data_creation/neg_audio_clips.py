"""
Usage Instructions:
-------------------
It is recommended to first run pos_audio_clips.py in both test and train modes,
as the number of positive clips generated will be used to determine the number
of negative clips that should be created in order to ensure a balanced dataset.

Ensure your virtual environment is activated before running.
"""

import os
import pandas as pd
import wave
import numpy as np
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

# The single source folder to scan for all negative .wav files
NEG_SOURCE_INPUT_DIR = os.path.join(CORNELL_DATA_ROOT, "Gunshot/Testing/PNNN/Sounds")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_NEG_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "training_clips_neg")
TEST_NEG_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "testing_clips_neg")

TRAIN_POS_CLIPS_DIR = os.path.join(PROJECT_ROOT, "data", "training_clips_pos")
TEST_POS_CLIPS_DIR = os.path.join(PROJECT_ROOT, "data", "testing_clips_pos")

os.makedirs(TRAIN_NEG_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_NEG_OUTPUT_DIR, exist_ok=True)

neg_wav_files = []
find_wav_files(NEG_SOURCE_INPUT_DIR, neg_wav_files)
print(f"{len(neg_wav_files)} negative wav files found in {NEG_SOURCE_INPUT_DIR}")

print(f"Checking number of existing positive clips to ensure balanced dataset when generating negative clips.")

pos_train_clips_count = count_wavs(TRAIN_POS_CLIPS_DIR)
pos_test_clips_count = count_wavs(TEST_POS_CLIPS_DIR)
print(f"pos_train_clips_count: {pos_train_clips_count}")
print(f"pos_test_clips_count: {pos_test_clips_count}")

if (pos_test_clips_count and pos_train_clips_count):
    train_ratio = pos_train_clips_count / (pos_test_clips_count + pos_train_clips_count)
    print(f"Train ratio: {train_ratio}")
    max_train_clips = pos_train_clips_count
    max_test_clips = pos_test_clips_count
else:
    train_ratio = 0.6
    print(f"Positive clips missing, using default train ratio: {train_ratio}")
    if not pos_train_clips_count:
        max_train_clips = 2944
        print(f"No positive train clips found, using default value for number of neg train clips to generate: {max_train_clips}")
    if not pos_test_clips_count:
        max_test_clips = 1840
        print(f"No positive test clips found, using default value for number of neg test clips to generate: {max_test_clips}")
    

num_neg_train_input_wavs = int(len(neg_wav_files)*train_ratio)
num_neg_test_input_wavs = len(neg_wav_files)-num_neg_train_input_wavs
print(f"Using {num_neg_train_input_wavs} neg input wavs for creating training clips and {num_neg_test_input_wavs} neg input wavs for creating testing clips.")

# Train test split
neg_train_wavs, neg_test_wavs = train_test_split(neg_wav_files, test_size=num_neg_test_input_wavs, random_state=42)
print(f"Train test split performed.")
print(f"Training neg input wav files: \n{neg_train_wavs}")
print(f"Testing neg input wav files: \n{neg_test_wavs}")

# Process clips
sample_length = 5 # seconds
target_sr = 4000 # hz
expected_final_frames = sample_length * target_sr  # Expected length after downsampling

for type in ["train", "test"]:

    print(f"Generating neg {type}ing clips.")

    if type == "train":
        max_clips = max_train_clips
        input_wav_files = neg_train_wavs
        output_dir = TRAIN_NEG_OUTPUT_DIR
    else:
        max_clips = max_test_clips
        input_wav_files = neg_test_wavs
        output_dir = TEST_NEG_OUTPUT_DIR

    counter = 0

    for file in input_wav_files:
        params = get_wav_params(file)
        max_frames = params.nframes
        sample_width = params.sampwidth

        sample_frames_to_grab = sample_length * params.framerate

        if max_frames >= sample_frames_to_grab and counter < max_clips:
            starting_pos = 0

            while starting_pos + sample_frames_to_grab <= max_frames:
                
                with wave.open(file, 'rb') as wav_file:
                    wav_file.setpos(starting_pos)
                    frames = wav_file.readframes(sample_frames_to_grab)

                    dtype = None
                    if sample_width == 1:
                        dtype = np.int8
                    elif sample_width == 2:
                        dtype = np.int16
                    elif sample_width == 4:
                        dtype = np.int32
                    else:
                        exit()

                    data = np.frombuffer(frames, dtype=dtype)

                    if params.nchannels > 1:
                        data = data.reshape((-1, params.nchannels))
                
                data = apply_low_pass_filter(data, params.framerate, cutoff_hz=200)
                data = down_sample(data, params.framerate, target_sr, expected_final_frames)

                if len(data) == expected_final_frames:
                    save_audio_to_wav(os.path.join(output_dir, f'{os.path.basename(file.split(".")[0])}_neg_{round(starting_pos)}_{counter}.wav'), data, target_sr)
                else:
                    print("Not saving, insufficient frames")
                    
                counter += 1

                if counter == max_clips:
                    print(f"{max_clips} neg {type}ing clips generated.")
                    break

                starting_pos += int(sample_length * params.framerate)
