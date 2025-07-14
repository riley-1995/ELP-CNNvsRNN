"""
Usage Instructions:
-------------------
It is recommended to first run pos_audio_clips.py in both test and train modes,
as the number of positive clips generated will be used to determine the number
of negative clips that should be created in order to ensure a balanced dataset.

Ensure your virtual environment is activated before running.
"""

import os
import random
import pandas as pd
import wave
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from utils import *
from data_path_config import DataPathConfig

paths = DataPathConfig()

NEG_SOURCE_INPUT_DIR = paths.NEG_SOURCE_INPUT_DIR
TRAIN_VAL_NEG_OUTPUT_DIR = paths.TRAIN_VAL_NEG_CLIPS_DIR
HOLDOUT_TEST_NEG_OUTPUT_DIR = paths.HOLDOUT_TEST_NEG_CLIPS_DIR
TRAIN_VAL_POS_CLIPS_DIR = paths.POS_TRAIN_VAL_CLIPS_DIR
HOLDOUT_TEST_POS_CLIPS_DIR = paths.POS_HOLDOUT_TEST_CLIPS_DIR

os.makedirs(TRAIN_VAL_NEG_OUTPUT_DIR, exist_ok=True)
os.makedirs(HOLDOUT_TEST_NEG_OUTPUT_DIR, exist_ok=True)

neg_wav_files = []
find_wav_files(NEG_SOURCE_INPUT_DIR, neg_wav_files)
print(f"{len(neg_wav_files)} negative wav files found in {NEG_SOURCE_INPUT_DIR}")

# Shuffle wav files for randomness
random.seed(42)
random.shuffle(neg_wav_files)

print(f"Checking number of existing positive clips to ensure balanced dataset when generating negative clips.")

pos_train_clips_count = count_wavs(TRAIN_VAL_POS_CLIPS_DIR)
pos_test_clips_count = count_wavs(HOLDOUT_TEST_POS_CLIPS_DIR)
print(f"pos_train_clips_count: {pos_train_clips_count}")
print(f"pos_test_clips_count: {pos_test_clips_count}")

if (pos_test_clips_count and pos_train_clips_count):
    train_ratio = pos_train_clips_count / (pos_test_clips_count + pos_train_clips_count)
    print(f"Train ratio: {train_ratio}")
    max_train_clips = pos_train_clips_count
    max_test_clips = pos_test_clips_count
else:
    train_ratio = 0.3
    print(f"Positive clips missing, using default train ratio: {train_ratio}")
    if not pos_train_clips_count:
        max_train_clips = 4784
        print(f"No positive train clips found, using default value for number of neg train clips to generate: {max_train_clips}")
    if not pos_test_clips_count:
        max_test_clips = 10698
        print(f"No positive test clips found, using default value for number of neg test clips to generate: {max_test_clips}")
    

num_neg_train_input_wavs = int(len(neg_wav_files)*train_ratio)
num_neg_test_input_wavs = len(neg_wav_files)-num_neg_train_input_wavs
print(f"Using {num_neg_train_input_wavs} neg input wavs for creating training clips and {num_neg_test_input_wavs} neg input wavs for creating testing clips.")

# Train test split of input .wavs
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
        output_dir = TRAIN_VAL_NEG_OUTPUT_DIR
    else:
        max_clips = max_test_clips
        input_wav_files = neg_test_wavs
        output_dir = HOLDOUT_TEST_NEG_OUTPUT_DIR

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
                    counter += 1
                else:
                    print("Not saving, insufficient frames")

                if counter == max_clips:
                    print(f"{max_clips} neg {type}ing clips generated.")
                    break

                starting_pos += int(sample_length * params.framerate)
