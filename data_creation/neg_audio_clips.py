import os
import pandas as pd
import wave
import numpy as np
from utils import *

# Non-Elephant Signals
target_input_dir = '/home/lucas/Desktop/gunshot'
target_output_dir = 'testing_elephant_audio_neg'

if not os.path.exists(target_output_dir) or not os.path.isdir(target_output_dir):
    os.mkdir(target_output_dir)

wav_files = []

find_wav_files(target_input_dir, wav_files)

print(wav_files)
counter = 0
count = 920
sample_length = 5 # seconds
target_sr = 4000 # hz

expected_final_frames = sample_length * target_sr  # Expected length after downsampling

for file in wav_files:
    params = get_wav_params(file)
    max_frames = params.nframes
    sample_width = params.sampwidth

    sample_frames_to_grab = sample_length * params.framerate

    if max_frames >= sample_frames_to_grab:
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
            
            data = down_sample(data, params.framerate, target_sr, expected_final_frames)

            if len(data) == expected_final_frames:
                save_audio_to_wav(os.path.join(target_output_dir, f'{os.path.basename(file.split(".")[0])}_neg_{round(starting_pos)}_{counter}.wav'), data, target_sr)
            else:
                print("Not saving, insufficient frames")
                
            counter += 1

            if counter == count:
                print("Max count reached")
                exit()

            starting_pos += int(sample_length * params.framerate)
