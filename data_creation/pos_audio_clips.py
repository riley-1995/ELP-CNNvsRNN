import os
import pandas as pd
import numpy as np
from utils import *

# Open the meta data
meta_data_folder = "/home/lucas/Desktop/PNNN"

meta_files = [file for file in os.listdir(meta_data_folder) if file.split(".")[-1] == "txt"]

sounds_folder = "/home/lucas/Desktop/PNNN/Sounds"

print(os.listdir(sounds_folder))
# Output folders
positive_samples_folder = 'testing_elephant_audio'

total_positive_samples = 0

sample_length = 5 # seconds
target_sr = 4000 # hz

expected_frames = sample_length * target_sr 

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

    # make the target directory if it does not already exist
    if not os.path.exists(positive_samples_folder) or not os.path.isdir(positive_samples_folder):
        os.mkdir(positive_samples_folder)

    for file in set(data['Begin File']):
        target_file = os.path.join(sounds_folder, file)
        print(target_file)

        if os.path.exists(target_file):
            print(f"Opening file: {target_file}")
            clips = data[data['Begin File'] == file]

            # Create positive samples of elephant rumbles
            positive_sample_counter = 0
            params = None
            for i, clip in clips.iterrows():

                params, audio_clip = read_wav_frames(target_file, clip['File Offset (s)'], sample_length)
                sampling_rate = params.framerate

                # Process
                audio_clip = np.asarray(audio_clip, dtype=np.int16)
                audio_clip = down_sample(audio_clip, sampling_rate, target_sr, expected_frames)

                if len(audio_clip) == (sample_length * target_sr):
                    positive_sample_counter += 1
                    total_positive_samples += 1
                    save_audio_to_wav(os.path.join(positive_samples_folder, f'{file.split(".")[0]}_pos_{round(clip["File Offset (s)"])}_{total_positive_samples}.wav'), audio_clip, target_sr, 1, 2)
                else:
                    print("Removing audio clip: insufficient length.")

        else:
            print(f"Target file {target_file} does not exist...")
    print(f"{total_positive_samples} samples created...")

