import os
import pandas as pd
import wave
import numpy as np
from scipy.signal import resample
import random

# piush
def down_sample(audio, input_sr, output_sr):

    if input_sr <= output_sr:
        return audio

    num_samples = int(len(audio) * float(output_sr)/ input_sr)
    down_sampled_audio = resample(audio, num_samples)

    return down_sampled_audio

def read_wav_frames(file_path, starting_time, duration):
    with wave.open(file_path, 'rb') as wav_file:

        params = wav_file.getparams()
        sampling_rate = params.framerate
        n_channels = params.nchannels
        sample_width = params.sampwidth

        starting_pos = int(sampling_rate * starting_time)

        wav_file.setpos(starting_pos)
        
        frames = wav_file.readframes(int(sampling_rate * duration))

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
        if n_channels > 1:
            data = data.reshape((-1, n_channels))
        
    return params, data

def save_audio_to_wav(filename, audio_data, sample_rate, num_channels=1, sample_width=2):
    with wave.open(filename, 'wb') as wav_file:
        # Set the parameters for the wav file
        wav_file.setnchannels(num_channels)           # Mono (1) or Stereo (2)
        wav_file.setsampwidth(sample_width)           # 2 bytes for 16-bit samples
        wav_file.setframerate(sample_rate)            # Sample rate (e.g., 44100 Hz)

        # Convert audio data to the appropriate format (e.g., int16 for 16-bit audio)
        audio_data = np.array(audio_data, dtype=np.int16)

        # Write the audio data to the file
        wav_file.writeframes(audio_data.tobytes())

# Open the meta data
data_folder = "/home/lucas/Desktop/ELP-sounds-folder"
meta_files = ["nn_ele_hb_00-24hr_TrainingSet.txt", "NN_Random_Sample_Hand_ele.txt"]

total_positive_samples = 0
total_negative_samples = 0

for meta_file in meta_files:
    print(f"Opening {meta_file}")
    data = pd.read_csv(os.path.join(data_folder, meta_file), delimiter='\t')

    # Filter out meta data
    try:
        negative_signal = ['DUMMY_NoEles', 'DUMMY_noEles']
        data = data[~data['Tag 1'].isin(negative_signal)]
    except:
        pass

    try:
        data = data[~data['notes'].str.contains('faint|marginal|gorilla', case=False, na=False)]
    except:
        pass

    # Output folders
    positive_samples_folder = 'raw_audio_positive_samples'
    negative_samples_folder = 'raw_audio_negative_samples'

    if not os.path.exists(positive_samples_folder) or not os.path.isdir(positive_samples_folder):
        os.mkdir(positive_samples_folder)

    if not os.path.exists(negative_samples_folder) or not os.path.isdir(negative_samples_folder):
        os.mkdir(negative_samples_folder)
    
    sample_length = 10 # seconds
    target_sr = 4000 # hz

    for file in set(data['Begin File']):
        target_file = os.path.join(data_folder, file)

        if os.path.exists(target_file):
            print(f"Opening file: {target_file}")
            clips = data[data['Begin File'] == file]

            # Create positive samples of elephant rumbles
            positive_sample_counter = 0
            params = None
            for i, clip in clips.iterrows():

                params, audio_clip = read_wav_frames(target_file, clip['File Offset (s)'], sample_length)
                sampling_rate = params.framerate
                positive_sample_counter += 1
                total_positive_samples += 1

                # Process
                audio_clip = np.asarray(audio_clip, dtype=np.int16)
                audio_clip = down_sample(audio_clip, sampling_rate, target_sr)

                save_audio_to_wav(os.path.join(positive_samples_folder, f'positive_sample_{total_positive_samples}.wav'), audio_clip, target_sr, 1, 2)

            #print(f"{positive_sample_counter} collected from {target_file}.")
            negative_samples = 0

            clip_start_times=clips['File Offset (s)']
            clip_end_times=(clips['File Offset (s)'] + sample_length).astype(int)

            total_time = params.nframes / params.framerate
            # Create samples of no elephant rumble
            while negative_samples < positive_sample_counter:
                
                negative_sample_begin_time = max(0, random.randint(0, int(total_time)) - sample_length - 1)# subtract the length of the sample so the clip doesn't go off the file
                negative_sample_end_time = negative_sample_begin_time + sample_length

                overlap = np.any(
                    (clip_start_times <= negative_sample_begin_time) & (negative_sample_begin_time <= clip_end_times) |
                    (clip_start_times <= negative_sample_end_time) & (negative_sample_end_time <= clip_end_times)
                )

                if not overlap:
                    negative_samples += 1
                    total_negative_samples += 1
                
                    params, audio_clip = read_wav_frames(target_file, negative_sample_begin_time, sample_length)
                    sampling_rate = params.framerate

                    # Process
                    audio_clip = np.asarray(audio_clip, dtype=np.int16)
                    audio_clip = down_sample(audio_clip, sampling_rate, target_sr)

                    save_audio_to_wav(os.path.join(negative_samples_folder, f'negative_sample_{total_negative_samples}.wav'), audio_clip, target_sr, 1, 2)
        else:
            print(f"Target file {target_file} does not exist...")
    print(f"{total_positive_samples} samples created...")