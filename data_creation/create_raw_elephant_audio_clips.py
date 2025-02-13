import os
import pandas as pd
import wave
import numpy as np
from scipy.signal import resample

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
data_folder = "/home/lucas/Desktop/ELP-sounds-folder/elephant_signals"
meta_files = ["nn_ele_hb_00-24hr_TrainingSet.txt"] #, "NN_Random_Sample_Hand_ele.txt"]

# Output folders
positive_samples_folder = 'elephant_raw_audio'

total_positive_samples = 0

sample_length = 5 # seconds
target_sr = 4000 # hz

# Create the Elephant Clips
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


    if not os.path.exists(positive_samples_folder) or not os.path.isdir(positive_samples_folder):
        os.mkdir(positive_samples_folder)

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

                # Process
                audio_clip = np.asarray(audio_clip, dtype=np.int16)
                audio_clip = down_sample(audio_clip, sampling_rate, target_sr)

                if len(audio_clip) == (sample_length * target_sr):
                    positive_sample_counter += 1
                    total_positive_samples += 1
                    save_audio_to_wav(os.path.join(positive_samples_folder, f'positive_sample_{total_positive_samples}.wav'), audio_clip, target_sr, 1, 2)
                else:
                    print("Removing audio clip: insufficient length.")

        else:
            print(f"Target file {target_file} does not exist...")
    print(f"{total_positive_samples} samples created...")

