import os
import pandas as pd
import wave
import numpy as np
from scipy.signal import resample

def down_sample(audio, input_sr, output_sr, target_frames):

    if input_sr <= output_sr:
        return audio[:target_frames]

    num_samples = int(len(audio) * float(output_sr)/ input_sr)
    down_sampled_audio = resample(audio, num_samples)

    # Ensure the sampled signal is exactly 'target frames'
    if len(down_sampled_audio) > target_frames:
        down_sampled_audio = down_sampled_audio[:target_frames]
    elif len(down_sampled_audio) < target_frames:
        down_sampled_audio = np.pad(down_sampled_audio, (0, target_frames - len(down_sampled_audio)))
    
    return down_sampled_audio

def get_wav_params(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        params = wav_file.getparams()

    return params

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

def find_wav_files(folder_file, array):
    if os.path.isdir(folder_file):
        for file in os.listdir(folder_file):
            find_wav_files(os.path.join(folder_file, file), array)
    elif folder_file.endswith(".wav"):
        array.append(folder_file)
    
# Non-Elephant Signals
target_input_dir = '/home/lucas/Desktop/ELP-sounds-folder/Non-elephant signals'
target_output_dir = 'non_elephant_raw_audio'

if not os.path.exists(target_output_dir) or not os.path.isdir(target_output_dir):
    os.mkdir(target_output_dir)

wav_files = []

find_wav_files(target_input_dir, wav_files)

counter = 0

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
                save_audio_to_wav(os.path.join(target_output_dir, f'non_elephant_sample_{counter}.wav'), data, target_sr)
            else:
                print("Not saving, insufficient frames")
                
            counter += 1
            starting_pos += int(params.framerate // 2) # Shift the starting postition forward 1 second