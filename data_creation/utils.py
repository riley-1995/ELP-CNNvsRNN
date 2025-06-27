import wave
from scipy.signal import resample, butter, lfilter
import numpy as np
import os
import tensorflow as tf
from collections import Counter
from sklearn.model_selection import train_test_split

def write_tfrecords(dataset, file_prefix):
    filename = f'{file_prefix}.tfrecord'
    with tf.io.TFRecordWriter(filename) as writer:
        for sample, label in dataset:
            feature = {
                'sample': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(sample).numpy()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()]))
            }
            writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())

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

def apply_low_pass_filter(audio_data, sample_rate, cutoff_hz=200, order=5):
    """
    Applies a low-pass Butterworth filter to audio data.

    Args:
        audio_data (np.array): The audio time-series data.
        sample_rate (int): The sample rate of the audio.
        cutoff_hz (int): The cutoff frequency for the filter.
        order (int): The order of the filter.

    Returns:
        np.array: The filtered audio data.
    """
    nyquist_freq = 0.5 * sample_rate
    normal_cutoff = cutoff_hz / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Ensure audio_data is float for filtering
    filtered_data = lfilter(b, a, audio_data.astype(np.float64))
    
    return filtered_data

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

def count_wavs(directory):
    """Counts .wav files in a directory"""
    print(f"Checking for existing clips in: {directory}")
    if not os.path.isdir(directory):
        print("Directory not found.")
        return -1
    count = len([f for f in os.listdir(directory) if f.endswith('.wav')])
    print(f"{count} clips found.")
    return count

def find_wav_files(folder_file, array):
    if os.path.isdir(folder_file):
        for file in os.listdir(folder_file):
            find_wav_files(os.path.join(folder_file, file), array)
    elif folder_file.endswith(".wav"):
        array.append(folder_file)

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

######################################################### TFRECORDS creation

def load_wav_file(file_path):
    raw_audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(raw_audio, desired_channels=1)
    return tf.squeeze(audio, axis=-1)

def load_wavs_into_dataset(directory):
    file_paths = tf.io.gfile.glob(os.path.join(directory, "*.wav"))
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(lambda path: (load_wav_file(path), tf.shape(load_wav_file(path))[0]), num_parallel_calls=tf.data.AUTOTUNE)
    
    length_counts = Counter(length for _, length in dataset.as_numpy_iterator())
    print("📏 Sample Length Distribution:", {length: count for length, count in sorted(length_counts.items())})
    
    return dataset.map(lambda audio, _: audio)

def compute_statistics(dataset):
    total_sum, total_sum_sq, total_count = 0.0, 0.0, 0.0
    
    for sample in dataset.as_numpy_iterator():
        total_sum += np.sum(sample)
        total_sum_sq += np.sum(np.square(sample))
        total_count += sample.shape[0]
    
    mean, std = total_sum / total_count, np.sqrt((total_sum_sq / total_count) - (total_sum / total_count) ** 2)
    return mean, std

def normalize_dataset(dataset, mean, std):
    return dataset.map(lambda audio: (audio - mean) / std)

def add_label(dataset, label):
    label_tensor = tf.convert_to_tensor(label, dtype=tf.int64)
    return dataset.map(lambda sample: (sample, label_tensor), num_parallel_calls=tf.data.AUTOTUNE)

def stratified_split(dataset, val_size=0.2):
    samples, labels = zip(*dataset.as_numpy_iterator())
    X_train,  X_val, y_train, y_val = train_test_split(samples, labels, test_size=val_size, stratify=labels, random_state=42)
    return map(tf.data.Dataset.from_tensor_slices, [(X_train, y_train), (X_val, y_val)])
