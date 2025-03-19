import os
import numpy as np
import tensorflow as tf
import argparse

def write_tfrecords(dataset: tf.data.Dataset, file_prefix):

    filename = f'{file_prefix}.tfrecord'
    with tf.io.TFRecordWriter(filename) as writer:
        for sample, label in dataset:
            feature = {
                'sample': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(sample).numpy()])
                ),
                'label': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label.numpy()])
                )
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example_proto.SerializeToString())

def read_tfrecord(example_proto):
    """
    Parses a single TFRecord example.
    """
    feature_description = {
        'sample': tf.io.FixedLenFeature([], tf.string),  # Serialized tensor
        'label': tf.io.FixedLenFeature([], tf.int64)     # Integer label
    }
    
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    # Deserialize the sample tensor
    sample = tf.io.parse_tensor(parsed_example['sample'], out_type=tf.float32)
    label = parsed_example['label']
    
    return sample, label

def load_tfrecords(file_pattern):
    """
    Loads and parses TFRecords into a tf.data.Dataset.
    """
    dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(file_pattern))
    dataset = dataset.map(read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset

def stft_hann_window(audio, frame_length, frame_step, bins_to_grab):
    stft = tf.signal.stft(
        audio,
        frame_length=frame_length,
        frame_step=frame_step,
        window_fn=tf.signal.hann_window
    )

    # Grab up to 512 hz, make all the values real, add a third axis
    stft = stft[:, 2:bins_to_grab]
    stft = tf.math.log(tf.abs(stft) + 1e-10)
    stft = tf.expand_dims(stft, axis=-1)

    return stft

def median_filter(spectrogram):
    from scipy.ndimage import median_filter
    return median_filter(spectrogram, size=(3,6))

def apply_stft(dataset, frame_length, frame_step, sample_rate, max_frequency):
    freq_resolution = sample_rate / frame_length
    tf.print(freq_resolution)
    bins_to_grab = int(max_frequency / freq_resolution)
    return dataset.map(lambda audio, label: (stft_hann_window(audio, frame_length, frame_step, bins_to_grab), label), 
                       num_parallel_calls=tf.data.AUTOTUNE)

def compute_global_stats(datasets):
    all_spectrograms = []
    for dataset, _ in datasets:
        for spectrogram, _ in dataset:
            all_spectrograms.append(spectrogram.numpy())
    all_spectrograms = np.concatenate(all_spectrograms, axis=0)
    global_mean = np.mean(all_spectrograms)
    global_std = np.std(all_spectrograms)
    return global_mean, global_std

def normalize_spectrogram(spectrogram, global_mean, global_std):
    return (spectrogram - global_mean) / global_std

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():
    parser = argparse.ArgumentParser(description="Generate spectrograms from TFRecords.")
    parser.add_argument("--audio_tfrecords_directory", type=str, required=True, help="Path to the directory containing audio TFRecords.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save the spectrogram dataset.")
    args = parser.parse_args()

    # Create spectrograms and write
    audio_files_directory = args.audio_tfrecords_directory
    spectrogram_dataset = args.output_folder

    if not os.path.exists(spectrogram_dataset) or not os.path.isdir(spectrogram_dataset):
        os.mkdir(spectrogram_dataset)
        
    # Load the audio tfrecords
    files = []
    for file in os.listdir(audio_files_directory):
        if file.endswith(".tfrecord"):
            files.append(
                (
                    os.path.join(audio_files_directory, file),
                    os.path.basename(file.replace(".tfrecord", ""))
                ))

    datasets = [(load_tfrecords(file[0]), file[1]) for file in files]

    frame_length = 2000
    frame_step = 32
    sample_rate = 4000
    max_frequency = 200

    for i, (dataset, name) in enumerate(datasets):
        datasets[i] = (apply_stft(dataset, frame_length, frame_step, sample_rate, max_frequency), name)

    global_mean, global_std = compute_global_stats(datasets)

    for dataset, file_name in datasets:
        normalized_dataset = dataset.map(lambda spectrogram, label: (normalize_spectrogram(spectrogram, global_mean, global_std), label), num_parallel_calls=tf.data.AUTOTUNE)
        
        write_tfrecords(normalized_dataset, os.path.join(spectrogram_dataset, f"{file_name}"))

if __name__ == "__main__":
    main()
