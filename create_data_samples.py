#   Creates TFRecords of spectrograms from wav files
#   Lucas Butler

import tensorflow as tf
from collections import Counter
import wave
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from collections import Counter

def load_wav_file(file_path):
    raw_audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(raw_audio, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)

    return audio

def load_dataset(directory):
    file_paths = tf.io.gfile.glob(os.path.join(directory, "*.wav"))
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    
    def load_and_check_length(file_path):
        audio = load_wav_file(file_path)
        length = tf.shape(audio)[0]
        return audio, length  # Return both audio and length
    
    dataset = dataset.map(load_and_check_length, num_parallel_calls=tf.data.AUTOTUNE)

    # Collect lengths and count occurrences
    lengths = []
    for _, length in dataset.as_numpy_iterator():
        lengths.append(length)

    length_counts = Counter(lengths)

    print("📏 Sample Length Distribution:")
    for length, count in sorted(length_counts.items()):
        print(f"  - Length {length}: {count} samples")

    # Remove length before returning dataset
    dataset = dataset.map(lambda audio, length: audio)

    return dataset

def compute_statistics(dataset):
    total_sum = dataset.map(lambda x: tf.reduce_sum(x)).reduce(tf.constant(0.0), lambda x, y: x + y)
    total_sum_sq = dataset.map(lambda x: tf.reduce_sum(tf.square(x))).reduce(tf.constant(0.0), lambda x, y: x + y)
    total_count = dataset.map(lambda x: tf.cast(tf.shape(x)[0], tf.float32)).reduce(tf.constant(0.0), lambda x, y: x + y)

    mean = total_sum / total_count
    std = tf.sqrt((total_sum_sq / total_count) - tf.square(mean))

    return mean, std

def normalize_dataset(dataset, mean, std):
    return dataset.map(lambda audio: (audio - mean ) / std)

def add_label_to_sample(dataset: tf.data.Dataset, label):

    label_tensor = tf.convert_to_tensor(label, dtype=tf.int64)
    return dataset.map(lambda sample: (sample, label_tensor), num_parallel_calls=tf.data.AUTOTUNE)

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

def stratified_split(dataset: tf.data.Dataset, test_size=0.2, val_size=0.1):

    # Convert dataset to numpy lists
    samples, labels = [], []
    for sample, label in dataset.as_numpy_iterator():
        samples.append(sample)
        labels.append(label)

    samples = np.array(samples)
    labels = np.array(labels)

    # First split: Train + Validation vs. Test
    X_train, X_test, y_train, y_test = train_test_split(
        samples, labels, test_size=test_size, stratify=labels, random_state=42
    )

    # Second split: Train vs. Validation
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val size to remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size_adjusted, stratify=y_train, random_state=42
    )

    # Convert back to tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    return train_ds, val_ds, test_ds

#######################################################

targets = [
    {
        "folder": "/home/lucas/Desktop/ELP-CNN-Spectrogram/elephant_raw_audio",
        "label": 1,
    },
    {
        "folder": "/home/lucas/Desktop/ELP-CNN-Spectrogram/non_elephant_raw_audio",
        "label": 0,
    }
]

# Open all of the wav files and compute the mean and std, then close them
for target in targets:
    target['dataset'] = load_dataset(target['folder'])

combined_dataset = targets[0]['dataset']
for target in targets[1:]:
    combined_dataset = combined_dataset.concatenate(target['dataset'])

print("Calculating global statistics...")
global_mean, global_std = compute_statistics(combined_dataset)

del combined_dataset

# Read all of the wav files and process them into TFExamples, then export all the examples into a TFRecords file
for target in targets:
    print("Normalizing")
    target['dataset'] = normalize_dataset(target['dataset'], global_mean, global_std)
    target['dataset'] = add_label_to_sample(target['dataset'], target['label'])

####### Create the Train, Test, Validate split
combined_dataset = targets[0]['dataset']
for target in targets[1:]:
    combined_dataset = combined_dataset.concatenate(target['dataset'])

combined_dataset = combined_dataset.shuffle(10000, reshuffle_each_iteration=False)
train, validate, test = stratified_split(combined_dataset)
del combined_dataset

sets = [
    (train, 'train'), 
    (validate, 'validate'),
    (test, 'test')
]

# Writing raw audio dataset
audio_folder = "audio_tfrecords"
if not os.path.exists(audio_folder) or not os.path.isdir(audio_folder):
    os.mkdir(audio_folder)

for set in sets:
    write_tfrecords(set[0], os.path.join(audio_folder, f"audio_{set[1]}"))
