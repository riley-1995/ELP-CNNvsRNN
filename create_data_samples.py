#   Creates TFRecords of spectrograms from wav files
#   Lucas Butler

import tensorflow as tf
import wave
import numpy as np
import pandas as pd
import os

def load_wav_file(file_path):
    raw_audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(raw_audio, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    audio = tf.cast(audio, tf.float32)

    return audio

def load_dataset(directory):
    file_paths = tf.io.gfile.glob(os.path.join(directory, "*.wav"))
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(load_wav_file, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset

def compute_statistics(dataset):
    sum_x = tf.constant(0.0, dtype=tf.float32)
    sum_x2 = tf.constant(0.0, dtype=tf.float32)
    count = tf.constant(0, dtype=tf.float32)

    for audio in dataset:
        sum_x += tf.reduce_sum(audio)
        sum_x2 += tf.reduce_sum(audio **2 )
        count += tf.cast(tf.size(audio), tf.float32)

    mean = sum_x / tf.cast(count, tf.float32)
    std = tf.sqrt((sum_x2 / tf.cast(count, tf.float32)) - (mean ** 2))

    return mean, std

def normalize_dataset(dataset, mean, std):
    return dataset.map(lambda audio: (audio - mean ) / std)

def stft_hann_window(audio, frame_length, frame_step):
    stft = tf.signal.stft(
        audio,
        frame_length=frame_length,
        frame_step=frame_step,
        window_fn=tf.signal.hann_window
    )

    return tf.math.log(tf.abs(stft) + 1e-10)

def apply_stft(dataset, frame_length, frame_step):
    return dataset.map(lambda audio: stft_hann_window(audio, frame_length, frame_step))

def write_tfrecords(dataset: tf.data.Dataset, label, file_prefix):
    '''
    Input:
        dataset:        tf.Dataset of data to write.
        file_prefix:    Name of the file to write the data.
    '''
    filename = f'{file_prefix}.tfrecord'
    with tf.io.TFRecordWriter(filename) as writer:
        for sample in dataset:

            sample = tf.convert_to_tensor(sample)
            label = tf.convert_to_tensor(label)

            feature = {
                'sample': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(sample).numpy()])),  
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(label).numpy()]))
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example_proto.SerializeToString())

targets = [
    {
        "folder": "/home/lucas/Desktop/ELP-CNN-Spectrogram/raw_audio_positive_samples",
        "label": 1,
        "tfrecord_name": "rumble_positive"
    },
    {
        "folder": "/home/lucas/Desktop/ELP-CNN-Spectrogram/raw_audio_positive_samples",
        "label": 0,
        "tfrecord_name": "rumble_negative"
    }
]

frame_length = 2048
frame_step = 128

# Read all of the wav files and process them into TFExamples, then export all the examples into a TFRecords file
for target in targets:

    print("loading")
    dataset = load_dataset(target['folder'])

    print("calculating statistics")
    mean, std = compute_statistics(dataset)

    print("Normalizing")
    dataset = normalize_dataset(dataset, mean, std)

    print("Applying stft")
    dataset = apply_stft(dataset, frame_length, frame_step)

    write_tfrecords(dataset, target['label'], target['tfrecord_name'])

