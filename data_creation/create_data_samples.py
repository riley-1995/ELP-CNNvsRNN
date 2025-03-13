import os
import tensorflow as tf
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

tf.config.set_visible_devices([], 'GPU')
print(tf.config.list_physical_devices())

def load_wav_file(file_path):
    raw_audio = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(raw_audio, desired_channels=1)
    return tf.squeeze(audio, axis=-1)

def load_dataset(directory):
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

def write_tfrecords(dataset, file_prefix):
    filename = f'{file_prefix}.tfrecord'
    with tf.io.TFRecordWriter(filename) as writer:
        for sample, label in dataset:
            feature = {
                'sample': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(sample).numpy()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label.numpy()]))
            }
            writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())

def stratified_split(dataset, test_size=0.2, val_size=0.1):
    samples, labels = zip(*dataset.as_numpy_iterator())
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=test_size, stratify=labels, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1 - test_size), stratify=y_train, random_state=42)
    return map(tf.data.Dataset.from_tensor_slices, [(X_train, y_train), (X_val, y_val), (X_test, y_test)])

def main():
  # Define targets
  targets = [
      {"folder": os.path.join(os.getcwd(), "wav_files/elephant_raw_audio"), "label": 1},
      {"folder": os.path.join(os.getcwd(), "wav_files/non_elephant_raw_audio"), "label": 0},
  ]
  
  # Load and combine datasets
  for target in targets:
      target['dataset'] = load_dataset(target['folder'])
  combined_dataset = tf.data.Dataset.concatenate(*[t['dataset'] for t in targets])
  
  # Compute statistics
  global_mean, global_std = compute_statistics(combined_dataset)
  
  del combined_dataset
  
  # Normalize and label datasets
  for target in targets:
      target['dataset'] = add_label(normalize_dataset(target['dataset'], global_mean, global_std), target['label'])
  
  # Combine, shuffle, and split
  dataset = tf.data.Dataset.concatenate(*[t['dataset'] for t in targets]).shuffle(10000, reshuffle_each_iteration=False)
  train, val, test = stratified_split(dataset)
  
  audio_folder = "audio_tfrecords"
  os.makedirs(audio_folder, exist_ok=True)
  
  for split, name in zip([train, val, test], ["train", "validate", "test"]):
      write_tfrecords(split, os.path.join(audio_folder, f"audio_{name}"))

if __name__ == "__main__":
  main()

