import os
import tensorflow as tf
from functools import reduce
from utils import *

tf.config.set_visible_devices([], 'GPU')
print(tf.config.list_physical_devices())

# Define targets
targets = [
    {"folder": os.path.join(os.getcwd(), "training_wav_files/pos"), "label": 1},
    {"folder": os.path.join(os.getcwd(), "training_wav_files/neg"), "label": 0},
    {"folder": os.path.join(os.getcwd(), "testing_wav_files/pos"), "label": 1},
    {"folder": os.path.join(os.getcwd(), "testing_wav_files/neg"), "label": 0},
]

# Load and combine datasets
for target in targets:
    target['dataset'] = load_wavs_into_dataset(target['folder'])

combined_dataset = reduce(lambda d1, d2: d1.concatenate(d2), [t['dataset'] for t in targets])

# Compute statistics
global_mean, global_std = compute_statistics(combined_dataset)

del combined_dataset

# Normalize and label datasets
for target in targets:
    target['dataset'] = add_label(normalize_dataset(target['dataset'], global_mean, global_std), target['label'])

audio_folder = "audio_tfrecords"
os.makedirs(audio_folder, exist_ok=True)

############## Create the training and validation sets
# Combine, shuffle, and split
training_datasets = [t['dataset'] for t in targets if t['folder'].split('/')[-2] == 'training_wav_files']

training_dataset = tf.data.Dataset.concatenate(*training_datasets).shuffle(10000, reshuffle_each_iteration=False)
train, val = stratified_split(training_dataset)

for split, name in zip([train, val], ["train", "validate"]):
    write_tfrecords(split, os.path.join(audio_folder, f"{name}"))

############## Create the testing set

testing_datasets = [t['dataset'] for t in targets if t['folder'].split('/')[-2] == 'testing_wav_files']

testing_dataset = tf.data.Dataset.concatenate(*testing_datasets).shuffle(10000, reshuffle_each_iteration=False)

os.makedirs(audio_folder, exist_ok=True)

write_tfrecords(testing_dataset, os.path.join(audio_folder, f"test"))

