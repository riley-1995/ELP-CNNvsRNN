import os
import tensorflow as tf
from functools import reduce
from utils import *
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

tf.config.set_visible_devices([], 'GPU')
print(tf.config.list_physical_devices())

# Define inputs
inputs = [
    {"folder": os.path.join(PROJECT_ROOT, "data", "training_clips_pos"), "label": 1, "split": "train"},
    {"folder": os.path.join(PROJECT_ROOT, "data", "training_clips_neg"), "label": 0, "split": "train"},
    {"folder": os.path.join(PROJECT_ROOT, "data", "testing_clips_pos"),  "label": 1, "split": "test"},
    {"folder": os.path.join(PROJECT_ROOT, "data", "testing_clips_neg"),  "label": 0, "split": "test"},
]

output_audio_folder = os.path.join(PROJECT_ROOT, "data", "audio_tfrecords")
os.makedirs(output_audio_folder, exist_ok=True)

# Load and combine datasets
for input in inputs:
    input['dataset'] = load_wavs_into_dataset(input['folder'])

combined_dataset = reduce(lambda d1, d2: d1.concatenate(d2), [input['dataset'] for input in inputs])

# Compute statistics
global_mean, global_std = compute_statistics(combined_dataset)

del combined_dataset

# Normalize and label datasets
for input in inputs:
    input['dataset'] = add_label(normalize_dataset(input['dataset'], global_mean, global_std), input['label'])

############## Create the training and validation sets
# Combine, shuffle, and split
training_datasets = [input['dataset'] for input in inputs if input['split'] == 'train']

training_dataset = tf.data.Dataset.concatenate(*training_datasets).shuffle(20000, reshuffle_each_iteration=False)
train, val = stratified_split(training_dataset)

for split, name in zip([train, val], ["train", "validate"]):
    write_tfrecords(split, os.path.join(output_audio_folder, f"{name}"))

############## Create the testing set

testing_datasets = [input['dataset'] for input in inputs if input['split'] == 'test']

testing_dataset = tf.data.Dataset.concatenate(*testing_datasets).shuffle(10000, reshuffle_each_iteration=False)

write_tfrecords(testing_dataset, os.path.join(output_audio_folder, f"test"))

