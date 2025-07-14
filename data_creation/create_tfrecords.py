import os
import tensorflow as tf
from functools import reduce
from utils import *
from data_path_config import DataPathConfig

tf.config.set_visible_devices([], 'GPU')
print(tf.config.list_physical_devices())

paths = DataPathConfig()

# Define inputs based on the data paths
inputs = [
    {"folder": paths.POS_TRAIN_VAL_CLIPS_DIR, "label": 1, "split": "train"},
    {"folder": paths.TRAIN_VAL_NEG_CLIPS_DIR, "label": 0, "split": "train"},
    {"folder": paths.POS_HOLDOUT_TEST_CLIPS_DIR, "label": 1, "split": "test"},
    {"folder": paths.HOLDOUT_TEST_NEG_CLIPS_DIR, "label": 0, "split": "test"},
]

# Create output folder for audio TFRecords
output_audio_folder = paths.TFRECORDS_AUDIO_DIR
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

'''
***** Choose whether to create a validation set or not. *****

- If you do not want to create a validation set, use Option 1 and comment out Option 2.
- If you want to create a validation set, use Option 2 and comment out Option 1.

Note: The option you use depends on your training strategy and whether you want to validate your model 
during training. If using cross-validation, you may not need a separate validation set and it would be
better to use all available data for training. If you are not using cross-validation, it is recommended 
to create a validation set.
'''

# Option 1 - No validation set - Better for cross-validation
# write_tfrecords(training_dataset, os.path.join(output_audio_folder, "train"))

# Option 2 - Create a validation set - Better for regular (non-cross-validation) training
train, val = stratified_split(training_dataset)
for split, name in zip([train, val], ["train", "validate"]):
    write_tfrecords(split, os.path.join(output_audio_folder, f"{name}"))

############## Create the testing set

testing_datasets = [input['dataset'] for input in inputs if input['split'] == 'test']

testing_dataset = tf.data.Dataset.concatenate(*testing_datasets).shuffle(10000, reshuffle_each_iteration=False)

write_tfrecords(testing_dataset, os.path.join(output_audio_folder, f"test"))