# inspect_tfrecord.py
import tensorflow as tf
from utils import read_tfrecords
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Path inside the container
tfrecord_path = "/mnt/data/tfrecords_spectrogram/train.tfrecord"

# Load dataset
dataset = read_tfrecords(tfrecord_path)

# Inspect a few samples
for i, (spectrogram, label) in enumerate(dataset.take(5)):
    print(f"\nExample {i+1}")
    print("Spectrogram shape:", spectrogram.shape)
    print("Spectrogram dtype:", spectrogram.dtype)
    print("Label value:", label.numpy())
    print("Label shape:", label.shape)
    print("Label dtype:", label.dtype)
