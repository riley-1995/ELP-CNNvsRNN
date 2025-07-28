import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from utils import read_tfrecords
import tensorflow as tf

tfrecord_path = "/mnt/data/tfrecords_spectrogram/train.tfrecord"

counts = {0: 0, 1: 0}
for _, label in read_tfrecords(tfrecord_path).take(1000):
    val = int(tf.round(label).numpy())
    counts[val] += 1

print("Label counts (rounded):", counts)
