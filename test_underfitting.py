import os
import tensorflow as tf
from utils import read_tfrecords
from cnn import CNN
from cnn_config import CNNConfig

# Suppress GPU usage for compatibility in some environments
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load a small slice of the training dataset
dataset_path = "/mnt/data/tfrecords_spectrogram/train.tfrecord"
small_train = read_tfrecords(dataset_path).take(20).batch(4).repeat()

# Build and compile model
config = {
    "activation_function": "ReLU",
    "dropout_rate": 0.0,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "learning_rate_decay_steps": 500,
    "learning_rate_decay": 0.97,
    "momentum": 0.0,
    "config": CNNConfig,
}

model = CNN(model_config=config, training=True)
model.build(input_shape=(None, 563, 98, 1))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Fit on small data
print("\\nFitting on small dataset to check for underfitting...")
model.fit(small_train, steps_per_epoch=5, epochs=100)
