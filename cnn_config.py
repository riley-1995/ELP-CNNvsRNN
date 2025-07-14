# Configuration used for training
import os
from pathlib import Path
from data_creation.data_path_config import DataPathConfig


class CNNConfig(object):

	paths = DataPathConfig()
	# Data related
	DATASET_FOLDER = paths.TFRECORDS_SPECTROGRAM_DIR

	TRAIN_FILE = 'train.tfrecord'
	VALIDATE_FILE = 'validate.tfrecord'
	TEST_FILE = 'test.tfrecord'

	K_FOLDS=5
	MAX_CV_EPOCHS=5

	# Prediction
	NUM_CLASSES = 1
	PROB_THRESHOLD = 0.5

	# Paths for Output
	SUMMARY_PATH = 'results/summary'
	LOG_DIR = 'results/logs'
	MODEL_FILE = 'cnn_model'
	TESTING_IMAGES = 'cnn_results'

