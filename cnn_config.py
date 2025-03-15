# Configuration used for training
import os

class CNNConfig(object):

	# Data related
	DATASET_FOLDER = '/home/lbutler2/ELP-CNN-Spectrogram/data/spectrogram_tfrecords'

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

