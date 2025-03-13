# Configuration used for training
import os

class RNNConfig(object):

	# Data related
	DATASET_FOLDER = '/home/lbutler2/ELP-CNN-Spectrogram/data/audio_records_large'

	TRAIN_FILE = 'train.tfrecord'
	VALIDATE_FILE = 'validate.tfrecord'
	TEST_FILE = 'test.tfrecord'

	K_FOLDS=5
	MAX_CV_EPOCHS=5

	# Prediction
	NUM_CLASSES = 1
	PROB_THRESHOLD = 0.5

	# Display steps
	TRAIN_STEP = 10
	VALIDATION_STEP = 50
	SAVE_STEP = 5000

	# Paths for Output
	SUMMARY_PATH = 'results/summary'
	LOG_DIR = 'results/logs'
	MODEL_FILE = 'rnn_model'
	TESTING_IMAGES = 'rnn_results'

	# Early Stopping Config
	PATIENCE = 20
	MIN_DELTA = 0.0005

