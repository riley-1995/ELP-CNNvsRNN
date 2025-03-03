# Configuration used for training
import os

class GlobalConfiguration(object):

	# Data related
	DATASET_FOLDER = '/home/lbutler2/ELP-CNN-Spectrogram/elp_spectrogram_records'

	TRAIN_FILE = 'spectrogram_train_cherrypick.tfrecord'
	VALIDATE_FILE = 'spectrogram_validate_cherrypick.tfrecord'
	TEST_FILE = 'spectrogram_test_cherrypick.tfrecord'

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
	MODEL_FILE = 'model'
	TESTING_IMAGES = 'results'

	# Early Stopping Config
	PATIENCE = 20
	MIN_DELTA = 0.0005

