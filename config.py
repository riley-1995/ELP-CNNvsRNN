# Configuration used for training
import os

class Configuration(object):

	# Data related
	DATASET_FOLDER = 'spectrogram_tfrecords_cherrypick'

	TRAIN_FILE = 'spectrogram_train_cherrypick.tfrecord'
	VALIDATE_FILE = 'spectrogram_validate_cherrypick.tfrecord'
	TEST_FILE = 'spectrogram_test_cherrypick.tfrecord'

	CROSS_VALIDATE = False
	K_FOLDS=5
	MAX_CV_EPOCHS=5

	# Prediction
	NUM_CLASSES = 1
	PROB_THRESHOLD = 0.5

	# Training hyperparameters
	LEARNING_RATE = 1e-4
	LEARNING_RATE_DECAY = 0.98
	LEARNING_RATE_DECAY_STEPS = 500
	MOMENTUM = 0.9
	BATCH_SIZE = 64
	EPOCHS = 50

	# Display steps
	TRAIN_STEP = 10
	VALIDATION_STEP = 50
	SAVE_STEP = 5000

	# Paths for Output
	SUMMARY_PATH = 'results/summary'
	LOG_DIR = 'results/logs'
	MODEL_FILE = 'results/model.keras'
	TESTING_IMAGES = 'results'

	# Early Stopping Config
	PATIENCE = 50
	MIN_DELTA = 0.0005
	RESTORE_BEST_WEIGHTS = False

	# Net architecture hyperparamaters
	LAMBDA = 5e-4 #for weight decay
	DROPOUT = 0.5
