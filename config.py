# Configuration used for training
import os

class Configuration(object):

	# Data related
	DATASET_FOLDER = 'stft_records'

	TRAIN_FILE = 'train.tfrecord'
	VALIDATE_FILE = 'validate.tfrecord'
	TEST_FILE = 'test.tfrecord'

	CROSS_VALIDATE = False
	K_FOLDS=5
	MAX_CV_EPOCHS=5

	NUM_CLASSES = 1

	# Train/Validate/Text Split
	TRAIN_PERCENTAGE = 0.70
	TEST_PERCENTAGE = 0.15
	VALIDATE_PERCENTAGE = 0.15

	# Training hyperparameters
	LEARNING_RATE = 1e-4
	LEARNING_RATE_DECAY = 0.98
	LEARNING_RATE_DECAY_STEPS = 500
	MOMENTUM = 0.9
	BATCH_SIZE = 128
	EPOCHS = 50

	# Display steps
	TRAIN_STEP = 10
	VALIDATION_STEP = 50
	SAVE_STEP = 5000

	# Paths for checkpoint
	SUMMARY_PATH = 'summary'
	LOG_DIR = 'logs'
	MODEL_FILE = 'model.keras'	

	# Early Stopping Config
	PATIENCE = 7
	MIN_DELTA = 0.0005
	RESTORE_BEST_WEIGHTS = False

	# Net architecture hyperparamaters
	LAMBDA = 5e-4 #for weight decay
	DROPOUT = 0.5
