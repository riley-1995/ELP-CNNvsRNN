# Configuration used for training
import os

class Configuration(object):

	# Data related
	DATABASE_ROOT_PATH = 'ptb-xl'	# Location of the ptb-xl dataset
	DATABASE_FILE_NAME = 'updated_ptbxl_database.json'
	DATASET_STORAGE = 'data_storage'
	DATASET_FOLDER = '3_lead_data_2_label_abnormal_1_label'

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
