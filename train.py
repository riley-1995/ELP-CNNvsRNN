# Lucas Butler
# Training script

import time
import datetime
import gc
import itertools

from alexnet import AlexNet
import tensorflow as tf

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from trainer import Trainer
from config import Configuration

tf.random.set_seed(1)	# For deterministic ops

# Get the list of GPUs
# gpus = tf.config.list_physical_devices('GPU')

def get_tfrecord_length(dataset):
	count = 0
	for d in dataset:
		count += 1	
	return count

# if gpus:
#     try:
#         # Set memory growth to True for each GPU
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

def read_tfrecords(file_name, buffer_size=1000):
        '''
        Input:
            file_name:  File name to read records from.
        Output:
            dataset:    TFRecordDataset.
        '''
        
        features = {
            'sample': tf.io.FixedLenFeature([], tf.string),  
            'label': tf.io.FixedLenFeature([], tf.string)
        }
        
        def _parse_function(example_proto):
            """Parse a serialized Example."""
            parsed = tf.io.parse_single_example(example_proto, features)
            # Deserialize tensors
            sample = tf.io.parse_tensor(parsed['sample'], out_type=tf.float32)
            label = tf.io.parse_tensor(parsed['label'], out_type=tf.int32)

            return sample, label
        
        data = tf.data.TFRecordDataset(file_name, buffer_size=buffer_size)
        dataset = data.map(_parse_function)

        return dataset

if __name__ == '__main__':
	i = 0

	# Make dir for logs
	if not os.path.exists("logs"):
		os.makedirs('logs')
	while os.path.exists("logs/log%s.txt" % i):
		i += 1

	# Initialize log path
	LOG_PATH = "logs/log%s.txt" % i
	def print(msg):
		with open(LOG_PATH,'a') as f:
			f.write(f'{time.ctime()}: {msg}\n')

	# Global configuration settings for training and testing
	cfg = Configuration()

	train = read_tfrecords(os.path.join(cfg.DATASET_FOLDER, cfg.TRAIN_FILE), buffer_size=64000)
	tf.print(f"Number of train records: {get_tfrecord_length(train)}")

	cross_validate = cfg.CROSS_VALIDATE
	if cross_validate:
		batch_sizes = [32, 64, 128]

		# Enumerate all possible options of hyperparameters
		hyperparameters = list(itertools.product(batch_sizes))

		# Helpful for viewing what we are doing
		for p in hyperparameters:
			tf.print(p)

		k_folds = cfg.K_FOLDS
		dataset_size = sum(1 for _ in train)  # Calculate the total number of samples
		fold_size = dataset_size // k_folds  # Calculate the size of each fold
		fold_config_results = []

		for parameters in hyperparameters:
			fold_results = []

			for fold_idx in range(k_folds):
				# Create a fresh model for each fold
				net = AlexNet(cfg=cfg, training=True)
				trainer = Trainer(cfg=cfg, net=net)

				# Create validation dataset for the current fold
				val_dataset = train.skip(fold_idx * fold_size).take(fold_size)

				# Create training dataset by skipping the validation fold and concatenating the rest
				train_dataset = train.take(fold_idx * fold_size).concatenate(
					train.skip((fold_idx + 1) * fold_size)
				)

				# Run training for this fold
				fold_val_loss = trainer.train(
					trainset=train_dataset.batch(parameters[0]), 
					valset=val_dataset.batch(parameters[0]), 
					cross_validate=True, 
					epochs=cfg.MAX_CV_EPOCHS
					)
				
				tf.print(f'Cross validation fold {fold_idx} loss: {fold_val_loss}')
				fold_results.append(fold_val_loss)

				# Clean up
				del train_dataset
				del val_dataset
				del net
				del trainer
				tf.keras.backend.clear_session()
				gc.collect()

			# Store the average loss for this set of parameters
			avg_val_loss = sum(fold_results) / len(fold_results)
			print(f"Cross-validation average validation loss: {avg_val_loss}")
			fold_config_results.append((parameters, avg_val_loss))

		# Best configuration
		best_hyperparameters = max(fold_config_results, key=lambda x: x[1])
		print(f"Best configuration:\n{best_hyperparameters}")

		# Best parameter assignments, only using batchsize for now
		batch_size = best_hyperparameters[0]
	else:
		batch_size = 1

	validate = read_tfrecords(os.path.join(cfg.DATASET_FOLDER, cfg.VALIDATE_FILE), buffer_size=10000)
	tf.print(f"Number of validate records: {get_tfrecord_length(validate)}")

	# Tensorboard start
	run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
	log_dir = os.path.join(cfg.LOG_DIR, run_name)
	tensorboard_writer = tf.summary.create_file_writer(log_dir)

	# Log the chosen parameters 
	with tensorboard_writer.as_default():
		tf.summary.text("Batch Size", str(batch_size), step=0)

	# Load the model and trainer
	net = AlexNet(cfg=cfg, training=True)
	trainer = Trainer(cfg=cfg, net=net)

	# Call train function on trainer class
	trainer.train(trainset=train.batch(batch_size), valset=validate.batch(batch_size), cross_validate=False, tensorboard_writer=tensorboard_writer, max_epochs=cfg.EPOCHS)
