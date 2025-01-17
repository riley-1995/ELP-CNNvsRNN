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

from data import PTBXLDataset
from config import Configuration

tf.random.set_seed(1)	# For deterministic ops

# Get the list of GPUs
gpus = tf.config.list_physical_devices('GPU')

def get_tfrecord_length(dataset):
	count = 0
	for d in dataset:
		count += 1	
	return count

if gpus:
    try:
        # Set memory growth to True for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Train class for training the model
class Trainer(object):

	def __init__(self, cfg, net, resume=False):
		# Configuration and Model
		self.cfg = cfg
		self.net = net

		# Define learning rate schedule
		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
			initial_learning_rate=self.cfg.LEARNING_RATE,  
			decay_steps=self.cfg.LEARNING_RATE_DECAY_STEPS,
			decay_rate=self.cfg.LEARNING_RATE_DECAY,
			staircase=True 
		)

		self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
		self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
		self.global_step = tf.Variable(0, name='global_step', dtype=tf.int64, trainable=False)

	def predict(self, x):
		"""
		Predicts the class labels using softmax probabilities.
		Input:
			x: tf.Tensor
		Output:
			predictions: tf.Tensor in one-hot encoded format
		"""
		probabilities = self.net(x)  # Output shape: (batch_size, num_classes)
		predictions = tf.cast(probabilities > 0.5, dtype=tf.int32)
		return predictions
	
	def compute_loss(self, x, y, training):
		'''
		Computes the loss between actual and predicted labels
		input:
			x: tf.Tensor
			y: tf.Tensor
		output:
			loss_value: float
		'''
		pred = self.net(x, training=training)		
		loss_value = self.loss_fn(y, pred)
		return loss_value

	# Accuracy Calulations
	def compute_accuracy(self, x, y):
		'''
		Calculates the accuracy given a sample and a label pair.
		input:
			x: tf.Tensor
			y: tf.Tensor
		output:
			accuracy_value: float
		'''

		probabilities = self.net(x, training=False)  # Output shape: (batch_size, 1)
		predictions = tf.cast(probabilities > 0.5, dtype=tf.int32)
		correct_predictions = tf.equal(predictions, tf.cast(y, tf.int32))
		accuracy_value = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
		return accuracy_value

	def calc_metrics_on_dataset(self, dataset: tf.data.Dataset):
		'''
		Calculates the loss and accuracy of the model on a dataset
		input:
			data: tf.data.Dataset
		output:
			loss: float
			accuracy: float
		'''
		total_loss = 0.0
		total_accuracy = 0.0
		batches = 0
		for images, labels in dataset:
			loss = self.compute_loss(images, labels, training=False)
			accuracy = self.compute_accuracy(images, labels).numpy()
			total_loss += loss.numpy()
			total_accuracy += accuracy
			batches += 1
		loss = total_loss / batches
		accuracy = total_accuracy / batches
		return loss, accuracy
	
	def log_metric_pairs(self, loss, acc, var_name, tensorboard_writer, epoch):
		'''
		Writes the loss and accuracy for a variable to Tensorboard
		input:
			loss: float
			acc: float
			var_name: string
			tensorboard_write: tf.summary.writer
			epoch: int
		output:
			None
		'''
		with tensorboard_writer.as_default():
			tf.print(f"{var_name}_acc {acc:.3f} {var_name}_loss {loss:.3f}")
			tf.summary.scalar(f'{var_name}_acc', acc, step=epoch)
			tf.summary.scalar(f'{var_name}_loss', loss, step=epoch)

	def check_early_stopping(self, val_loss):
		'''
		Checks if the validation loss from the last step has improved. Uses
			patience to determine when to stop training. Saves the model
			with the best validation loss. Returns true when it is time to
			stop training.
		input:
			val_loss: float
		output:
			bool
		'''
		if val_loss < self.best_val_loss - self.cfg.MIN_DELTA:
			self.best_val_loss = val_loss
			self.patience_counter = 0
			if not self.cross_validate:
				self.net.save(self.cfg.MODEL_FILE)
				tf.print(f"Validation loss improved to {val_loss:.4f}. Saved model weights.")
		else:
			self.patience_counter += 1
			tf.print(f"No improvement in validation loss for {self.patience_counter} epoch(s).")
		if self.patience_counter >= self.cfg.PATIENCE:
			tf.print(f"Early stopping triggered...")
			return True
		return False

	def train(self, trainset, valset, tensorboard_writer=None, cross_validate=False, max_epochs=None):
		'''
		Trains the model on the training set and validates each epoch
			with the validation set. Will write to tensorboard and 
			save the model if cross_validate is false.
		input:
			trainset: tf.data.Dataset
			valset: tf.data.Dataset
			tensorboard_writer: tf.summary.writer
			cross_validate: bool
			epochs: int
		output:
			best_val_loss: float

		'''
		self.best_val_loss = float('inf')
		self.patience_counter = 0
		self.cross_validate = cross_validate

		for e in range(0, self.cfg.EPOCHS if not max_epochs else max_epochs):

			for step, (sample, labels) in enumerate(trainset.shuffle(buffer_size=1000)):
				self.global_step.assign_add(1)
				g_step = self.global_step.numpy() + 1

				with tf.GradientTape() as tape:
					loss = self.compute_loss(sample, labels, training=True)
					
				gradients = tape.gradient(loss, self.net.trainable_weights)
				self.optimizer.apply_gradients(zip(gradients, self.net.trainable_weights))

				if tensorboard_writer:
					current_lr = self.optimizer.learning_rate
					with tensorboard_writer.as_default():
						tf.summary.scalar("learning_rate", current_lr, step=self.global_step.numpy())
			
			val_loss, val_acc = self.calc_metrics_on_dataset(valset)

			if not cross_validate and tensorboard_writer:
				train_loss, train_acc = self.calc_metrics_on_dataset(trainset)
				self.log_metric_pairs(loss=val_loss, acc=val_acc, var_name='validate', tensorboard_writer=tensorboard_writer, epoch=e)
				self.log_metric_pairs(loss=train_loss, acc=train_acc, var_name='train', tensorboard_writer=tensorboard_writer, epoch=e)

			if self.check_early_stopping(val_loss):
				break

		return self.best_val_loss

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

	# Loads the dataset
	dataset = PTBXLDataset(cfg=cfg)
	
	train = dataset.read_tfrecords('train.tfrecord', buffer_size=64000)
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
		batch_size = 128

	validate = dataset.read_tfrecords('validate.tfrecord', buffer_size=10000)
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
