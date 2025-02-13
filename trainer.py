import tensorflow as tf

# Train class for training the model
class Trainer(object):

	def __init__(self, cfg, net):
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
	def compute_accuracy(self, x, y, threshold):
		'''
		Calculates the accuracy given a sample and a label pair.
		input:
			x: tf.Tensor
			y: tf.Tensor
		output:
			accuracy_value: float
		'''

		probabilities = self.net(x, training=False)  # Output shape: (batch_size, 1)
		predictions = tf.cast(probabilities > threshold, dtype=tf.int32)
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
		for samples, labels in dataset:
			predictions = self.net(samples, training=False)
			loss = self.loss_fn(predictions, labels)
			total_loss += loss.numpy()

			accuracy = self.compute_accuracy(samples, labels, self.cfg.PROB_THRESHOLD).numpy()
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

			for step, (samples, labels) in enumerate(trainset.shuffle(buffer_size=1000)):
				self.global_step.assign_add(1)
				g_step = self.global_step.numpy() + 1

				with tf.GradientTape() as tape:
					predictions = self.net(samples, training=True)
					loss = self.loss_fn(labels, predictions)
					
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