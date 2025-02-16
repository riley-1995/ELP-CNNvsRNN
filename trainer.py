import tensorflow as tf
from config import GlobalConfiguration

cfg = GlobalConfiguration()

# Train class for training the model
class Trainer(object):

	def __init__(self, config, net):
		# Configuration and Model
		self.config = config
		self.net = net

		# Define learning rate schedule
		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
			initial_learning_rate=self.config['learning_rate'],  
			decay_steps=self.config['learning_rate_decay_steps'],
			decay_rate=self.config['learning_rate_decay'],
			staircase=True 
		)

		self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
		self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
		self.global_step = tf.Variable(0, name='global_step', dtype=tf.int64, trainable=False)

	def compute_loss(self, x, y, training):
		pred = self.net(x, training=training)		
		loss_value = self.loss_fn(y, pred)
		return loss_value

	# Accuracy Calulations
	def compute_accuracy(self, x, y, threshold):
		probabilities = self.net(x, training=False)  # Output shape: (batch_size, 1)
		predictions = tf.cast(probabilities > threshold, dtype=tf.int32)
		correct_predictions = tf.equal(predictions, tf.cast(y, tf.int32))
		accuracy_value = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
		return accuracy_value

	def calc_metrics_on_dataset(self, dataset: tf.data.Dataset):
		total_loss = 0.0
		total_accuracy = 0.0
		batches = 0
		for samples, labels in dataset:
			predictions = self.net(samples, training=False)
			loss = self.loss_fn(predictions, labels)
			total_loss += loss.numpy()

			accuracy = self.compute_accuracy(samples, labels, cfg.PROB_THRESHOLD).numpy()
			total_accuracy += accuracy
			batches += 1
			
		loss = total_loss / batches
		accuracy = total_accuracy / batches
		return loss, accuracy

	def train(self, trainset, valset, tensorboard_writer=None, cross_validate=False, max_epochs=None):
		self.best_val_loss = float('inf')
		self.patience_counter = 0
		self.cross_validate = cross_validate

		for e in range(0, self.config['epochs'] if not max_epochs else max_epochs):

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