import tensorflow as tf

# Class for Alexnet model
class SmallCNN(tf.keras.Model):

	def __init__(self, cfg, training):
		super(SmallCNN, self).__init__()

		self.cfg = cfg
		self.training = training

		# Convolutional layers 1 to 5
		conv_init = tf.compat.v1.glorot_normal_initializer()
		
		# Adaptive height, width, and channels 
		self.conv1 = tf.keras.Sequential([
			tf.keras.layers.Conv2D(64, 3, 3, 'same', kernel_initializer=conv_init),
			tf.keras.layers.Activation(tf.nn.relu)
		])
		self.pool1 = tf.keras.layers.MaxPooling2D(2, 2, 'VALID')

		self.conv2 = tf.keras.Sequential([
			tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=conv_init),
			tf.keras.layers.Activation(tf.nn.relu)
		])
		self.pool2 = tf.keras.layers.MaxPooling2D(2, 2, 'VALID')

		self.conv3 = tf.keras.Sequential([
			tf.keras.layers.Conv2D(256, 3, 1, 'same', kernel_initializer=conv_init),
			tf.keras.layers.Activation(tf.nn.relu)
		])
		
		self.pool3 = tf.keras.layers.MaxPooling2D(2, 2, 'VALID')

		self.conv4 = tf.keras.Sequential([
			tf.keras.layers.Conv2D(256, 3, 1, 'same', kernel_initializer=conv_init),
			tf.keras.layers.Activation(tf.nn.relu)
		])
		
		self.pool4 = tf.keras.layers.MaxPooling2D(2, 2, 'VALID')

		# Fully connected layers
		fc_init = tf.compat.v1.glorot_normal_initializer()

		self.fc1 = tf.keras.layers.Dense(256, activation=tf.nn.relu, kernel_initializer=fc_init)
		self.drop1 = tf.keras.layers.Dropout(self.cfg.DROPOUT)

		self.fc2 = tf.keras.layers.Dense(50, activation=tf.nn.relu, kernel_initializer=fc_init)

		# Sigmoid for multilabel classification
		self.out = tf.keras.layers.Dense(self.cfg.NUM_CLASSES, activation='sigmoid', kernel_initializer=fc_init)


	def call(self, x):
		# Function that executes the model on call

		output = self.conv1(x)
		output = self.pool1(output)

		output = self.conv2(output)
		output = self.pool2(output)

		output = self.conv3(output)
		output = self.pool3(output)
		
		output = self.conv4(output)
		output = self.pool4(output)

		output = tf.keras.layers.Flatten()(output)

		output = self.fc1(output)

		# Execute dropout if training
		if self.training:
			output = self.drop1(output)

		output = self.fc2(output)

		output = self.out(output)

		return output