import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Reshape, Dropout

class HierarchicalRNN(tf.keras.Model):
    def __init__(self, model_config, training, input_shape=None):
        super(HierarchicalRNN, self).__init__()
        
        self.cfg = model_config
        self.training = training
  
        if self.cfg['activation_function'] == 'ReLU':
            self.activation_function = tf.nn.relu
        elif self.cfg['activation_function'] == 'LeakyReLU':
            activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.2)
            self.activation_function = activation_fn
        else:
            tf.print("No activation specified")
            exit()

        self.sequence_length = 20000
        self.chunk_size = 500
        self.num_chunks = self.sequence_length // self.chunk_size  
        self.feature_dim = 1

        self.reshape = Reshape((self.num_chunks, self.chunk_size, self.feature_dim))

        self.first_lstm = TimeDistributed(LSTM(128, return_sequences=False))
        self.second_lstm = LSTM(64, return_sequences=False)

        # Fully connected layers for additional learning
        fc_init = tf.keras.initializers.GlorotNormal(seed=42)
        self.fc1 = tf.keras.layers.Dense(256, activation=self.activation_function, kernel_initializer=fc_init)
        self.drop1 = tf.keras.layers.Dropout(self.cfg['dropout_rate'])
        self.fc2 = tf.keras.layers.Dense(50, activation=self.activation_function, kernel_initializer=fc_init)
        self.out = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=fc_init)

    def call(self, inputs, training=False):
        x = self.reshape(inputs)
        x = self.first_lstm(x)
        x = self.second_lstm(x)
        x = self.fc1(x)
        if self.training:
            x = self.drop1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x
    
