import itertools
import tensorflow as tf
from utils import read_tfrecords
import os

from trainer import Trainer
from config import GlobalConfiguration

from ray import tune
import ray

from model import Model
cfg = GlobalConfiguration()

search_space = {  
    "learning_rate" : [1e-3, 1e-4, 1e-5],
    "learning_rate_decay_steps": [500],
    "learning_rate_decay": [0.98],
    "momentum": [0.9],
    "batch_size": [8, 16, 32, 64],
    "epochs": [50],
    "activation_function": ["ReLU", "LeakyReLU"],
    "dropout_rate": [0.2, 0.5, 0.7],
    "optimizer": ["adam", "sgd"],
    "model": Model
    }


def trainable(config):

    training_dataset = read_tfrecords(os.path.join(cfg.DATASET_FOLDER, cfg.TRAIN_FILE), buffer_size=64000)

    # Fold calculation
    k_folds = cfg.K_FOLDS
    dataset_size = sum(1 for _ in training_dataset)  # Calculate the total number of samples
    fold_size = dataset_size // k_folds  # Calculate the size of each fold

    # Define learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config['learning_rate'],  
        decay_steps=config['learning_rate_decay_steps'],
        decay_rate=config['learning_rate_decay'],
        staircase=True 
    )
    
    # Loss and optimization
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    # Iterate through folds
    for fold_idx in range(k_folds):

        # Create a fresh model for each fold
        net = config['model'](cfg=config, training=True)

        # Create validation dataset for the current fold
        val_dataset = training_dataset.skip(fold_idx * fold_size).take(fold_size)

        # Create training dataset by skipping the validation fold and concatenating the rest
        train_dataset = training_dataset.take(fold_idx * fold_size).concatenate(
            training_dataset.skip((fold_idx + 1) * fold_size)
        )

        # Iterate through epochs
        for _ in config['epochs']:

            # Training passes
            for step, (samples, labels) in enumerate(train_dataset.batch(config['batch_size']).shuffle(buffer_size=1000)):

                with tf.GradientTape() as tape:
                    predictions = net(samples, training=True)
                loss = loss_fn(labels, predictions)
                
                gradients = tape.gradient(loss, net.trainable_weights)
                optimizer.apply_gradients(zip(gradients, net.trainable_weights))

            total_loss = 0.0
            total_accuracy = 0.0
            batches = 0

            # Validation pass
            for samples, labels in val_dataset.batch(config['batch_size']):
            
                # Get loss
                predictions = net(samples, training=False)
                loss = loss_fn(predictions, labels)

                # Calculate accuracy
                probabilities = net(samples, training=False)
                predictions = tf.cast(probabilities > cfg.PROB_THRESHOLD, dtype=tf.int32)
                correct_predictions = tf.equal(predictions, tf.cast(labels, tf.int32))
                
                # Collect statistics
                total_accuracy += tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
                total_loss += loss.numpy()
                batches += 1
            
            # Compute validation metrics
            validation_loss = total_loss / batches
            validation_accuracy = total_accuracy / batches

            tf.print(f"{validation_loss} {validation_accuracy}")


if __name__ == "__main__":
    tuner = tune.Tuner(
        trainable,
        param_space=search_space
    )
    tuner.fit()
    