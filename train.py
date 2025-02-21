# Lucas Butler
# Training script

import tensorflow as tf
from utils import read_tfrecords, get_tfrecord_length
from model import Model
import os
from config import GlobalConfiguration
cfg = GlobalConfiguration()

tf.random.set_seed(1)

# Get the list of GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to True for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

@tf.function
def train_step(net, optimizer, loss_fn, samples, labels):
    with tf.GradientTape() as tape:
        predictions = net(samples, training=True)
        loss = loss_fn(labels, predictions)
    
    gradients = tape.gradient(loss, net.trainable_weights)
    optimizer.apply_gradients(zip(gradients, net.trainable_weights))

    return loss

def trainable(config):

    # Load datasets
    training_dataset = read_tfrecords(os.path.join(cfg.DATASET_FOLDER, cfg.TRAIN_FILE), buffer_size=64000)
    validation_dataset = read_tfrecords(os.path.join(cfg.DATASET_FOLDER, cfg.VALIDATE_FILE), buffer_size=64000)

    # Get shape and dataset size
    for sample, label in training_dataset.take(1):
        shape = [None] + sample.shape
    tf.print(shape)
    dataset_size = get_tfrecord_length(training_dataset)

    tf.print(f"Number of train records: {dataset_size}")
    tf.print(f"Number of validate records: {get_tfrecord_length(validation_dataset)}")

    # Define learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config['learning_rate'],  
        decay_steps=config['learning_rate_decay_steps'],
        decay_rate=config['learning_rate_decay'],
        staircase=True 
    )
    
    # Loss Function
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    # Create a fresh model for each fold
    net = config['model'](model_config=config, training=True)
    net.build(shape)

    # Optimization
    if config["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    elif config["optimizer"] == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=config["momentum"])

    optimizer.build(net.trainable_weights)

    patience_counter = 0
    best_loss = float('inf')

    # Iterate through epochs
    for _ in range(config['epochs']):

        # Training passes
        train_loss = 0.0
        for step, (samples, labels) in enumerate(training_dataset.batch(config['batch_size']).shuffle(buffer_size=dataset_size)):
            train_loss += train_step(net, optimizer, loss_fn, samples, labels)

        # Validation runs
        validation_loss = 0.0
        total_accuracy = 0.0
        batches = 0
        for samples, labels in validation_dataset.batch(config['batch_size']):
            # Get loss
            predictions = net(samples, training=False)
            validation_loss += loss_fn(labels, predictions).numpy()

            # Calculate accuracy
            pred_classes = tf.cast(predictions > cfg.PROB_THRESHOLD, dtype=tf.int32)
            correct_predictions = tf.equal(pred_classes, tf.cast(labels, tf.int32))
            
            total_accuracy += tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            batches += 1
        
        # Check early stopping
        if best_loss - config['min_delta'] > validation_loss:
            best_loss = validation_loss
            patience_counter = 0
            net.save('model')
            tf.print(f"Best loss updated: {best_loss}, net saved.")
        else:
            patience_counter += 1
            tf.print(f"No loss decrease, patience counter: {patience_counter}")
            if patience_counter > config['patience']:
                tf.print(f"Stopping training.")
                break

        # Compute validation metrics
        validation_accuracy = total_accuracy / batches

        tf.print(f"Validation loss: {validation_loss:.2f} Validation Accuracy: {validation_accuracy:.2f}")


if __name__ == '__main__':

    training_config = {  
        "learning_rate": 0.0001,
        "learning_rate_decay_steps": 500,
        "learning_rate_decay": 0.98,
        "momentum": 0.9,
        "batch_size": 8,
        "epochs": 50,
        "activation_function": "ReLU",
        "dropout_rate": 0.7,
        "optimizer": "sgd",
        "model": Model,
        "patience": 10,
        "min_delta": 0.001,
    }

    trainable(training_config)