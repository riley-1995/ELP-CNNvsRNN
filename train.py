# Lucas Butler
# Training script

import tensorflow as tf
from utils import read_tfrecords, get_tfrecord_length
import os
import csv

from cnn_config import CNNConfig
from rnn_config import RNNConfig
from cnn import CNN
from rnn import RNN

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

    predicted_labels = tf.cast(predictions >= 0.5, tf.int64)
    correct = tf.equal(predicted_labels, labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    return loss, accuracy

def trainable(config):

    results_dict = {
        "val_loss": 0,
        "val_acc": 0,
        "train_loss": 0,
        "train_acc": 0
    }
    cfg = config['config']
    print(f"Training with config: {config}")
    print(f"config['config']: {config['config']}")
    print(f"Dataset folder: {cfg.DATASET_FOLDER}")

    run_folder = f"training_run_{config['name']}-{os.path.basename(cfg.DATASET_FOLDER)}"

    if not os.path.exists(run_folder):
        os.mkdir(run_folder)

    # set up the output file 
    with open(f"{run_folder}/{cfg.MODEL_FILE}-run-metrics.csv", mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results_dict.keys())
        writer.writeheader()

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
        train_accuracy = 0.0
        batches = 0
        for step, (samples, labels) in enumerate(training_dataset.batch(config['batch_size']).shuffle(buffer_size=dataset_size)):
            loss, acc = train_step(net, optimizer, loss_fn, samples, labels)
            train_loss += loss
            train_accuracy += acc
            batches += 1

        results_dict['train_loss'] = train_loss.numpy()
        results_dict['train_acc'] = (train_accuracy / batches).numpy()
        
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

        results_dict['val_loss'] = validation_loss

        # Check early stopping
        if best_loss - config['min_delta'] > validation_loss:
            best_loss = validation_loss
            patience_counter = 0
            net.save(f'{run_folder}/{cfg.MODEL_FILE}')
            tf.print(f"Best loss updated: {best_loss}, net saved.")
        else:
            patience_counter += 1
            tf.print(f"No loss decrease, patience counter: {patience_counter}")
            if patience_counter > config['patience']:
                tf.print(f"Stopping training.")
                break

        # Compute validation metrics
        validation_accuracy = total_accuracy / batches

        results_dict['val_acc'] = validation_accuracy.numpy()

        with open(f"{run_folder}/{cfg.MODEL_FILE}-run-metrics.csv", mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(results_dict.values())

        tf.print(f"Validation loss: {validation_loss:.2f} Validation Accuracy: {validation_accuracy:.2f}")
        tf.print(f"Train loss: {results_dict['train_loss']:.2f} Train Accuracy: {results_dict['train_acc']:.2f}")


if __name__ == '__main__':

    cnn = False
    # CNN
    if cnn: 
        training_config = {  
            "learning_rate": 0.0001,
            "learning_rate_decay_steps": 200,
            "learning_rate_decay": 1.0,
            "momentum": 0.5,
            "batch_size": 8,
            "epochs": 300,
            "activation_function": "LeakyReLU",
            "dropout_rate": 0.2,
            "optimizer": "sgd",
            "model": CNN,
            "config": CNNConfig,
            "patience": 10,
            "min_delta": 0.001,
            "name": "cnn"
        }

    # RNN
    else:

        training_config = {  
            "learning_rate": 0.001,
            "learning_rate_decay_steps": 200,
            "learning_rate_decay": 0.97,
            "momentum": 0.9,
            "batch_size": 8,
            "epochs": 300,
            "activation_function": "ReLU",
            "dropout_rate": 0.2,
            "optimizer": "adam",
            "model": RNN,
            "config": RNNConfig,
            "patience": 10,
            "min_delta": 0.001,
            "name": "rnn"
        }

    trainable(training_config)
