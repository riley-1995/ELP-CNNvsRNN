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

def k_fold_split(dataset, num_folds, fold_idx):
    dataset = dataset.enumerate().cache().prefetch(tf.data.AUTOTUNE)

    # Validation dataset: Select elements belonging to the current fold
    val_dataset = dataset.filter(lambda i, data: i % num_folds == fold_idx).map(lambda i, data: data)

    # Training dataset: Select elements NOT in the current fold
    train_dataset = dataset.filter(lambda i, data: i % num_folds != fold_idx).map(lambda i, data: data)

    return train_dataset, val_dataset

@tf.function
def train_step(net, optimizer, loss_fn, samples, labels):
    with tf.GradientTape() as tape:
        predictions = net(samples, training=True)
    loss = loss_fn(labels, predictions)
    
    gradients = tape.gradient(loss, net.trainable_weights)
    optimizer.apply_gradients(zip(gradients, net.trainable_weights))

    return loss

search_space = {  
    "learning_rate": tune.loguniform(1e-5, 1e-3),
    "learning_rate_decay_steps": tune.choice([500]),
    "learning_rate_decay": tune.choice([0.98]),
    "momentum": tune.choice([0.9]),
    "batch_size": tune.choice([8, 16, 32, 64]),
    "epochs": tune.choice([50]),
    "activation_function": tune.choice(["ReLU", "LeakyReLU"]),
    "dropout_rate": tune.choice([0.2, 0.5, 0.7]),
    "optimizer": tune.choice(["adam", "sgd"]),
    "model": tune.choice([Model])
}

def trainable(config):

    training_dataset = read_tfrecords(os.path.join(cfg.DATASET_FOLDER, cfg.TRAIN_FILE), buffer_size=64000)

    # Fold calculation
    k_folds = cfg.K_FOLDS
    dataset_size = sum(1 for _ in training_dataset)
    fold_size = dataset_size // k_folds

    # Define learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config['learning_rate'],  
        decay_steps=config['learning_rate_decay_steps'],
        decay_rate=config['learning_rate_decay'],
        staircase=True 
    )
    
    # Optimization
    if config["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    elif config["optimizer"] == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=config["momentum"])

    # Loss Function
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    
    # store results
    fold_loss_results = []
    fold_accuracy_results = []

    # Iterate through folds
    for fold_idx in range(k_folds):

        # Create a fresh model for each fold
        net = config['model'](model_config=config, training=True)

        train_dataset, val_dataset = k_fold_split(training_dataset, cfg.K_FOLDS, fold_idx)

        # Iterate through epochs
        for _ in config['epochs']:

            # Training passes
            for step, (samples, labels) in enumerate(train_dataset.batch(config['batch_size']).shuffle(buffer_size=dataset_size)):

                loss = train_step(net, optimizer, loss_fn, samples, labels)

        # Validation runs
        total_loss = 0.0
        total_accuracy = 0.0
        batches = 0
        for samples, labels in val_dataset.batch(config['batch_size']):
            # Get loss
            predictions = net(samples, training=False)
            loss = loss_fn(labels, predictions)

            # Calculate accuracy
            pred_classes = tf.cast(predictions > cfg.PROB_THRESHOLD, dtype=tf.int32)
            correct_predictions = tf.equal(pred_classes, tf.cast(labels, tf.int32))
            
            # Collect statistics
            total_accuracy += tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            total_loss += loss.numpy()
            batches += 1
        
        # Compute validation metrics
        validation_loss = total_loss / batches
        validation_accuracy = total_accuracy / batches

        fold_loss_results.append(validation_loss)
        fold_accuracy_results.append(validation_accuracy)


    avg_loss = sum(fold_loss_results)/len(fold_loss_results)
    avg_acc = sum(fold_accuracy_results)/len(fold_accuracy_results)

    tune.report(average_validation_loss=avg_loss, average_validation_accuracy=avg_acc)


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    tuner = tune.Tuner(
        trainable,
        param_space=search_space
    )
    tuner.fit()
    