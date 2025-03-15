import itertools
import tensorflow as tf
from utils import read_tfrecords
import os
from cnn import CNN
from rnn import RNN
from cnn_config import CNNConfig
from rnn_config import RNNConfig
import tensorflow as tf
from ray import tune
import ray
import argparse
from ray.tune.search.optuna import OptunaSearch

print(tf.config.list_physical_devices('GPU'))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def k_fold_split(dataset, num_folds, fold_idx):

    dataset_size = sum(1 for _ in dataset)  # Calculate the total number of samples
    fold_size = dataset_size // num_folds  # Calculate the size of each fold
    
    # Create validation dataset for the current fold
    val_dataset = dataset.skip(fold_idx * fold_size).take(fold_size)

    # Create training dataset by skipping the validation fold and concatenating the rest
    train_dataset = dataset.take(fold_idx * fold_size).concatenate(
        dataset.skip((fold_idx + 1) * fold_size)
    )

    return train_dataset, val_dataset

@tf.function
def train_step(net, optimizer, loss_fn, samples, labels):
    with tf.device('/GPU:0'):  # Explicitly run on GPU
        with tf.GradientTape() as tape:
            predictions = net(samples, training=True)
            loss = loss_fn(labels, predictions)
    
        gradients = tape.gradient(loss, net.trainable_weights)
        optimizer.apply_gradients(zip(gradients, net.trainable_weights))

    return loss

def trainable(config):
    with tf.device('/GPU:0'):  # Ensure computations happen on GPU
        cfg = config['config']
        training_dataset = read_tfrecords(os.path.join(cfg.DATASET_FOLDER, cfg.TRAIN_FILE), buffer_size=64000)

        for sample, label in training_dataset.take(1):
            shape = [None] + sample.shape
        tf.print(shape)
        dataset_size = sum(1 for _ in training_dataset)  # Calculate the total number of samples

        # Define learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=config['learning_rate'],  
            decay_steps=config['learning_rate_decay_steps'],
            decay_rate=config['learning_rate_decay'],
            staircase=True 
        )
    
        # Loss Function
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # store results
        fold_loss_results = []
        fold_accuracy_results = []

        # Iterate through folds
        for fold_idx in range(cfg.K_FOLDS):
            tf.print("New fold")

            # Create a fresh model for each fold
            net = config['model'](model_config=config, training=True)
            net.build(shape)

            # Optimization
            if config["optimizer"] == "adam":
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
            elif config["optimizer"] == "sgd":
                optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=config["momentum"])

            optimizer.build(net.trainable_weights)
            train_dataset, val_dataset = k_fold_split(training_dataset, cfg.K_FOLDS, fold_idx)

            # Iterate through epochs
            for _ in range(config['epochs']):

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

            tf.print(f"Validation loss: {validation_loss:.2f} Validation Accuracy: {validation_accuracy:.2f}")

            fold_loss_results.append(validation_loss)
            fold_accuracy_results.append(validation_accuracy)
            tune.report({'val_loss': validation_loss})


    avg_loss = sum(fold_loss_results)/len(fold_loss_results)
    avg_acc = sum(fold_accuracy_results)/len(fold_accuracy_results)

    tf.print(f"Config: {config} Avg. Val. Loss: {avg_loss:.2f} Avg. Val. Acc: {avg_acc:.2f}")
    tune.report({'config': config, 'avg_loss': avg_loss, 'avg_acc': avg_acc})


if __name__ == "__main__":

    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Choose between CNN and RNN.")
    parser.add_argument("--model", choices=["cnn", "rnn"], required=True, help="Specify the model type: 'cnn' or 'rnn'")
    args = parser.parse_args()

    # Model selection
    if args.model == "cnn":
        model = CNN
        cfg = CNNConfig
        name = "cnn"
    else:
        model = RNN
        cfg = RNNConfig
        name = "rnn"

    search_space = {  
        "learning_rate": tune.choice([0.01, 0.001, 0.0001]),
        "learning_rate_decay_steps": tune.choice([200, 500]),
        "learning_rate_decay": tune.choice([.92, 0.97, 1.0]),
        "momentum": tune.choice([0.5, 0.7, 0.9]),
        "batch_size": tune.choice([ 8, 16, 32]),
        "epochs": tune.choice([15]),
        "activation_function": tune.choice(["ReLU", "LeakyReLU"]),
        "dropout_rate": tune.choice([0.2, 0.5, 0.7]),
        "optimizer": tune.choice(["adam", "sgd"]),
        "model": model,
        "config": cfg
    }
    
    ray.init(ignore_reinit_error=True)

    resources={"cpu": 1, "gpu": 1}

    search_alg = OptunaSearch()

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=15,
            max_concurrent_trials=2,
            #metric="val_loss",
            #mode="min",
        #    search_alg=search_alg
        ),
        run_config=tune.RunConfig(
            storage_path=os.path.join(os.getcwd(), f'{name}_cross_validation_results')
        )
    )
    tuner.fit()
