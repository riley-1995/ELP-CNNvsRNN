import itertools
import tensorflow as tf
from utils import read_tfrecords
import os

from trainer import Trainer
from config import GlobalConfiguration

from ray import tune
import ray

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
    "optimizer": ["adam", "sgd"]
    }

def train(config):

    training_dataset = read_tfrecords(os.path.join(cfg.DATASET_FOLDER, cfg.TRAIN_FILE), buffer_size=64000)

    k_folds = cfg.K_FOLDS
    dataset_size = sum(1 for _ in training_dataset)  # Calculate the total number of samples
    fold_size = dataset_size // k_folds  # Calculate the size of each fold
    fold_config_results = []

    for fold_idx in range(k_folds):
        # Create a fresh model for each fold
        net = model(cfg=cfg, training=True)
        trainer = Trainer(cfg=cfg, net=net)

        # Create validation dataset for the current fold
        val_dataset = training_dataset.skip(fold_idx * fold_size).take(fold_size)

        # Create training dataset by skipping the validation fold and concatenating the rest
        train_dataset = training_dataset.take(fold_idx * fold_size).concatenate(
            training_dataset.skip((fold_idx + 1) * fold_size)
        )

        # Run training for this fold
        fold_val_loss = trainer.train(
            trainset=train_dataset.batch(parameters[0]), 
            valset=val_dataset.batch(parameters[0]), 
            cross_validate=True, 
            max_epochs=cfg.MAX_CV_EPOCHS
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