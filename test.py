# Lucas Butler
# Testing script

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import json
import pandas as pd
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
)

from utils import read_tfrecords, get_tfrecord_length

from cnn_config import CNNConfig
from rnn_config import RNNConfig
from cnn import CNN
from rnn import RNN


# Function to recursively convert NumPy types to Python types
def convert_to_python_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # Converts NumPy scalar types (e.g., float32) to native Python types
    elif isinstance(obj, list):
        return [convert_to_python_types(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    return obj


tf.random.set_seed(1)

if __name__ == '__main__':

    cnn = True
    if cnn:
        model = CNN
        cfg = CNNConfig()
    else:
        model = RNN
        cfg = RNNConfig()

    testset = read_tfrecords(os.path.join(cfg.DATASET_FOLDER, cfg.TEST_FILE), buffer_size=64000)
    testset = testset.batch(8)
    
    shape = None
    for t in testset.take(1):
        shape = t[0].shape
    tf.print(shape)

    net = tf.keras.models.load_model(cfg.MODEL_FILE, custom_objects={'Model': model})
    _ = net(tf.random.normal(shape))

    all_predictions = []
    all_labels = []
        
    # Iterate through the test dataset
    start = time.time()
    for samples, labels in testset:
        predictions = net(samples, training=False)  # Apply sigmoid if needed
        all_predictions.append(predictions.numpy().flatten())  # Collect probabilities
        all_labels.append(labels.numpy().flatten())  # Collect true labels
            
    end = time.time()
    tf.print(f"\n\nProcessed {len(all_labels) * len(all_labels[0])} samples in {round(end - start, 2)} seconds.")

    # Convert to numpy arrays
    all_predictions = np.concatenate(all_predictions)  # Concatenate batches
    all_labels = np.concatenate(all_labels)  # Concatenate batches

    # Validate data
    assert np.all((all_predictions >= 0) & (all_predictions <= 1))
    assert np.all((all_labels == 0) | (all_labels == 1))

    # Check if the results directory exists
    if not os.path.exists(cfg.TESTING_IMAGES):
        os.mkdir(cfg.TESTING_IMAGES)

    # Compute ROC curve components
    fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
    auc_score = roc_auc_score(all_labels, all_predictions)

    # Calculate Youden's Index: sensitivity (TPR) - false positive rate (FPR)
    youden_index = tpr + (1 - fpr) - 1
    best_threshold_index = np.argmax(youden_index)  # Index of the best threshold
    best_threshold = thresholds[best_threshold_index]  # Best threshold value

    results = {
            "youden" : best_threshold
            }
    tf.print(f"Best Threshold (Youden's Index): {best_threshold:.4f}")

    # Binarize probabilities using the best threshold
    binary_predictions = (all_predictions >= best_threshold).astype(int)
		
    # Calculate Metrics
    accuracy = accuracy_score(all_labels, binary_predictions)
    precision = precision_score(all_labels, binary_predictions, zero_division=0)
    recall = recall_score(all_labels, binary_predictions, zero_division=0)
    f1 = f1_score(all_labels, binary_predictions, zero_division=0)
    
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_predictions)
    
    # After calculating metrics, you can use the function to ensure compatibility with JSON
    results = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'best_threshold': float(best_threshold),
        'thresholds': thresholds.tolist() + [1],  # Ensure lists are Python native types
        'precisions': precisions.tolist(),  # Convert NumPy array to list
        'recalls': recalls.tolist()  # Convert NumPy array to list
    }

    # Convert any NumPy types in the results dictionary to Python native types
    results = convert_to_python_types(results)

    # Save PR curve data
    pr_data = pd.DataFrame(results)
    pr_data.to_csv(os.path.join(cfg.TESTING_IMAGES, "pr_curve.csv"), index=False)

    # Set common plot styling for all plots
    plt.rcParams.update({
    'font.family': 'serif',           # Set serif font for better readability in academic papers
    'font.size': 14,                  # Set a consistent font size
    'axes.titlesize': 16,             # Title font size
    'axes.labelsize': 14,             # Label font size
    'axes.labelweight': 'bold',       # Bold axis labels
    'axes.grid': True,                # Display gridlines for clarity
    'grid.alpha': 0.3,                # Set gridlines to be light for aesthetics
    'figure.figsize': (8, 6),         # Default figure size for all plots
    'legend.frameon': False,          # Remove legend frame for cleaner look
    })

    # 1. Plot Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='.', label="Precision-Recall Curve", color='b')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.5)  # Use dashed grid lines for subtlety
    plt.tight_layout()  # Adjust layout to fit elements
    plt.savefig(os.path.join(cfg.TESTING_IMAGES, "pr_curve.png"))
    plt.close()

    # 2. Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve', color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Chance')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.TESTING_IMAGES, 'ROC_curve.png'))
    plt.close()

    # 3. Plot Confusion Matrix
    cm = confusion_matrix(all_labels, binary_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

    # Configure colormap and plot settings for Confusion Matrix
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix at Prediction Threshold: {best_threshold:.2f}')
    plt.tight_layout()  # Adjust layout to fit the confusion matrix
    plt.grid(False)  # Disable grid in the confusion matrix plot
    plt.savefig(os.path.join(cfg.TESTING_IMAGES, 'Confusion_matrix.png'))
    plt.close()

    # Print Metrics
    tf.print(f'Accuracy: {accuracy:.2f} | Precision (normal): {precision} | Recall (normal): {recall} | F1 Score (normal): {f1}')
