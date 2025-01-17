# Lucas Butler
# Testing script

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import time
import numpy as np
import tensorflow as tf
from data import PTBXLDataset
from config import Configuration
from alexnet import AlexNet
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

tf.random.set_seed(1)

def get_tfrecord_length(dataset):
	count = 0
	for d in dataset:
		count += 1	
	return count

# Class for Tester
class Tester(object):

	def __init__(self, cfg, net, testset):
		self.cfg = cfg
		self.net = net
		self.testset = testset

	def test(self):
		'''
		Test the model.
		'''
		all_predictions = []
		all_labels = []

		# Iterate through the test dataset
		start = time.time()
		for x_batch, y_batch in self.testset:
			# Ensure output probabilities are in [0, 1]
			probabilities = tf.sigmoid(self.net(x_batch, training=False))  # Apply sigmoid if needed
			all_predictions.append(probabilities.numpy().flatten())  # Collect probabilities
			all_labels.append(y_batch.numpy().flatten())  # Collect true labels

		end = time.time()
		tf.print(f"\n\nProcessed {len(all_labels) * len(all_labels[0])} samples in {round(end - start, 2)} seconds.")

		# Convert to numpy arrays
		all_predictions = np.concatenate(all_predictions)  # Concatenate batches
		all_labels = np.concatenate(all_labels)  # Concatenate batches

		# Validate data
		assert np.all((all_predictions >= 0) & (all_predictions <= 1)), "Predictions out of range [0, 1]"
		assert np.all((all_labels == 0) | (all_labels == 1)), "Labels must be binary (0 or 1)"

		return all_predictions, all_labels

	def find_best_threshold_and_produce_metrics(self, actual, predicted_probs):
		"""
		Sweeps through thresholds to find the best threshold value based on Youden's Index,
		generates the confusion matrix, and calculates accuracy, precision, recall, and F1 score.

		Inputs:
			actual: np.array - Ground truth binary labels (0 or 1).
			predicted_probs: np.array - Predicted probabilities for the positive class.
		
		Outputs:
			best_threshold: float - The threshold that maximizes Youden's Index.
			accuracy: float - Accuracy at the best threshold.
			precision: float - Precision at the best threshold.
			recall: float - Recall at the best threshold.
			f1: float - F1 score at the best threshold.
		"""
		# Compute ROC curve components
		fpr, tpr, thresholds = roc_curve(actual, predicted_probs)
		auc_score = roc_auc_score(actual, predicted_probs)

		# Plot ROC Curve
		plt.figure(figsize=(7, 7))
		plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', color='blue')
		plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Chance')
		plt.title('Receiver Operating Characteristic (ROC) Curve')
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.legend()
		plt.grid()
		plt.savefig('ROC_curve.png')
		
		# Calculate Youden's Index: sensitivity (TPR) - false positive rate (FPR)
		youden_index = tpr + (1 - fpr) - 1
		best_threshold_index = np.argmax(youden_index)  # Index of the best threshold
		best_threshold = thresholds[best_threshold_index]  # Best threshold value
		
		tf.print(f"Best Threshold (Youden's Index): {best_threshold:.4f}")

		# Binarize probabilities using the best threshold
		binary_predictions = (predicted_probs >= best_threshold).astype(int)
		
		# Compute Confusion Matrix
		cm = confusion_matrix(actual, binary_predictions)
		disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
		disp.plot(cmap=plt.cm.Blues)
		plt.title(f'Confusion Matrix at Best Threshold (Abnormal) {best_threshold:.2f}')
		plt.grid(False)
		plt.savefig('Confusion_matrix.png')

		# Calculate Metrics
		accuracy = accuracy_score(actual, binary_predictions)
		# Accuracy of of positive predictions
		precision = np.round(np.sum((binary_predictions == 1) & (actual == 1)) / np.sum(binary_predictions == 1), 2) if np.sum(binary_predictions == 1) > 0 else 0.0
		# ability to idenitify actual positives
		recall = np.round(np.sum((binary_predictions == 1) & (actual == 1)) / np.sum(actual == 1), 2) if np.sum(actual == 1) > 0 else 0.0
		f1 = np.round(2 * (precision * recall) / (precision + recall), 2) if (precision + recall) > 0 else 0.0

		# Print Metrics
		tf.print(f'Accuracy: {accuracy:.2f} | Precision (normal): {precision} | Recall (normal): {recall} | F1 Score (normal): {f1}')
		
		return best_threshold, accuracy, precision, recall, f1
	
if __name__ == '__main__':
	i = 0

	# Path for test results
	if not os.path.exists("Tests"):
		os.makedirs('Tests')

	while os.path.exists("Tests/Test%s.txt" % i):
		i += 1

	LOG_PATH = "Tests/Test%s.txt" % i
	def print(msg):
		with open(LOG_PATH,'a') as f:
			f.write(f'{time.ctime()}: {msg}\n')

	cfg = Configuration()

	dataset = PTBXLDataset(cfg=cfg)
	testset = dataset.read_tfrecords('test.tfrecord', buffer_size=64000)
	tf.print(f"\n\nNumber of samples: {get_tfrecord_length(testset)}\n\n")
	testset = testset.batch(128)

	shape = None
	for t in testset.take(1):
		shape = t[0].shape

	# Get the Alexnet form models
	net = AlexNet(cfg=cfg, training=False)
	net.build(input_shape=shape)

	net.load_weights(cfg.MODEL_FILE)

	# Create a tester object
	tester = Tester(cfg, net, testset)

	# Call test function on tester object
	predictions, labels = tester.test()

			# Generate ROC Curve and AUC
	tester.find_best_threshold_and_produce_metrics(labels, predictions)

