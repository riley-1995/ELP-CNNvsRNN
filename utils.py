import tensorflow as tf 

def format_time(time):
	m, s = divmod(time, 60)
	h, m = divmod(m, 60)
	d, h = divmod(h, 24)
	return ('{:02d}d {:02d}h {:02d}m {:02d}s').format(int(d), int(h), int(m), int(s))

def get_tfrecord_length(dataset):
	count = 0
	for d in dataset:
		count += 1	
	return count

def read_tfrecords(file_name, buffer_size=1000):

	feature_description = {
		'sample': tf.io.FixedLenFeature([], tf.string),
		'label': tf.io.FixedLenFeature([], tf.int64)
	}

	def _parse_function(example_proto):
		"""Parse a serialized Example."""
		parsed = tf.io.parse_single_example(example_proto, feature_description)
		# Deserialize tensors
		sample = tf.io.parse_tensor(parsed['sample'], out_type=tf.float32)
		label = parsed['label']

		return sample, label

	data = tf.data.TFRecordDataset(file_name, buffer_size=buffer_size)
	dataset = data.map(_parse_function)

	return dataset
