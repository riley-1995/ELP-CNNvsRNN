import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config

import ast
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.signal import butter, sosfilt, ShortTimeFFT, get_window
from PIL import Image
import wfdb
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Have to place this outside of the class inorder to use multiprocessing on it
def process_file(args):
    index, root_path, f, leads = args
    data = wfdb.rdsamp(os.path.join(root_path, os.path.splitext(f)[0]), channel_names=leads, return_res=32)[0].T
    return [index, data]

class Filter():
    def __init__(self):

        # Define the filter
        lowcut = 1.0          # Lower cutoff frequency (Hz)
        highcut = 45.0        # Upper cutoff frequency (Hz)
        fs = 100.0            # Sampling frequency (Hz)
        order = 2             # Filter order (higher = steeper rolloff)
        self.sos = self.design_iir_bandpass(lowcut, highcut, fs, order)

        # Define a spectrogram
        nperseg = fs * 1    # Length of each segment in seconds
        hop = 2             # Similar to stride
        w = get_window(('gaussian', 15), nperseg)
        self.sft = ShortTimeFFT(w, hop, fs=fs, mfft=150)

    def design_iir_bandpass(self, lowcut, highcut, fs, order=4):
        '''
        Designs the bandpass filer
        '''
        nyquist = 0.5 * fs  # Nyquist frequency
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = butter(order, [low, high], btype='band', output='sos')  # Second-order sections
        return sos
    
    def apply_filter_spectrogram(self, data):
        '''
        Applies the filter and the spectrogram to the 1D data
        '''
        filtered = sosfilt(self.sos, data, axis=-1)
        spectrogram = self.sft.spectrogram(filtered, axis=-1)
        tensor = np.transpose(spectrogram, (1, 2, 0)).astype(np.float32)
        return tensor
    
# Class for creatinng dataset
class PTBXLDataset(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.leads = ['I', 'II', 'V2']
        self.classes = ['ABNORM']
        self.sampling_rate = 100 # or 500

        if not os.path.exists(os.path.join(self.cfg.DATABASE_ROOT_PATH, self.cfg.DATABASE_FILE_NAME)):
            self.create_updated_database_metafile(sampling_rate=self.sampling_rate)

        self.dataset = pd.read_json(os.path.join(self.cfg.DATABASE_ROOT_PATH, self.cfg.DATABASE_FILE_NAME))

        self.dataset = self.dataset[['filename_lr', 'filename_hr'] + self.classes]

        self.filter = Filter()
        self.filter_function = self.filter.apply_filter_spectrogram

    def create_updated_database_metafile(self, sampling_rate=100):
        print('Creating updated database metafile...') 
        files = pd.read_csv(os.path.join(self.cfg.DATABASE_ROOT_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
        files.scp_codes = files.scp_codes.apply(lambda x: ast.literal_eval(x))
        files[['age', 'sex', 'height', 'weight', 'scp_codes']].head(20)

        def less_than_100_check(row):
            codes = row['scp_codes']
            if 'NORM' in codes and codes['NORM'] < 100:
                return 1
            else:
                return 0
            
        removal_mask = files.apply(less_than_100_check, axis=1).values
        filtered_files = files[removal_mask == 0]

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(os.path.join(self.cfg.DATABASE_ROOT_PATH, 'scp_statements.csv'), index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in agg_df.index:
                    tmp.append(agg_df.loc[key].diagnostic_class)
            return list(set(tmp))

        # Apply diagnostic superclass
        filtered_files['diagnostic_superclass'] = filtered_files.scp_codes.apply(aggregate_diagnostic)
        diagnostic_superclass = filtered_files['diagnostic_superclass'].value_counts()
        classes = diagnostic_superclass.index
        counts = diagnostic_superclass.values
        frame = pd.DataFrame([co for co in zip(classes, counts)], columns=['class', 'counts'])

        def convert_and_check(row):
            classes = row['class']
            count = row['counts']
            if 'NORM' in classes and len(classes) > 1:
                return 1    # Drop
            elif len(classes) == 0:
                return 1
            elif count < 20:
                return 1
            else:
                return 0

        drop_indices = frame.apply(convert_and_check, axis=1).values
        filtered_classes = frame[drop_indices == 0]

        def all_values_match(x, y):
            '''Check if all values in x are in y'''
            if len(x) != len(y):
                return 0
            else:
                for value in x:
                    if value not in y:
                        return 0
            return 1

        def in_class_list(row):
            '''Check if the diagnostic superclass is in the filtered classes'''
            try:
                for allowable_classes in filtered_classes['class'].values:
                    if all_values_match(allowable_classes, row['diagnostic_superclass']):
                        return 1
                return 0
            
            except:
                print(row['diagnostic_superclass'])
                print(filtered_classes['class'].values)

        keep_indices = filtered_files.apply(in_class_list, axis=1)
        filtered_files = filtered_files[keep_indices == 1]

        def is_normal(row):
            '''Check if the diagnostic superclass is normal'''
            if 'NORM' in row['diagnostic_superclass']:
                return 0
            else:
                return 1
            
        norm_col = filtered_files.apply(is_normal, axis=1)

        filtered_files['ABNORM'] = norm_col

        filtered_files.to_json(os.path.join(self.cfg.DATABASE_ROOT_PATH, self.cfg.DATABASE_FILE_NAME))

    def create_splits(self, multilabel=False):
        """
        Creates train, validation, and test splits with stratification.
        """
        X = self.dataset[['filename_lr', 'filename_hr']]
        y = self.dataset[self.classes]

        # Choose the appropriate splitter
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=self.cfg.TEST_PERCENTAGE, random_state=0
        )

        # First, split into train+validation and test
        train_val_indices, test_indices = next(splitter.split(X, y))

        # Isolate train+validation data
        X_train_val, y_train_val = X.iloc[train_val_indices], y.iloc[train_val_indices]

        # Now split train+validation into train and validation
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=self.cfg.VALIDATE_PERCENTAGE / (1 - self.cfg.TEST_PERCENTAGE), random_state=0
        )

        train_indices, validation_indices = next(splitter.split(X_train_val, y_train_val))

        # Map back to original indices
        train_indices = train_val_indices[train_indices]
        validation_indices = train_val_indices[validation_indices]

        # Assertions to ensure no overlap
        assert not set(train_indices).intersection(validation_indices), "Train and validation overlap!"
        assert not set(train_indices).intersection(test_indices), "Train and test overlap!"
        assert not set(validation_indices).intersection(test_indices), "Validation and test overlap!"

        # Assign splits
        self.train_df = pd.concat([X.iloc[train_indices], y.iloc[train_indices]], axis=1)
        self.validate_df = pd.concat([X.iloc[validation_indices], y.iloc[validation_indices]], axis=1)
        self.test_df = pd.concat([X.iloc[test_indices], y.iloc[test_indices]], axis=1)

        del X, y
        
    def load_batch(self, df):
        '''
        Loads signals from the signal database into tensors
        '''
        df_files = df.filename_lr if self.sampling_rate == 100 else df.filename_hr

        args = [
            (i, self.cfg.DATABASE_ROOT_PATH, f, self.leads)
            for i, f in enumerate(df_files)
        ]

        with ProcessPoolExecutor() as executor:
            data = list(tqdm(executor.map(process_file, args), total=len(df_files), desc="Processing files"))

        # Ensure that the ordering of the data is correct.
        data = sorted(data, key=lambda x: x[0])
        data = [d[1] for d in data]

        labels = df[self.classes].to_numpy().astype(np.int32)
        return data, labels
    
    def write_tfrecords(self, dataset: tf.data.Dataset, file_prefix):
        '''
        Input:
            dataset:        tf.Dataset of data to write.
            file_prefix:    Name of the file to write the data.
        '''
        filename = f'{file_prefix}.tfrecord'
        with tf.io.TFRecordWriter(filename) as writer:
            for sample, label in dataset:

                sample = tf.convert_to_tensor(sample)
                label = tf.convert_to_tensor(label)

                feature = {
                    'sample': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(sample).numpy()])),  
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(label).numpy()]))
                }
                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example_proto.SerializeToString())

    def calculate_global_mean_std(self):
        '''
        Loads the full dataset to calculate the global mean and std
        '''

        data, labels = self.load_batch(self.dataset)
        dataset = tf.data.Dataset.from_tensor_slices((data , labels))

        # Calculate mean and std across the entire dataset
        def get_mean_and_std(dataset):
            means = []
            stds = []
            for sample, _ in dataset:
                means.append(tf.math.reduce_mean(sample, axis=(0, 1)))  # Mean across height and width
                stds.append(tf.math.reduce_std(sample, axis=(0, 1)))    # Std across height and width
            return tf.math.reduce_mean(means, axis=0), tf.math.reduce_mean(stds, axis=0)

        global_mean, global_std = get_mean_and_std(dataset)

        del dataset
        return global_mean, global_std
    
    def normalize(self, dataset: np.array, mean, std):
        '''
            Applies the mean and standard deviation
        '''
        def map_function(data, mean, std):
            '''Normmalization function'''
            data = (data - mean) / std
            return data

        dataset = np.array([map_function(d, mean, std) for d in dataset])

        return dataset
    
    def apply_filter(self, dataset: np.array):
        '''
            Applies the filter and spectrogram function
        '''
        def map_function(data):
            data = self.filter_function(data)
            return data
        
        dataset = np.array([map_function(d) for d in dataset])

        return dataset

    def create_dataset_from_df(self, df, mean, std, normalize=False):
        '''
        Loads a dataset from the file names in a dataframe, applied the normalization to the signal
            using the mean and std. Applies the filter function, then stores in a TFRecord and writes
            to disk.
        '''

        data, labels = self.load_batch(df)

        if normalize:
            data = self.normalize(data, mean, std)

        data = self.apply_filter(data)
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        return dataset
        
    def create_dataset(self, normalize=False):
        '''
        Calculates the mean and std of the dataset, then creates each dataset after normalizing, applying
            the noise filter, and producing the spectrogram.
        '''

        storage_folder = os.path.join(self.cfg.DATASET_STORAGE, self.cfg.DATASET_FOLDER)

        if not os.path.exists(storage_folder):
            os.makedirs(storage_folder)

        mean, std = self.calculate_global_mean_std()

        self.create_splits()

        test_dataset = self.create_dataset_from_df(self.test_df, mean, std, normalize=normalize)
        self.write_tfrecords(test_dataset, os.path.join(storage_folder, 'test'))
        del test_dataset

        validate_dataset = self.create_dataset_from_df(self.validate_df, mean, std, normalize=normalize)
        self.write_tfrecords(validate_dataset, os.path.join(storage_folder, 'validate'))
        del validate_dataset

        train_dataset = self.create_dataset_from_df(self.train_df, mean, std, normalize=normalize)
        self.write_tfrecords(train_dataset, os.path.join(storage_folder,'train')) 
        del train_dataset


    def read_tfrecords(self, file_name, buffer_size=1000):
        '''
        Input:
            file_name:  File name to read records from.
        Output:
            dataset:    TFRecordDataset.
        '''
        
        features = {
            'sample': tf.io.FixedLenFeature([], tf.string),  
            'label': tf.io.FixedLenFeature([], tf.string)
        }
        
        def _parse_function(example_proto):
            """Parse a serialized Example."""
            parsed = tf.io.parse_single_example(example_proto, features)
            # Deserialize tensors
            sample = tf.io.parse_tensor(parsed['sample'], out_type=tf.float32)
            label = tf.io.parse_tensor(parsed['label'], out_type=tf.int32)

            return sample, label
        
        file_name = os.path.join(os.path.join(os.path.join(os.getcwd(), self.cfg.DATASET_STORAGE), self.cfg.DATASET_FOLDER), file_name)
        data = tf.data.TFRecordDataset(file_name, buffer_size=buffer_size)
        dataset = data.map(_parse_function)

        return dataset
    
if __name__ == "__main__":

    cfg = config.Configuration()
    dataset = PTBXLDataset(cfg=cfg)
    dataset.create_dataset(normalize=False)
