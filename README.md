# Environment Setup 


# Data Creation

First you need the tfrecords_cherrypicked.zip file. Unzip it.

```
unzip tfrecords_cherrypicked.zip 
```

This file contains labeled audio examples of elephant and non elephant signals. We must create spectrogram examples of this data.

Move into folder 'ELP-CNN-Spectrogram/data_creation' and run the following command to create the spectrogram files from the tfrecords_cherrypicked.

```
python convert_audio_samples_to_spectrogram.py --audio_tfrecords_directory /home/lbutler2/tfrecords_cherrypicked --output_folder ../spectrogram_records
```

Move back one directory into ELP-CNN-Spectrogram. Now, go into config.py and make sure that these two parameters are set properly:

```
DATASET_FOLDER = './spectrogram_records'

TRAIN_FILE = 'spectrogram_train_cherrypick.tfrecord'
VALIDATE_FILE = 'spectrogram_validate_cherrypick.tfrecord'
TEST_FILE = 'spectrogram_test_cherrypick.tfrecord'
```

Once you have verified this, then we can run either the exploration, training or testing script.

# Exploration Script