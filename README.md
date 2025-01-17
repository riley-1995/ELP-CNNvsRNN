This project trainings AlexNet on Spectrograms of ECG signals. It was found that this combination does not produce great results

To get started with this project, first clone this repository and move into the project directory.

```
git clone git@github.com:r-butl/Heart-Abnormality-Classification.git
```

To make installation of the packages very quick and easy, use anaconda to install the packages using the environment.yaml file present in the repository folder.
```
conda env create -n training_env -f environment.yml
conda activate training_env
```
	
To download the dataset, navigate to this link, scroll down to the ‘Files’ section, and download the official ptb-xl dataset zip file: https://physionet.org/content/ptb-xl/1.0.3/. Then unzip and place the dataset folder in the project directory.

```
unzip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
```

Open up config.py and set DATABASE_ROOT_PATH to the name of the unzipped dataset folder. Now it is time to generate the dataset used for training. First, a new database meta file will be generated inside of the dataset folder with the name ‘updated_ptbxl_database.json’. Then, a folder with train.tfrecord, test.tfrecord, and validate.tfrecord files will be created inside of the data_storage folder with the preprocessed data. All that you need to do is run the following command.

```
python data.py
```

Now that the dataset has been generated, type the following command to begin the test run on the dataset.

```
python train.py
```

You can view the training of the model by loading tensorboard with the following command.

```
tensorboard --logdir=logs
```

Finally, evaluate the model using the following command.

```
python test.py
```
