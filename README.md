# SDSC Setup

If you are using the SDSC Expanse/ACCESS server, they provide a variety of containers for Tensorflow training. In my case, I needed a container with the Ray Tune package, which was unavailable. If you are too lazy to get permission to build an image, a nice work around to is to convert a prexisting container into a Singularity sandbox, run the sandbox container in a shell, add the needed packages, then launch scripts using that sandbox container.

First, login and run the NSF ACCESS expanse shell either via the online user portal: https://portal.expanse.sdsc.edu/ or from your terminal's command line via ssh. See documentation for details: https://www.sdsc.edu/systems/expanse/user_guide.html


Next, build the sandbox container. Note: This may take a while!

Start by creating a bash script with the editor of your choice. In this example, nano is used:
```
nano setup-elp-sandbox.sh
```

Paste the following into the script:
```
#!/usr/bin/bash

module load singularitypro

singularity build --sandbox sandbox_container/ /cm/shared/apps/containers/singularity/tensorflow/tensorflow-latest.sif
```

These commands will load the singularitypro module and build the sandbox container.

Exit and save the script.

Change the script permissions to executable:
```
chmod +x setup-elp-sandbox.sh
```

Verify file permissions:
```
ls -la setup-elp-sandbox.sh
```

Run the script in the background using:
```
nohup ./setup-elp-sandbox.sh &
```
This will allow you to exit the expanse shell session safely without killing the process.

You can view the output of the script with:
```
cat nohup.out
```

#### Other useful commands:

View all the users and resource usage on the current node:
```
htop
```

View the storage space the build is taking up. If this number increases when periodically checking it, the build is still in progress.
Replace build-temp-1430907312/ with the appropriate build-temp-######### folder in your directory.
```
du-ha --max-depth=1 build-temp-1430907312/
```

<!-- Old commands by Lucas

Next, load the singularity module:
```
module load singularitypro  
```

Then, build the container into a Singularity sandbox. The output of this command will be a directory named 'sandbox_container':
```
singularity build --sandbox sandbox_container/ /cm/shared/apps/containers/singularity/tensorflow/tensorflow-latest.sif
``` -->

Once the container has been built into a sandbox directory, load the container in a shell:
```
singularity exec --writable sandbox_container/ bash
```

Now, execute pip to install the necessary packages.
```
pip install -r requirements.txt
```

Once this installation is complete, exit the shell and the contents should be saved:
```
exit
```

Now, we have a container we can use for training.

# General Setup

If you dont need to make a container, install all of the necessary python packages into a virtual environment of your choosing.
```
pip install -r requirements.txt
```

# Data Setup

The 'data_creation' folder contains all of the necessary scripts to convert the Elephant data from raw 24-hour audio clips, to audio clippings of 5 seconds, to tfrecords of audio with appropriate labels, and finally to the tfrecords of spectrograms. These scripts are only helpful if you have access to the ELP data provided by Cornell.

Cut audio clippings.
```
python pos_audio_clips.py
python neg_audio_clips.py
```

Convert clips into tfrecords.
```
python create_tfrecords.py
```

Convert audio tfrecords into spectrograms.
```
python convert_audio_to_spec_tfrecords.py
```

Once you have the directories of the tfrecords for either audio or spectrogram, go into rnn_config.py and cnn_config.py and configure the following parameters to the location of the dataset directory and file names.
```
DATASET_FOLDER = 'audio_tfrecords'

TRAIN_FILE = 'train.tfrecord'
VALIDATE_FILE = 'validate.tfrecord'
TEST_FILE = 'test.tfrecord'
```

# Exploration Script

The exploration script 'cross_validation_experiment.py' will apply a variety of hyperparameter combinations on the models and asses performance using 5 fold cross validation. Ray Tune is used to scale up the experiments, running many at a time.  This script can be run using python in the terminal, or queued as a slurm job.

Python in the terminal.
```
python cross_validation_experiment.py rnn (or cnn)
```

Slurm job. To change the model, go into the script and change the final argument in the python call to either (rnn/cnn)
```
sbatch scripts/run-cross_validation_experiment-gpu-shared.sh
```

You can check the status of your job by running:
```
squeue -u $USER
```

This is an example output of running the previous commmand. The ST column refers to state, in this case the job is pending. When in the running state, the target nodes can be found unnder the Nodelist column.
```             
JOBID       PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
37109179    gpu-debug cross_va lbutler2 PD       0:00      1 (Priority)
```

Check if the script is properly using the GPU, ssh into a node in the Nodelist column.
```
ssh <node>
nvtop
```

# View the results of cross validation

I'm sure there is a better way to do this using ray, however I typically view cross validation results using the following command.
```
python view_cross_validation_results.py
```

The results.csv file created will contain an overview of the configuration and validation loss for each experiment. Choose the best one, then open up the 'train.py' script and adjust the configuration.
```
vim train.py
```

# Train a Model

You can train a model by running the train.py script in one of two ways.

Python in the terminal.
```
python train.py cnn (or rnn)
```

Queueing a slurm job. Again, you'll have to go into the script to change the model type.
```
sbatch scripts/run-train-gpu-shared.sh
```

# Testing
