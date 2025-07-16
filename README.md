# SDSC Setup (Automated Version)

If you are using the SDSC Expanse/ACCESS system and need to build a Singularity container (e.g., with additional packages like Ray Tune), you can now automate the full process via a SLURM job to convert a prexisting container into a singularity sandbox and add the needed packages. Doing this as a SLURM job as outlined below avoids login node timeouts, handles disk quotas correctly, and ensures everything is saved in project storage. Once the container is built and configured, scripts can be launched using that container.

## Prerequisites

- You must be a member of a project with access to Expanse project storage (e.g., `/expanse/lustre/projects/cso100/`).
- Your Python dependencies should be listed in a `requirements.txt` file which should be available in the project repo.
- You must have access to the Cornell ELP data

## Step-by-Step Instructions

First, login and run the NSF ACCESS expanse shell either via the online user portal: https://portal.expanse.sdsc.edu/ or from your terminal's command line via ssh. See documentation for details: https://www.sdsc.edu/systems/expanse/user_guide.html

Next, make a directory for the container in project storage and clone the github repo (or your fork of the repo) to project storage.
```bash
mkdir /expanse/lustre/projects/cso100/$USER/elp_container
cd /expanse/lustre/projects/cso100/$USER/elp_container
git clone <repo web url clone link>
```

### 1. Create the SLURM Script - Scratch Drive Option (recommended method)

📝 **Scratch Drive Option:** If you prefer to use scratch space for faster builds and larger temporary capacity (up to 10 TB), the script now supports using the scratch drive `/expanse/scratch/$USER` for the build process. It will copy the final container to project storage after building.

Ensure you are in the directory `/expanse/lustre/projects/cso100/$USER/elp_container/`

Create a job script called `build_and_setup_container.slurm`:
```bash
nano build_and_setup_container.slurm
```

Paste in the following:
```bash
#!/bin/bash
#SBATCH --job-name=elp_sandbox
#SBATCH --output=elp_sandbox.out
#SBATCH --error=elp_sandbox.err
#SBATCH --partition=gpu
#SBATCH --account=cso100
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --constraint=lustre

# Load Singularity module
module load singularitypro

# Set working paths
SCRATCH_DIR=/expanse/lustre/scratch/$USER/temp_project/elp_sandbox_tmp_$SLURM_JOB_ID
PROJECT_DIR=/expanse/lustre/projects/cso100/$USER/elp_container

echo "Hostname: $(hostname)"
echo "Running on: $(pwd)"
echo "Scratch build directory: $SCRATCH_DIR"
echo "Project directory: $PROJECT_DIR"

# Create scratch build directory
mkdir -p $SCRATCH_DIR/tmp || { echo "❌ Failed to create scratch tmp dir"; exit 1; }
export SINGULARITY_TMPDIR=$SCRATCH_DIR/tmp

# Cleanup old builds if needed
rm -rf $SCRATCH_DIR/sandbox/
rm -rf $SCRATCH_DIR/tmp/*
rm -rf $SCRATCH_DIR/build-temp-*/

# Build the container in scratch
singularity build --sandbox $SCRATCH_DIR/sandbox/ /cm/shared/apps/containers/singularity/tensorflow/tensorflow-latest.sif || { echo "❌ Singularity build failed"; exit 1; }

# Copy requirements file - Update your repo name accordingly
cp /expanse/lustre/projects/cso100/$USER/ELP-CNNvsRNN/requirements.txt $SCRATCH_DIR/requirements.txt || { echo "❌ Could not copy requirements.txt"; exit 1; }

# Install dependencies inside container
singularity exec --writable $SCRATCH_DIR/sandbox/ bash -c "\
  pip install --upgrade pip && \
  pip install -r /requirements.txt && \
  rm /requirements.txt" || { echo "❌ pip install failed"; exit 1; }

# Copy completed sandbox to project storage
rsync -av $SCRATCH_DIR/sandbox/ $PROJECT_DIR/sandbox/ || { echo "❌ rsync to project dir failed"; exit 1; }
```

### 2. Submit the Job to Build and Setup the Container

```bash
sbatch build_and_setup_container.slurm
```

### 📊 Monitor Your Job

To check the queue status of your job:
```bash
squeue -u $USER
```

To see details about the most recent job submission:
```bash
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS,AllocCPUs
```

To follow logs in real time:
```bash
tail -f elp_sandbox.out
```

Or to view errors:
```bash
tail -f elp_sandbox.err
```

To stop following:
```bash
Ctrl+C
```

If you prefer to just view a snapshot of these files, use
```bash
cat elp_sandbox.out
cat elp_sandbox.err
```

To cancel a running job:
```bash
scancel <job_id>
```

### 🔍 Live Monitoring with htop

To check how much space your scratch build is using:
```bash
du -sh /expanse/lustre/scratch/$USER/temp_project/elp_sandbox_tmp_*
```
This will give you a summary of total disk space used by each job-specific build folder.

For a deeper look at what's inside:
```bash
du -h --max-depth=1 /expanse/lustre/scratch/$USER/temp_project/elp_sandbox_tmp_<jobid>
```

You can also watch space usage live:
```bash
watch -n 2 'du -sh /expanse/lustre/scratch/$USER/temp_project/elp_sandbox_tmp_*'
```

If your job is running and you want to monitor live system resource usage:

1. Get the node where your job is running:
```bash
squeue -u $USER
```

2. SSH into the node:
```bash
ssh <nodelist>
```
(e.g., `ssh exp-9-57`)

3. Launch htop:
```bash
htop
```
Use `F6` to sort, and `F10` to exit htop view.

To exit the node and return to your main SDSC session, type:
```bash
exit
```

### 3. Use the Container

Once built, you can use the container with:

```bash
singularity exec --writable /expanse/lustre/projects/cso100/$USER/elp_container/sandbox/ bash
```

Now you're inside the container and ready to run training or experiments.

---

## Legacy Manual Build (Not Recommended)

If you prefer to use the login node to build the container (not recommended):
```bash
module load singularitypro
singularity build --sandbox sandbox_container/ /cm/shared/apps/containers/singularity/tensorflow/tensorflow-latest.sif
singularity exec --writable sandbox_container/ bash
pip install -r requirements.txt
exit
```

⚠️ **This method is not recommended** as it may get killed due to I/O, time, or memory limits.

---

### 🧼 Optional: Clean Up Old Scratch Builds With Script: `cleanup_scratch_builds.sh

To remove old container build folders in your scratch space (e.g., from previous failed or old SLURM jobs), you can run a script that automatically skips active jobs and logs deleted folders.

To do so, ensure you are in the directory `/expanse/lustre/projects/cso100/$USER/elp_container/`

Create a job script called `cleanup_scratch_builds.sh`:
```bash
nano cleanup_scratch_builds.sh
```

Paste in the following and save the script:
```bash
#!/bin/bash

SCRATCH_BASE=/expanse/lustre/scratch/$USER/temp_project
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="$SCRIPT_DIR/scratch_cleanup_$(date +%Y%m%d_%H%M%S).log"

echo "🧹 Cleaning up scratch containers in $SCRATCH_BASE" | tee -a $LOGFILE
mkdir -p empty_temp_dir

for dir in "$SCRATCH_BASE"/elp_sandbox_tmp_*; do
  if [ -d "$dir" ]; then
    JOB_ID=$(basename "$dir" | awk -F_ '{print $NF}')
    if squeue -j "$JOB_ID" -u $USER | grep -q "$JOB_ID"; then
      echo "🔒 Skipping active job dir: $dir" | tee -a $LOGFILE
      continue
    fi
    echo "→ Processing $dir" | tee -a $LOGFILE
    rsync -a --delete --omit-dir-times empty_temp_dir/ "$dir/" 2>> $LOGFILE
    chmod -R u+w "$dir" 2>> $LOGFILE
    rm -rf "$dir"
    echo "✅ Deleted $dir" | tee -a $LOGFILE
  fi
done

rm -rf empty_temp_dir

# Keep only most recent 3 cleanup logs
cd "$SCRIPT_DIR"
ls -tp scratch_cleanup_*.log | grep -v '/$' | tail -n +4 | xargs -r rm --

echo "✅ Scratch cleanup complete at $(date)" | tee -a $LOGFILE
```

Make it executable:
```bash
chmod +x cleanup_scratch_builds.sh
```

Run it in the background:
```bash
nohup ./cleanup_scratch_builds.sh > cleanup_scratch.log 2>&1 &
```

Check on progress:
```bash
tail -f cleanup_scratch.log
```

Stop monitoring:
```bash
Ctrl+C
```

View the logs:
```bash
less cleanup_scratch_*.log
```

---

# General Setup (Local Machine)

This project requires Python 3.11. These instructions will guide you through setting up a local development environment on macOS.

**Install Prerequisites (Homebrew & pyenv)**

If you don't already have them, install Homebrew (the macOS package manager) and then use it to install `pyenv` for managing Python versions.
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install pyenv
brew install pyenv
```

**Install Python 3.11**
Use `pyenv` to install the specific Python version for this project. This may take a few minutes.
```bash
pyenv install 3.11.9
```

**Configure the Project Environment**
Navigate to the root folder which will hold your ELP-CNNvsRNN cloned repository and your venv
```bash
cd /path/to/your/ElephantListeningProject
```

Clone this repo from github
```bash
git clone <insert repo's https link>
```

Create a venv with python 3.11, activate it, cd into repo, and install requirements.txt.
```bash
~/.pyenv/versions/3.11.9/bin/python -m venv elp-venv
source elp-venv/bin/activate
cd ELP-RNNvsCNN
pip install -r requirements.txt
```

Create your personal .env configuration file from the example
```bash
cp .env.example .env
```

Edit the .env file with your local data path:
Open the .env file in a text editor and change the path to point
to where you stored the ELP Cornell Data folder. For example:
CORNELL_DATA_ROOT="/Users/rileydenn/ELP_Cornell_Data"
You can either use nano or the editor of your choosing.
```bash
nano .env
```

---

## Data Preprocessing (Suggested to do on your local machine)
The 'data_creation' folder contains all of the necessary scripts to convert the Elephant data from raw 24-hour audio clips, to audio clippings of 5 seconds, to tfrecords of audio with appropriate labels, and finally to the tfrecords of spectrograms. These scripts are only helpful if you have access to the ELP data provided by Cornell.

Ensure you are in the correct location and your local venv is activated:
```bash
cd /path/to/your/ElephantListeningProject
source elp-venv/bin/activate
cd ELP-CNNvsRNN
```

Cut audio clippings:
```bash
python3 data_creation/pos_audio_clips.py --mode train
python3 data_creation/pos_audio_clips.py --mode test
python3 data_creation/neg_audio_clips.py
```

Convert clips into tfrecords, then convert audio tfrecords into spectrograms:
```bash
python3 data_creation/create_tfrecords.py
python3 data_creation/convert_audio_to_spec_tfrecords.py
```

Once you have the directories of the tfrecords for either audio or spectrogram, go into rnn_config.py and cnn_config.py and configure the following parameters to the location of the dataset directory and file names:
```python
DATASET_FOLDER = 'audio_tfrecords'
TRAIN_FILE = 'train.tfrecord'
VALIDATE_FILE = 'validate.tfrecord'
TEST_FILE = 'test.tfrecord'
```

---

## After Preprocessing Data Locally, Upload to Expanse Project Storage

To upload your local preprocessed tfrecords data to SDSC Expanse project storage, use the `rsync` command from your local terminal:
```bash
rsync -avh --progress \
"/path/to/local/project/ELP-CNNvsRNN/data" \
rdenn@login.expanse.sdsc.edu:/expanse/lustre/projects/cso100/your_username/elp_container/ELP-CNNvsRNN/data

rsync -avh --progress \
"/path/to/local/project/repo/ELP-CNNvsRNN/data/tfrecords_audio" \
"/path/to/local/project/repo/ELP-CNNvsRNN/data/tfrecords_spectrogram" \
rdenn@login.expanse.sdsc.edu:/expanse/lustre/projects/cso100/your_username/elp_container/ELP-CNNvsRNN/data/
```

Replace `"/path/to/local/project/repo/"` with the full path to your local git project repo and replace `your_username` with your ACCESS Expanse username.

Note: This may take a while!

If interrupted, you can re-run the same command to resume.

You can check storage usage (snapshot) on Expanse with:
```bash
du -sh /expanse/lustre/projects/cso100/$USER/elp_container/ELP-CNNvsRNN/data/
```

Or to monitor it continuously as it grows:
```bash
watch -n 5 'du -sh /expanse/lustre/projects/cso100/$USER/elp_container/ELP-CNNvsRNN/data/'
```

---

# Running Experiments

### Local Terminal:

```bash
python cross_validation_experiment.py cnn  # or rnn
```

### SLURM Batch Job:

For running cross-validation tuning experiment to find best hyperparameters:
```bash
sbatch scripts/run-cross_validation_experiment-gpu-shared.sh
```

For training debugging, making sure paths are correct, etc.:
```bash
sbatch scripts/run-train-gpu-shared.sh
```

For full training:
```bash
sbatch scripts/run-train-gpu-shared.sh
```

#### Monitor your job:

The job id, job name, status, node, and other info about the job can be found via:
```
squeue -u $USER -l
``` 

SSH into node to check GPU:
```bash
ssh <node>
nvtop
```

#### Check output logs:
```bash
ls -lh train.o*
```

Note: Replace train with whatever the job name is, which can be found in the script or with the squeue command above. For example, for a script, the job-name is assigned here:
```
#SBATCH --job-name=train-debug
```
Therefore, use:
```bash
ls -lh train-debug.o*
```

To see the logs:
```bash
cat <name of file>
```
Ex:
```bash
cat train.o41166992.exp-14-58
```

---

## View Results

```bash
python view_cross_validation_results.py
vim train.py  # edit best config
```

---

## Train Final Model

```bash
python train.py cnn  # or rnn
```

Or submit with:

```bash
sbatch scripts/run-train-gpu-shared.sh
```

# Other Tools/Resources

## RavenPro (or RavenLite - free) 
- Can be used to view and annotate audio waveforms and spectrograms

https://www.ravensoundsoftware.com/software/
https://www.ravensoundsoftware.com/knowledge-base/

## San Diego Supercomputer Center

#### SDSC User Guide
https://www.sdsc.edu/systems/expanse/user_guide.html#narrow-wysiwyg-7

#### SDSC Basic Skills
- Includes basic Linux skills, interactive computing, running Jupyter notebooks on Expanse, and info on using git/Github

https://github.com/sdsc-hpc-training-org/basic_skills

#### Intermediate Linux Workshop Slides
- Useful for navigating ACCESS Expanse server. Slides are from July 2025. Webinar video not yet available (as of July 2025) however they will soon be uploaded to SDSC On-Demand Learning archive. Previous Intermediate Linux webinars can be found there as well.

https://drive.google.com/file/d/1t8WwPcnAsieVc-3jisJiQyw6jFBv4Hmb/view?usp=sharing


#### SDSC On-Demand Learning
- Archive of webinars, educational videos, github repos and other educational resources related to the SDSC.

https://www.sdsc.edu/education/on-demand-learning/index.html


## Previous and current ELP Research, as well as related research with Dr. Siewert

#### 2024-2025 ELP CNN vs RNN research from which this repo builds upon:
https://www.ecst.csuchico.edu/~sbsiewert/extra/research/elephant/SSIF-2025-ELP-Presentation.pdf

#### Other research:
https://sites.google.com/csuchico.edu/research/home

 