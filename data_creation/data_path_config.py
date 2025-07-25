# Configuration used for data pre-processing
import os
from pathlib import Path
from dotenv import load_dotenv

# Get the root path from the environment.
# os.getenv() returns None if the variable isn't found.
load_dotenv()
CORNELL_DATA_ROOT = os.getenv("CORNELL_DATA_ROOT")
ENVIRONMENT = os.getenv("ENVIRONMENT")

# Add a check to make sure the path was found
if ENVIRONMENT == "local" and not CORNELL_DATA_ROOT:
    print("❌ Error: CORNELL_DATA_ROOT not found.")
    print("Please create a .env file in the project root and add the line:")
    print('CORNELL_DATA_ROOT="/path/to/your/data"')
    exit() # Stop the script if the path isn't configured

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class DataPathConfig(object):

    if ENVIRONMENT == "local":
    
        # ------------- Raw Input Data Paths -------------
        # Positive Raw Data Paths - Train
        POS_TRAIN_VAL1_METADATA_DIR = os.path.join(CORNELL_DATA_ROOT, "Rumble/Training/pnnn")
        POS_TRAIN_VAL1_SOUNDS_DIR = os.path.join(CORNELL_DATA_ROOT, "Rumble/Training/Sounds")
        POS_TRAIN_VAL2_METADATA_DIR = os.path.join(CORNELL_DATA_ROOT, "Rumble/Testing/PNNN")
        POS_TRAIN_VAL2_SOUNDS_DIR = os.path.join(CORNELL_DATA_ROOT, "Rumble/Testing/PNNN/Sounds")

        # Positive Raw Data Paths - Holdout Testing
        POS_HOLDOUT_TEST_METADATA_DIR = os.path.join(CORNELL_DATA_ROOT, "Rumble/Testing/Dzanga")
        POS_HOLDOUT_TEST_SOUNDS_DIR = os.path.join(CORNELL_DATA_ROOT, "Rumble/Testing/Dzanga/Sounds")

        # Negative Raw Data Path (Single Source)
        NEG_SOURCE_INPUT_DIR = os.path.join(CORNELL_DATA_ROOT, "Gunshot/Testing/PNNN/Sounds")

    
    
    # ------------- Preprocessed Clips Paths -------------
    # Positive Preprocessed Clips
    POS_TRAIN_VAL_CLIPS_DIR = os.path.join(PROJECT_ROOT, "data", "clips_train_val", "pos_pnnn_clips")
    POS_HOLDOUT_TEST_CLIPS_DIR = os.path.join(PROJECT_ROOT, "data", "clips_holdout_test", "pos_dzanga_clips")
    
    # Negative Preprocessed Clips
    TRAIN_VAL_NEG_CLIPS_DIR = os.path.join(PROJECT_ROOT, "data", "clips_train_val", "neg_pnnn_gunshot_clips")
    HOLDOUT_TEST_NEG_CLIPS_DIR = os.path.join(PROJECT_ROOT, "data", "clips_holdout_test", "neg_pnnn_gunshot_clips")

    # ------------- TFRecords Paths -------------
    # Path for TFRecords Audio Data
    TFRECORDS_AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "tfrecords_audio")

    # Path for TFRecords Spectrogram Data
    TFRECORDS_SPECTROGRAM_DIR = os.path.join(PROJECT_ROOT, "data", "tfrecords_spectrogram")