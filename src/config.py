import os

# Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Create directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Google Drive File IDs
DATASET_GDRIVE_ID = "1-EhN3JREEsGvcCj2R5LBJ2eczSPwth7A"
ID_FILES_ZIP_GDRIVE_ID = "1-Ar4UIMpGD3CF1omalpZVs5G5fJHMgfk"
ID_CSV_GDRIVE_ID = "1-EQN78KR6QOt_Xg7m7VQAW167XHWZsiq"
GENDER_TRAIN_CSV_ID = "1-1ZZ5FQujtj5SpvCGgvjEivOAiXuiNVj"
GENDER_TEST_CSV_ID = "1-3dfrkpD-Wm3cC-il3DMn-pV49AmMPBe"

# Audio Processing Parameters
SAMPLE_RATE = 44100
DURATION = 30  # seconds
HOP_LENGTH = 512
N_MFCC = 13
N_MELS = 128
SILENCE_THRESHOLD_DB = 20

# Feature Extraction Parameters
KEEP_RATIO = 0.25
MAX_DIMS = 15

# Model Parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0005
