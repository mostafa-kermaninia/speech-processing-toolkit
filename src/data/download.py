import os
import gdown
import shutil
import zipfile
from src.config import RAW_DATA_DIR, PROJECT_ROOT

def download_file_from_google_drive(id, destination):
    """Downloads a file from Google Drive."""
    gdown.download(id=id, output=destination, quiet=False)

def download_dataset(file_id, output_path, extract_to=None):
    """Downloads and optionally extracts a dataset."""
    if not os.path.exists(output_path):
        print(f"Downloading dataset to {output_path}...")
        download_file_from_google_drive(file_id, output_path)
    else:
        print(f"Dataset already exists at {output_path}.")

    if extract_to:
        if not os.path.exists(extract_to):
            print(f"Extracting to {extract_to}...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        else:
             print(f"Dataset already extracted to {extract_to}.")

def setup_data(dataset_id, id_files_id, id_csv_id):
    """Sets up the data for the project."""
    
    # Main Audio Dataset
    zip_path = os.path.join(RAW_DATA_DIR, "HW1_M.zip")
    extract_path = os.path.join(RAW_DATA_DIR, "audio_files")
    download_dataset(dataset_id, zip_path, extract_to=extract_path)

    # Move extracted files to a cleaner path if needed
    # The original notebook had logic to move files from nested folders.
    # We will handle this in the dataset loading part or simplify here.
    nested_path = os.path.join(extract_path, "content", "drive", "MyDrive", "ML_final", "Copy of Audio_project", "HW1_M")
    final_audio_path = os.path.join(RAW_DATA_DIR, "HW1_M")
    
    if os.path.exists(nested_path) and not os.path.exists(final_audio_path):
        print(f"Moving files from {nested_path} to {final_audio_path}...")
        shutil.move(nested_path, final_audio_path)
        shutil.rmtree(os.path.join(extract_path, "content"))

    # ID Files
    id_zip_path = os.path.join(RAW_DATA_DIR, "ID_files.zip")
    id_extract_path = os.path.join(RAW_DATA_DIR, "ID_files")
    download_dataset(id_files_id, id_zip_path, extract_to=id_extract_path)

    # ID CSV
    id_csv_path = os.path.join(RAW_DATA_DIR, "student_ids.csv")
    download_file_from_google_drive(id_csv_id, id_csv_path)

if __name__ == "__main__":
    from src.config import DATASET_GDRIVE_ID, ID_FILES_ZIP_GDRIVE_ID, ID_CSV_GDRIVE_ID
    setup_data(DATASET_GDRIVE_ID, ID_FILES_ZIP_GDRIVE_ID, ID_CSV_GDRIVE_ID)
