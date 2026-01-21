import gdown
import os
import zipfile
from pathlib import Path

def download_and_setup_data(url):
    """
    Downloads and sets up the 'assessment' dataset.

    - Downloads a zip file from Google Drive.
    - Extracts it into the 'data/assessment' folder.
    - Removes the zip after extraction.

    Args:
        url (str): The Google Drive sharing URL of the dataset.
    """
    # Base path is the root of Assignment-3/
    base_path = Path(__file__).parent.parent
    data_dir = base_path / "data" # Where we'll store all data
    assessment_dir = data_dir / "cropped_flowers" # The final folder for the assessment dataset
    zip_path = data_dir / "cropped_flowers.zip" # Temporary path for the zip file

    # If the folder already exists, skip downloading
    if assessment_dir.exists():
        print(f"Data already exists at {assessment_dir}. Skipping download.")
        return

    # Make sure the 'data' directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading assessment.zip from Google Drive...")
    try:
        # Download the zip file. `fuzzy=True` lets gdown parse the file ID from the URL automatically.
        gdown.download(url, str(zip_path), quiet=False, fuzzy=True)

        # Extract the zip contents into the data directory
        print("Extracting files to 'data/assessment'...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Clean up by removing the zip file
        os.remove(zip_path)
        print(f"\nSuccess! Data structure is ready at: {assessment_dir}")

    except Exception as e:
        print(f"\nAn error occurred while downloading or extracting the data: {e}")
        print("Make sure your internet connection is stable and the link is publicly accessible.")

if __name__ == "__main__":
    ########################################################################
    # Insert here the Google Drive link to the assessment dataset zipped !
    ########################################################################
    URL = "https://drive.google.com/file/d/1WhtUfCc6gGqNim3vtpCJjzAO0L-Wtpkt/view?usp=share_link"
    download_and_setup_data(URL)