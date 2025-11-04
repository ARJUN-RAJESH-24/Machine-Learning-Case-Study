# download_datasets.py
import os
import subprocess

DATA_DIR = "data"

# Replace these with actual Kaggle dataset slugs you want to download.
DATASETS = {
    "twitter": {
        "kaggle_slug": "kishan7/hatred-dataset",  # <-- replace with real slug
        "expected_files": []  # optional: list of files to extract/rename if needed
    },
    "reddit": {
        "kaggle_slug": "shubhendra/hate-speech-and-offensive-language-dataset",  # <-- replace
        "expected_files": []
    },
    "youtube": {
        "kaggle_slug": "skylion007/youtube-comments-classification",  # <-- replace
        "expected_files": []
    },
    "adult": {
        "kaggle_slug": "ashwiniyer176/adult-content-detection-dataset",  # <-- replace
        "expected_files": []
    }
}

def download_one(slug, dest=DATA_DIR):
    print(f"Downloading: {slug}")
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", dest, "--unzip"], check=True)
    print("Downloaded and unzipped.")

def download_all_datasets():
    os.makedirs(DATA_DIR, exist_ok=True)
    for key, info in DATASETS.items():
        try:
            download_one(info["kaggle_slug"], DATA_DIR)
        except subprocess.CalledProcessError as e:
            print(f"Failed to download {key} ({info['kaggle_slug']}): {e}")
            print("Make sure kaggle CLI is configured and slug is correct.")
    print("Download attempts finished. Please verify files in the data/ folder.")
